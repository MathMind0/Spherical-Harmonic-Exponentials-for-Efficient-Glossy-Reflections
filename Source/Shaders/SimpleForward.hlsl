#include "../CPUAndGPUCommon.h"
#include "Common.hlsli"
#include "SHE_Math.hlsli"
#include "SHBasis.h"

#define GRootSignature                                                                                                       \
	"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), "                                                                        \
	"CBV(b0), "                                                                                                              \
	"DescriptorTable(CBV(b1), CBV(b2), SRV(t0), SRV(t1), SRV(t2), SRV(t3), SRV(t4), SRV(t5), SRV(t6), SRV(t7), visibility = SHADER_VISIBILITY_PIXEL), " \
	"StaticSampler("                                                                                                         \
	"s0, "                                                                                                                   \
	"filter = FILTER_MIN_MAG_MIP_LINEAR, "                                                                                   \
	"visibility = SHADER_VISIBILITY_PIXEL, "                                                                                 \
	"addressU = TEXTURE_ADDRESS_BORDER, "                                                                                    \
	"addressV = TEXTURE_ADDRESS_BORDER, "                                                                                    \
	"addressW = TEXTURE_ADDRESS_BORDER)"

ConstantBuffer<FPerDrawConstantData> GPerDrawCB : register(b0);
ConstantBuffer<FPerFrameConstantData> GPerFrameCB : register(b1);
ConstantBuffer<FSHECoefficientConstantData> GSHECoeffCB : register(b2);
TextureCube GIrradianceMap : register(t0);
TextureCube GPrefilteredEnvMap : register(t1);
TextureCube GEnvMap : register(t2);
Texture2D GBRDFIntegrationMap : register(t3);
Texture2D GAccumulationBuffer : register(t4);
StructuredBuffer<float4> GCoeffLUT : register(t5);
StructuredBuffer<float4> GClassicSHCoeff : register(t6);
// SG lobes: two float4 per lobe. [2i] = (mu.xyz, lambda), [2i+1] = (a.rgb, 0).
StructuredBuffer<float4> GSGLobes : register(t7);
SamplerState GSampler : register(s0);

float3 FresnelSchlick(float CosTheta, float3 F0)
{
	return F0 + (1.0f - F0) * pow(1.0f - CosTheta, 5.0f);
}

float3 FresnelSchlickRoughness(float CosTheta, float3 F0, float Roughness)
{
	return F0 + (max(1.0f - Roughness, F0) - F0) * pow(1.0f - CosTheta, 5.0f);
}

// Exact spherical integral of the product of two isotropic Spherical Gaussians:
//   G(v)  = a   * exp(lambda  * (dot(mu , v) - 1))
//   K(v)  = aK  * exp(lambdaK * (dot(muK, v) - 1))   (aK is scalar)
// The product is another SG with
//   mu'  = (lambda*mu + lambdaK*muK) / lam,  lam = |lambda*mu + lambdaK*muK|,
//   a'   = a * aK * exp(lam - lambda - lambdaK),
// and int_{S^2} G'(v) dv = 2*pi*a'*(1 - exp(-2*lam))/lam  (-> 4*pi*a' as lam->0).
// Used to convolve the fitted environment lobes with a cosine SG (diffuse) or
// a GGX-as-SG lobe (specular) entirely in closed form.
float3 SGProductIntegral(float3 mu, float lambda, float3 a,
						 float3 muK, float lambdaK, float aK)
{
	const float d = dot(mu, muK);
	const float lam = sqrt(lambdaK * lambdaK + lambda * lambda
						   + 2.0f * lambdaK * lambda * d);
	// Stable form of (lam - lambda - lambdaK). Direct subtraction loses all
	// precision when lambdaK is large (mirror-like GGX lobes at low roughness).
	const float delta = (lambda * lambda + 2.0f * lambdaK * lambda * d)
						/ (lam + lambdaK) - lambda;
	const float3 aProd = a * (aK * exp(delta));
	const float integ = lam > 1e-4f
		? (2.0f * PI * (1.0f - exp(-2.0f * lam)) / lam)
		: (4.0f * PI);
	return aProd * integ;
}

[RootSignature(GRootSignature)] void MainVS(
	in float3 InPosition : _Position,
	in float3 InNormal : _Normal,
	out float4 OutPosition : SV_Position,
	out float3 OutPositionWS : _Position,
	out float3 OutNormalWS : _Normal)
{
	OutPosition = mul(float4(InPosition, 1.0f), GPerDrawCB.ObjectToClip);
	OutPositionWS = mul(float4(InPosition, 1.0f), GPerDrawCB.ObjectToWorld);
	OutNormalWS = mul(InNormal, (float3x3)GPerDrawCB.ObjectToWorld);
}

	[RootSignature(GRootSignature)] void MainPS(
		in float4 InPosition : SV_Position,
		in float3 InPositionWS : _Position,
		in float3 InNormalWS : _Normal,
		out float4 OutColor : SV_Target0)
{
	// Orthographic projection: all view rays are parallel to the camera forward
	// direction, so V is constant per pixel. ViewDirection points from the focus
	// (roughly the sphere plane) toward the camera eye, which is exactly the
	// surface-to-viewer direction V. Perspective projection: rays converge at
	// the viewer position.
	float3 V;
	if (GPerFrameCB.bOrthographic)
	{
		V = normalize(GPerFrameCB.ViewDirection.xyz);
	}
	else
	{
		V = normalize(GPerFrameCB.ViewerPosition.xyz - InPositionWS);
	}
	float3 N = normalize(InNormalWS);
	float NoV = saturate(dot(N, V));

	float3 Albedo = GPerDrawCB.Albedo;
	float Roughness = GPerDrawCB.Roughness;
	float Metallic = GPerDrawCB.Metallic;
	float AO = GPerDrawCB.AO;

	float3 F0 = float3(0.04f, 0.04f, 0.04f);
	F0 = lerp(F0, Albedo, Metallic);

	float3 Lo = 0.0f;
	// We manually disable point lights, only use image based lighting.
	for (int LightIdx = 0; LightIdx < 0; ++LightIdx)
	{
		float3 LightVector = GPerFrameCB.LightPositions[LightIdx].xyz - InPositionWS;

		float3 L = normalize(LightVector);
		float3 H = normalize(L + V);
		float NoL = saturate(dot(N, L));
		float HoV = saturate(dot(H, V));

		float Attenuation = 1.0f / dot(LightVector, LightVector);
		float3 Radiance = GPerFrameCB.LightColors[LightIdx].rgb * Attenuation;

		float3 F = FresnelSchlick(HoV, F0);

		float NDF = DistributionGGX(N, H, Roughness);
		float G = GeometrySmith(NoL, NoV, (Roughness + 1.0f) * 0.5f);

		float3 Specular = (NDF * G * F) / max(4.0f * NoV * NoL, 0.001f);

		float3 KS = F;
		float3 KD = 1.0f - KS;
		KD *= 1.0f - Metallic;

		Lo += (KD * (Albedo / PI) + Specular) * Radiance * NoL;
	}

	float3 R = reflect(-V, N);

	float3 F = FresnelSchlickRoughness(NoV, F0, Roughness);

	float3 KD = 1.0f - F;
	KD *= 1.0f - Metallic;

	float3 Diffuse = 0.0f;
	float3 Specular = 0.0f;
	if (GPerFrameCB.IBLMode == IBL_MODE_SPLIT_SUM_APPROXIMATION)
	{
		float3 Irradiance = GIrradianceMap.SampleLevel(GSampler, N, 0.0f).rgb;
		Diffuse = Irradiance * Albedo;

		float3 PrefilteredColor = GPrefilteredEnvMap.SampleLevel(GSampler, R, Roughness * 5.0f).rgb;
		float2 EnvBRDF = GBRDFIntegrationMap.SampleLevel(GSampler, float2(min(NoV, 0.999f), min(Roughness, 0.999f)), 0.0f).rg;
		Specular = PrefilteredColor * (F0 * EnvBRDF.x + EnvBRDF.y);
	}
	else if (GPerFrameCB.IBLMode == IBL_MODE_SPHERICAL_GAUSSIAN)
	{
		// Spherical Gaussian IBL. The environment radiance is the sum of
		// isotropic SGs a_i * exp(lambda_i * (dot(mu_i, v) - 1)), fitted on
		// the CPU. Two zonal SG kernels are convolved with those lobes via the
		// closed-form SG product integral:
		//   - cosine SG (mu = N)          -> irradiance E(N) for diffuse,
		//   - GGX NDF SG (mu = R)         -> prefiltered radiance for specular.
		const int LobeCount = clamp(GPerFrameCB.SGLobeCount, 1, SG_MAX_LOBES);

		// Cosine-kernel SG: sharpness tuned to the clamped-cosine lobe, with
		// amplitude calibrated so a constant environment L yields E = PI * L
		// (exact DC term). Diffuse below divides by PI, matching Split-Sum.
		const float lambdaCos = 2.0f;
		const float aCos = lambdaCos / (2.0f * (1.0f - exp(-2.0f * lambdaCos)));

		// GGX NDF as an SG centered on R. alpha = roughness^2, SG sharpness
		// lambda = 2 / alpha^2. Amplitude normalizes the lobe to integrate to
		// 1 over the sphere, so the convolution is a weighted radiance average.
		const float Alpha = max(Roughness * Roughness, 1e-3f);
		const float lambdaNDF = 2.0f / (Alpha * Alpha);
		const float aNDF = lambdaNDF / (2.0f * PI * (1.0f - exp(-2.0f * lambdaNDF)));

		float3 Irradiance = 0.0f;
		float3 PrefilteredColor = 0.0f;

		[loop]
		for (int i = 0; i < LobeCount; ++i)
		{
			const float4 AxisLambda = GSGLobes[2 * i];       // (mu.xyz, lambda)
			const float4 Amplitude = GSGLobes[2 * i + 1];    // (a.rgb, unused)
			const float3 muI = AxisLambda.xyz;
			const float lambdaI = AxisLambda.w;
			const float3 aI = Amplitude.rgb;

			Irradiance += SGProductIntegral(muI, lambdaI, aI, N, lambdaCos, aCos);
			PrefilteredColor += SGProductIntegral(muI, lambdaI, aI, R, lambdaNDF, aNDF);
		}

		Diffuse = max(Irradiance, float3(0.0f, 0.0f, 0.0f)) * (Albedo * (1.0f / PI));
		PrefilteredColor = max(PrefilteredColor, float3(0.0f, 0.0f, 0.0f));

		float2 EnvBRDF = GBRDFIntegrationMap.SampleLevel(GSampler, float2(min(NoV, 0.999f), min(Roughness, 0.999f)), 0.0f).rg;
		Specular = PrefilteredColor * (F0 * EnvBRDF.x + EnvBRDF.y);
	}
	else if (GPerFrameCB.IBLMode == IBL_MODE_CLASSIC_SH)
	{
		// Standard (classic) SH IBL. One set of projected radiance coefficients
		// is convolved with two different zonal kernels:
		//   - clamped cosine -> diffuse irradiance E(N),
		//   - vMF / GGX band-pass -> glossy prefiltered radiance along R.
		// The "SH Order" UI slider truncates the band sum at l = 0..order-1,
		// the classic band-limiting / detail control.
		const int Order = clamp(GPerFrameCB.ClassicSHOrder, 2, SH_CLASSIC_MAX_ORDER);
		const int MaxL = Order - 1;

		// Zonal-harmonic coefficients k_l = 2*pi * integral_0^1 x P_l(x) dx
		// of the clamped-cosine kernel (PI factor already included).
		static const float CosKernel[SH_CLASSIC_MAX_ORDER] =
		{
			PI,                // l = 0
			2.0f * PI / 3.0f, // l = 1
			PI / 4.0f,        // l = 2
			0.0f,             // l = 3
			-PI / 24.0f,      // l = 4
			0.0f,             // l = 5
			PI / 64.0f,       // l = 6
			0.0f,             // l = 7
			-PI / 128.0f,     // l = 8
			0.0f              // l = 9
		};

		float BasisN[SH_CLASSIC_COEFF_COUNT];
		float BasisR[SH_CLASSIC_COEFF_COUNT];
		SHBasisEval(N.x, N.y, N.z, BasisN);
		SHBasisEval(R.x, R.y, R.z, BasisR);

		float3 Irradiance = 0.0f;
		float3 PrefilteredColor = 0.0f;

		[loop]
		for (int l = 0; l <= MaxL; ++l)
		{
			// vMF(l, 1/Roughness) = exp(-l(l+1) / (2 * (1/Roughness))).
			const float VmfBand = exp(-float(l * (l + 1)) * Roughness * 0.5f);
			[loop]
			for (int m = -l; m <= l; ++m)
			{
				const int k = l * l + l + m;
				const float3 Llm = GClassicSHCoeff[k].rgb;
				Irradiance += CosKernel[l] * Llm * BasisN[k];
				PrefilteredColor += VmfBand * Llm * BasisR[k];
			}
		}

		Diffuse = max(Irradiance, float3(0.0f, 0.0f, 0.0f)) * (Albedo * (1.0f / PI));

		float2 EnvBRDF = GBRDFIntegrationMap.SampleLevel(GSampler, float2(min(NoV, 0.999f), min(Roughness, 0.999f)), 0.0f).rg;
		Specular = PrefilteredColor * (F0 * EnvBRDF.x + EnvBRDF.y);
	}
	else if (GPerFrameCB.IBLMode == IBL_MODE_SPHERICAL_HARMONICS_EXPONENTIAL)
	{
		// SHE specular: log-space radiance = dot(coeffs, features) + autoBias,
		// where features is the fixed SHE_SPEC_COEFFS-slot basis vector built
		// from Y_k(R) (P section) and Y_k(Hr) (Q section), each weighted by
		// the vMF band weight. Only slots within the baked P/Q orders carry
		// non-zero features; the rest contribute nothing. This mirrors the
		// feature construction in SHE_Build.hlsl exactly.
		const float3 Hr = normalize(N + R);
		float YR[SH_CLASSIC_COEFF_COUNT];
		float YH[SH_CLASSIC_COEFF_COUNT];
		SHBasisEval(R.x, R.y, R.z, YR);
		SHBasisEval(Hr.x, Hr.y, Hr.z, YH);

		const int POrder = clamp(GPerFrameCB.SHEPOrder, 1, SHE_MAX_P);
		const int QOrder = clamp(GPerFrameCB.SHEQOrder, 1, SHE_MAX_Q);
		const int PActive = POrder * POrder;
		const int QActive = QOrder * QOrder;

		float3 features[SHE_SPEC_COEFFS];
		[loop] for (int pk = 0; pk < SHE_P_COEFFS; ++pk)
		{
			const float v = pk < PActive ? YR[pk] * SHEVMFWeight(SHEBandOf(pk), Roughness) : 0.0f;
			features[pk] = float3(v, v, v);
		}
		[loop] for (int qj = 0; qj < SHE_Q_COEFFS; ++qj)
		{
			const int k = qj + 1;
			const float v = k < QActive ? YH[k] * SHEVMFWeight(SHEBandOf(k), Roughness) : 0.0f;
			features[SHE_Q_OFFSET + qj] = float3(v, v, v);
		}

		half3 coeffs[SHE_SPEC_COEFFS];
		half3 autoBias;
		if (GPerFrameCB.bUseRoughnessLUT && GPerFrameCB.SHECoeffLUTCount > 0)
		{
			const float Denom = max(GPerFrameCB.SHECoeffLUTRoughnessEnd - GPerFrameCB.SHECoeffLUTRoughnessStart, 1e-5f);
			const float T = saturate((Roughness - GPerFrameCB.SHECoeffLUTRoughnessStart) / Denom) * (float)(GPerFrameCB.SHECoeffLUTCount - 1);
			const int I0 = (int)floor(T);
			const int I1 = min(I0 + 1, GPerFrameCB.SHECoeffLUTCount - 1);
			const float F = T - (float)I0;
			if (GPerFrameCB.bUsePCA && GPerFrameCB.SHEPCABasisCount > 0)
			{
				// PCA-compressed LUT (flat float4 stream, stride SHE_LUT_STRIDE):
				//   [0 .. S-1]                       : per-parameter mean (rgb)
				//   [S + k*S + i]                   : basis vector k, param i
				//   [S + NB*S + g*NB + k]          : weight of group g on basis k
				const int NB = GPerFrameCB.SHEPCABasisCount;
				const int S = SHE_LUT_STRIDE;
				const int WeightBase = S + NB * S;
				float W[16];
				[loop] for (int kb = 0; kb < NB; ++kb)
				{
					W[kb] = lerp(GCoeffLUT[WeightBase + I0 * NB + kb].x,
								 GCoeffLUT[WeightBase + I1 * NB + kb].x, F);
				}
				[loop] for (int i = 0; i < SHE_LUT_STRIDE; ++i)
				{
					float3 val = GCoeffLUT[i].rgb; // mean
					[loop] for (int kb = 0; kb < NB; ++kb)
					{
						val += W[kb] * GCoeffLUT[S + kb * S + i].rgb;
					}
					if (i < SHE_SPEC_COEFFS) coeffs[i] = (half3)val;
					else                     autoBias = (half3)val;
				}
			}
			else
			{
				const int Base0 = I0 * SHE_LUT_STRIDE;
				const int Base1 = I1 * SHE_LUT_STRIDE;
				[loop] for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
				{
					coeffs[i] = lerp(GCoeffLUT[Base0 + i].rgb, GCoeffLUT[Base1 + i].rgb, F);
				}
				autoBias = lerp(GCoeffLUT[Base0 + SHE_AUTOBIAS_SLOT].rgb, GCoeffLUT[Base1 + SHE_AUTOBIAS_SLOT].rgb, F);
			}
		}
		else
		{
			[loop] for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
			{
				coeffs[i] = GSHECoeffCB.Coeffs[i].rgb;
			}
			autoBias = GSHECoeffCB.Coeffs[SHE_AUTOBIAS_SLOT].rgb;
		}

		half3 logSpec = autoBias;
		[loop] for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
		{
			logSpec += coeffs[i] * features[i];
		}
		half3 E0 = exp(logSpec);

		half3 E1 = E0 * GBRDFIntegrationMap.SampleLevel(GSampler, float2(min(NoV, 0.999f), min(Roughness, 0.999f)), 0.0f).g;
		Specular = F0 * E0 + (1.0f - F0) * E1;

		// 3-band SH diffuse irradiance (convolved with the clamped cosine
		// lobe at precompute time, Ramamoorthi 2001). Dot the stored RGB
		// irradiance coefficients with the SH basis evaluated at N.
		FThreeBandSHVector yN = SHBasisFunction3(half3(N));
		FThreeBandSHVectorRGB DiffuseSH;
		{
			half3 d0 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 0].rgb;
			half3 d1 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 1].rgb;
			half3 d2 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 2].rgb;
			half3 d3 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 3].rgb;
			half3 d4 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 4].rgb;
			half3 d5 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 5].rgb;
			half3 d6 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 6].rgb;
			half3 d7 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 7].rgb;
			half3 d8 = GSHECoeffCB.Coeffs[SHE_DIFFUSE_BASE + 8].rgb;

			DiffuseSH.R.V0 = half4(d0.r, d1.r, d2.r, d3.r);
			DiffuseSH.R.V1 = half4(d4.r, d5.r, d6.r, d7.r);
			DiffuseSH.R.V2 = half(d8.r);

			DiffuseSH.G.V0 = half4(d0.g, d1.g, d2.g, d3.g);
			DiffuseSH.G.V1 = half4(d4.g, d5.g, d6.g, d7.g);
			DiffuseSH.G.V2 = half(d8.g);

			DiffuseSH.B.V0 = half4(d0.b, d1.b, d2.b, d3.b);
			DiffuseSH.B.V1 = half4(d4.b, d5.b, d6.b, d7.b);
			DiffuseSH.B.V2 = half(d8.b);
		}
		Diffuse = DotSH3(DiffuseSH, yN);
		Diffuse = max(Diffuse, half3(0.0h, 0.0h, 0.0h)) * half3(Albedo * (1.0f / PI));
	}
	else if (GPerFrameCB.IBLMode == IBL_MODE_REFERENCE)
	{
		uint NumSamples = 64;
		float3 Irradiance = 0.0f;

		for (uint DiffuseSampleIdx = 0; DiffuseSampleIdx < NumSamples; ++DiffuseSampleIdx)
		{
			uint finalSeed = HashPixelAndSample(uint2(InPosition.xy),
												DiffuseSampleIdx + GPerFrameCB.NumFrames * NumSamples);
			float2 Xi = HaltonSequence2D(finalSeed);
			float3 L = CosineSampleHemisphere(Xi, N);

			float NoL = saturate(dot(N, L));
			if (NoL > 0.0f)
			{
				// Standard way of cosine importance sampling, no need to multiply by NoL.
				Irradiance += GEnvMap.SampleLevel(GSampler, L, log2(max(256.0f * Roughness, 1.0f))).rgb;
			}
		}

		for (uint SpecularSampleIdx = 0; SpecularSampleIdx < NumSamples; ++SpecularSampleIdx)
		{
			uint finalSeed = HashPixelAndSample(uint2(InPosition.xy),
												SpecularSampleIdx + GPerFrameCB.NumFrames * NumSamples);
			float2 Xi = HaltonSequence2D(finalSeed);
			float3 H = ImportanceSampleGGX(Xi, Roughness, N);
			float3 L = normalize(2.0f * dot(V, H) * H - V);

			float NoL = saturate(dot(N, L));
			float NoH = saturate(dot(N, H));
			float VoH = saturate(dot(V, H));

			if (NoL > 0.0f)
			{
				float3 Li = GEnvMap.SampleLevel(GSampler, L, log2(max(256.0f * Roughness, 1.0f))).rgb;
				float G = GeometrySmith(NoL, NoV, Roughness);
				float G_Vis = G * VoH / (NoH * NoV);
				// GGX NDF sampling of H: the estimator is Li * F(VoH) * G * VoH / (NoH * NoV).
				// The NoL from f_spec * NoL cancels with the pdf; multiplying it back dims the result.
				float3 FH = FresnelSchlick(VoH, F0);
				Specular += Li * FH * G_Vis;
			}
		}

		Diffuse = (Irradiance * Albedo) * (1.0f / NumSamples);
		Specular *= (1.0f / NumSamples);
	}

	float3 Ambient = 0.0f;
	if (GPerFrameCB.MaterialMode == MATERIAL_MODE_DIFFUSE_AND_SPECULAR)
	{
		Ambient = KD * Diffuse + Specular;
	}
	else if (GPerFrameCB.MaterialMode == MATERIAL_MODE_DIFFUSE_ONLY)
	{
		Ambient = KD * Diffuse;
	}
	else if (GPerFrameCB.MaterialMode == MATERIAL_MODE_SPECULAR_ONLY)
	{
		Ambient = Specular;
	}
	Ambient *= AO;

	float3 Color = Ambient + Lo;

	if (GPerFrameCB.IBLMode == IBL_MODE_REFERENCE && GPerFrameCB.NumFrames > 0)
	{
		// Progressive rendering for reference mode
		uint2 Dimensions;
		GAccumulationBuffer.GetDimensions(Dimensions.x, Dimensions.y);
		float3 AccumulatedColor = GAccumulationBuffer.SampleLevel(GSampler, InPosition.xy / Dimensions, 0).rgb;
		float3 AccumulatedLinearColor = pow(AccumulatedColor, 2.2f);
		float3 AccumulatedHDRColor = AccumulatedLinearColor / (1.0f - AccumulatedLinearColor);
		Color = (AccumulatedHDRColor * (GPerFrameCB.NumFrames - 1) + Color) / GPerFrameCB.NumFrames;
	}

	Color = Color / (Color + 1.0f);
	Color = pow(Color, 1.0f / 2.2f);
	OutColor = float4(Color, 1.0f);
}
