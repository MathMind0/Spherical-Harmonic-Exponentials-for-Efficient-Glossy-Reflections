#include "../CPUAndGPUCommon.h"
#include "Common.hlsli"
#include "SHE_Math.hlsli"

// Project the environment map onto 3-band SH (9 coefficients per color channel)
// convolved with the clamped cosine lobe (Ramamoorthi 2001), so that the
// diffuse irradiance is a simple dot product of 9 coefficients with the
// SH basis evaluated at the surface normal.
//
// Result layout in GSHESHCoeff (float4 each):
//   [34 .. 42] : RGB diffuse irradiance SH coefficients, w unused
//   [43]       : unused
//
// Slots [0..32] hold the SHE specular coefficients and [33] the auto bias;
// they are written by SHE_Solve/SHE_Calibrate and must not be touched here.
//
// The pass writes 10 float4s, i.e. one thread.

#define GRootSignature \
    "RootFlags(0), " \
    "DescriptorTable(SRV(t0), UAV(u0), visibility = SHADER_VISIBILITY_ALL), " \
    "StaticSampler(" \
        "s0, " \
        "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT, " \
        "visibility = SHADER_VISIBILITY_ALL, " \
        "addressU = TEXTURE_ADDRESS_BORDER, " \
        "addressV = TEXTURE_ADDRESS_BORDER, " \
        "addressW = TEXTURE_ADDRESS_BORDER)"

TextureCube GEnvMap : register(t0);
SamplerState GSampler : register(s0);

RWStructuredBuffer<float4> GSHESHCoeff : register(u0);

[RootSignature(GRootSignature)]
[numthreads(1, 1, 1)]
void MainCS()
{
    // Clamped cosine lobe convolved in SH space (Ramamoorthi 2001):
    // E(N) = sum_lm A_l * L_lm * y_lm(N), with L_lm = int L(s) y_lm(s) ds.
    const float IrradianceBandA[3] =
    {
        PI,             // A_0
        2.0f * PI / 3.0f, // A_1
        PI / 4.0f,      // A_2
    };

    float3 SHRadiance[9] = (float3[9])0;

    // Monte Carlo integration over the sphere with uniform sampling.
    // Estimator of a projection coefficient: E[ L(s) * y_j(s) ] with pdf 1/4pi.
    const uint NumSamples = 65536;
    for (uint SampleIndex = 0; SampleIndex < NumSamples; ++SampleIndex)
    {
        float2 Xi = Hammersley(SampleIndex, NumSamples);
        float Phi = 2.0f * PI * Xi.x;
        float CosTheta = 1.0f - 2.0f * Xi.y;
        float SinTheta = sqrt(saturate(1.0f - CosTheta * CosTheta));

        float3 Direction = float3(
            SinTheta * cos(Phi),
            SinTheta * sin(Phi),
            CosTheta);

        float3 Radiance = GEnvMap.SampleLevel(GSampler, Direction, 0).rgb;

        // Evaluate the 3-band SH basis at the sample direction (index 0..8).
        // Basis ordering matches FThreeBandSHVector / SHBasisFunction3 in SHE_Math.hlsli.
        float Y[9];
        {
            FThreeBandSHVector Y3 = SHBasisFunction3(half3(Direction));

            Y[0] = Y3.V0.x;
            Y[1] = Y3.V0.y;
            Y[2] = Y3.V0.z;
            Y[3] = Y3.V0.w;
            Y[4] = Y3.V1.x;
            Y[5] = Y3.V1.y;
            Y[6] = Y3.V1.z;
            Y[7] = Y3.V1.w;
            Y[8] = Y3.V2;
        }

        [unroll]
        for (int CoeffIndex = 0; CoeffIndex < 9; ++CoeffIndex)
        {
            SHRadiance[CoeffIndex] += Radiance * Y[CoeffIndex];
        }
    }

    [unroll]
    for (int CoeffOut = 0; CoeffOut < 9; ++CoeffOut)
    {
        // Uniform sphere sampling has pdf 1/(4*pi), so the projection
        // integral int L(s)*y_j(s) ds = mean(L*y_j) * 4*pi.
        float3 c = SHRadiance[CoeffOut] * (4.0f * PI / float(NumSamples));

        // Band index of this coefficient (0, 0, 0, 1, 1, 1, 2, 2, 2).
        const int BandIndex[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};

        // Convolve with the clamped cosine lobe in SH space so that the
        // reconstruction sum_j (A_lj * c_j) * y_j(N) yields irradiance E(N).
        float3 IrradianceCoeff = c * IrradianceBandA[BandIndex[CoeffOut]];

        // Slots [34..42] follow the 33 SHE specular coefficients + 1 auto bias.
        GSHESHCoeff[34 + CoeffOut] = float4(IrradianceCoeff, 0.0f);
    }

    GSHESHCoeff[43] = float4(0.0f, 0.0f, 0.0f, 0.0f);
}
