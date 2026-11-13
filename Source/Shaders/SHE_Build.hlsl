#include "../CPUAndGPUCommon.h"
#include "Common.hlsli"
#include "SHE_Math.hlsli"
#include "SHBasis.h"

#define GRootSignature \
    "RootFlags(0), " \
    "DescriptorTable(CBV(b0), SRV(t0), UAV(u0), UAV(u1), visibility = SHADER_VISIBILITY_ALL), " \
    "StaticSampler(" \
        "s0, " \
        "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT, " \
        "visibility = SHADER_VISIBILITY_ALL, " \
        "addressU = TEXTURE_ADDRESS_BORDER, " \
        "addressV = TEXTURE_ADDRESS_BORDER, " \
        "addressW = TEXTURE_ADDRESS_BORDER)" 

ConstantBuffer<FSHEBuildConstantData> GSHEBuildCB : register(b0);
TextureCube GEnvMap : register(t0);
SamplerState GSampler : register(s0);

RWTexture2D<float> GSHEMatrixA : register(u0);
RWTexture2D<float4> GSHEMatrixb : register(u1);

[RootSignature(GRootSignature)]
[numthreads(8, 8, 1)]
void MainCS(
    uint3 ThreadID : SV_DispatchThreadID,
    uint3 GroupThreadID : SV_GroupThreadID,
    uint3 GroupID : SV_GroupID,
    uint GroupIndex : SV_GroupIndex
)
{
    // get view, normal and roughness indices
    int ViewIndex = ThreadID.x;
    const int NormalIndex = ThreadID.y;
    const int RoughnessIndex = ThreadID.z;
    if (ViewIndex >= GSHEBuildCB.ViewCount || NormalIndex >= GSHEBuildCB.NormalCount || RoughnessIndex >= GSHEBuildCB.RoughnessCount)
    {
        return;
    }

    // get normal ws
    const float2 NormalIndex2D = (float2(NormalIndex % GSHEBuildCB.NormalCountSqrt, NormalIndex / GSHEBuildCB.NormalCountSqrt) + 0.5f) / (GSHEBuildCB.NormalCountSqrt);
    const float3 N = EquiAreaSphericalMapping(NormalIndex2D);

    // get view ws
    const float2 ViewIndex2D = (float2(ViewIndex % GSHEBuildCB.ViewCountSqrt, ViewIndex / GSHEBuildCB.ViewCountSqrt) + 0.5f) / GSHEBuildCB.ViewCountSqrt; // avoid situations where y is 0 or 1
    const float3 V = UniformSampleHemisphere(ViewIndex2D, N);

    // get roughness
    float Roughness;
    if (GSHEBuildCB.bRandomRoughness)
    {
        // "Random Roughness": instead of pinning every row of a group to one
        // fixed r_g, give each (view, normal, group) row its own deterministic
        // pseudo-random roughness drawn from that group stratified sub-band.
        // Stratifying by RoughnessIndex keeps the group anchored to a roughness
        // band (so the LUT interpolation stays meaningful), while the union of
        // all rows covers the whole [Start, End] range continuously. The same
        // Roughness feeds both the vMF features (A) and GGX integration (b)
        // below, so every row stays self-consistent.
        const uint2 Seed = uint2((uint)ViewIndex * 64u + (uint)NormalIndex, (uint)RoughnessIndex);
        const uint Hash = HashPixelAndSample(Seed, 0x9e3779b9u);
        const float u = (float)Hash * 2.3283064365386963e-10f; // [0, 1)
        const float Ratio = ((float)RoughnessIndex + u) / (float)GSHEBuildCB.RoughnessCount;
        Roughness = lerp(GSHEBuildCB.RoughnessStart, GSHEBuildCB.RoughnessEnd, Ratio);
    }
    else
    {
        const float RoughnessRatio = GSHEBuildCB.RoughnessCount > 1 ? (float)RoughnessIndex / (GSHEBuildCB.RoughnessCount - 1) : 0.0f;
        Roughness = lerp(GSHEBuildCB.RoughnessStart, GSHEBuildCB.RoughnessEnd, RoughnessRatio);
    }

    // calculate b
    float NoV = saturate(dot(N, V));
    const uint NumSamples = 1024;
    float3 E0 = 0.0f;
    for (uint i = 0; i < NumSamples; i++)
    {
        float2 Xi = Hammersley(i, NumSamples);
        float3 H = ImportanceSampleGGX(Xi, Roughness, N);
        float3 L = normalize(2.0f * dot(V, H) * H - V);

        float NoL = saturate(dot(N, L));
        float NoH = saturate(dot(N, H));
        float VoH = saturate(dot(V, H));

        if (NoL > 0.0f)
        {     
            float G = GeometrySmith(NoL, NoV, Roughness);
            float G_Vis = G * VoH / (NoH * NoV + 1e-5);
            // GGX NDF sampling of H: estimator is Li * G * VoH / (NoH * NoV), no extra NoL.
            E0 += G_Vis * GEnvMap.SampleLevel(GSampler, L, 0).rgb;
        }
    }

    E0 /= NumSamples;
	E0 = log(max(E0, 1e-6f));

    // calculate A: build the SHE feature vector into the fixed spec-coefficient
    // slots using the shared real-SH basis (up to 10th order). The P section
    // uses Y_k(R) for k=0..99, the Q section uses Y_k(Hr) for k=1..24 (Q drops
    // the DC term). Only slots within the selected P/Q orders are active; the
    // rest are zero, so their solved coefficients stay exactly 0 and the
    // reduction/solve/calibrate stages can keep a constant dimension.
    const float3 R = reflect(-V, N);
    const float3 Hr = normalize(N + R);

    float YR[SH_CLASSIC_COEFF_COUNT];
    float YH[SH_CLASSIC_COEFF_COUNT];
    SHBasisEval(R.x, R.y, R.z, YR);
    SHBasisEval(Hr.x, Hr.y, Hr.z, YH);

    const int POrder = GSHEBuildCB.POrder;
    const int QOrder = GSHEBuildCB.QOrder;
    const int PActive = POrder * POrder;   // active P coefficients
    const int QActive = QOrder * QOrder;   // Q k range is 1 .. QActive-1

    const int height = NormalIndex + RoughnessIndex * GSHEBuildCB.NormalCount;
    GSHEMatrixb[uint2(ViewIndex, height)] = float4(E0.x, E0.y, E0.z, 1.0f);

    ViewIndex *= GSHEBuildCB.SphericalHarmonicCount;

    // P section: slots [0 .. SHE_P_COEFFS-1], feature = vMF(band(k)) * Y_k(R).
    for (int pk = 0; pk < SHE_P_COEFFS; ++pk)
    {
        const float w = SHEVMFWeight(SHEBandOf(pk), Roughness);
        GSHEMatrixA[uint2(ViewIndex + pk, height)] = pk < PActive ? YR[pk] * w : 0.0f;
    }
    // Q section: slots [SHE_Q_OFFSET .. SHE_Q_OFFSET+SHE_Q_COEFFS-1]. Slot
    // (SHE_Q_OFFSET + j) corresponds to flat SH index k = j+1 (DC dropped),
    // feature = vMF(band(k)) * Y_k(Hr).
    for (int qj = 0; qj < SHE_Q_COEFFS; ++qj)
    {
        const int k = qj + 1;
        const float w = SHEVMFWeight(SHEBandOf(k), Roughness);
        const bool bActive = k < QActive;
        GSHEMatrixA[uint2(ViewIndex + SHE_Q_OFFSET + qj, height)] = bActive ? YH[k] * w : 0.0f;
    }
}
