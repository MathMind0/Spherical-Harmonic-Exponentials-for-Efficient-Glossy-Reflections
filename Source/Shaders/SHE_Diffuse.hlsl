#include "../CPUAndGPUCommon.h"
#include "Common.hlsli"
#include "SHE_Math.hlsli"

// Project the environment map onto 3-band SH (9 coefficients per color channel),
// stage 1 of 2: parallel partial projection.
//
// Dispatch: NumGroups groups, one wave per group (32 or 64 threads depending
// on the GPU wave size, selected by the CPU via the appropriate PSO variant).
// Each thread integrates 16 Hammersley samples into a local accumulator and
// the whole group is reduced with WaveActiveSum. The group's lane 0 writes
// one partial result (9 x float3) into GPartialRadiance.
//
// Stage 2 (SHE_Diffuse_Finalize.hlsl) sums the partials, applies the 4*pi/N
// normalization and the Ramamoorthi cosine-band convolution, and writes the
// final coefficients into slots [34..42] of GSHESHCoeff.

#define GRootSignature \
    "RootFlags(0), " \
    "DescriptorTable(CBV(b0), SRV(t0), UAV(u0), visibility = SHADER_VISIBILITY_ALL), " \
    "StaticSampler(" \
        "s0, " \
        "filter = FILTER_MIN_MAG_LINEAR_MIP_POINT, " \
        "visibility = SHADER_VISIBILITY_ALL, " \
        "addressU = TEXTURE_ADDRESS_BORDER, " \
        "addressV = TEXTURE_ADDRESS_BORDER, " \
        "addressW = TEXTURE_ADDRESS_BORDER)"

TextureCube GEnvMap : register(t0);
SamplerState GSampler : register(s0);

RWStructuredBuffer<float3> GPartialRadiance : register(u0);

cbuffer GDiffuseParams : register(b0)
{
    FSHEDiffuseParams GParams;
}

[RootSignature(GRootSignature)]
[numthreads(GROUP_SIZE_X, 1, 1)]
void MainCS(uint3 GroupID : SV_GroupID, uint LaneIndex : SV_GroupThreadID)
{
    const uint SamplesPerGroup = GParams.SamplesPerThread * GParams.ThreadCountX;

    float3 SHRadiance[9] = (float3[9])0;

    // Hammersley samples owned by this thread: a contiguous chunk so that
    // consecutive lanes touch consecutive (low-discrepancy) sample indices,
    // which keeps the per-group sample set identical to the serial version.
    const uint FirstSample = GroupID.x * SamplesPerGroup + LaneIndex * GParams.SamplesPerThread;

    [unroll]
    for (uint LocalIdx = 0; LocalIdx < SAMPLES_PER_THREAD; ++LocalIdx)
    {
        const uint SampleIndex = FirstSample + LocalIdx;
        if (SampleIndex >= GParams.TotalSampleCount)
        {
            break;
        }

        float2 Xi = Hammersley(SampleIndex, GParams.TotalSampleCount);
        float Phi = 2.0f * PI * Xi.x;
        float CosTheta = 1.0f - 2.0f * Xi.y;
        float SinTheta = sqrt(saturate(1.0f - CosTheta * CosTheta));

        float3 Direction = float3(
            SinTheta * cos(Phi),
            SinTheta * sin(Phi),
            CosTheta);

        float3 Radiance = GEnvMap.SampleLevel(GSampler, Direction, 0).rgb;

        // Evaluate the 3-band SH basis at the sample direction.
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

    // Reduce across the whole group (one wave): no shared memory, no barrier.
    [unroll]
    for (int CoeffIndex = 0; CoeffIndex < 9; ++CoeffIndex)
    {
        SHRadiance[CoeffIndex] = WaveActiveSum(SHRadiance[CoeffIndex]);
    }

    // Lane 0 writes the group partial: 9 consecutive float3 elements.
    if (LaneIndex == 0)
    {
        [unroll]
        for (int CoeffIndex = 0; CoeffIndex < 9; ++CoeffIndex)
        {
            GPartialRadiance[GroupID.x * 9 + CoeffIndex] = SHRadiance[CoeffIndex];
        }
    }
}
