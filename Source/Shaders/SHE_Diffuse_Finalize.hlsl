#include "../CPUAndGPUCommon.h"
#include "Common.hlsli"

// Stage 2 of 2: finalize the 3-band SH diffuse projection.
//
// Dispatch: a single group (one wave, 32 or 64 threads depending on the GPU).
// Each thread strided-reads a subset of the partial sums from stage 1,
// accumulates them locally, and the group is reduced with WaveActiveSum.
// Lane 0 applies the 4*pi/N Monte Carlo normalization, multiplies in the
// Ramamoorthi clamped-cosine band convolution coefficients A_l
// {pi, 2pi/3, pi/4}, and writes the final coefficients into slots [34..42]
// of GSHESHCoeff. Slot [43] is zeroed as unused.
//
// Result layout in GSHESHCoeff (float4 each):
//   [34 .. 42] : RGB diffuse irradiance SH coefficients, w unused
//   [43]       : unused (zeroed)
//
// Slots [0..32] hold the SHE specular coefficients and [33] the auto bias;
// they are written by SHE_Solve/SHE_Calibrate and must not be touched here.

#define GRootSignature \
    "RootFlags(0), " \
    "DescriptorTable(CBV(b0), SRV(t0), UAV(u0), visibility = SHADER_VISIBILITY_ALL)"

StructuredBuffer<float3> GPartialRadiance : register(t0);

RWStructuredBuffer<float4> GSHESHCoeff : register(u0);

cbuffer GDiffuseParams : register(b0)
{
    FSHEDiffuseParams GParams;
}

[RootSignature(GRootSignature)]
[numthreads(GROUP_SIZE_X, 1, 1)]
void MainCS(uint LaneIndex : SV_GroupThreadID)
{
    // Total number of stage-1 partial results (one per group).
    const uint PartialCount = GParams.TotalSampleCount / (GParams.SamplesPerThread * GParams.ThreadCountX);

    // Each lane accumulates partials at stride ThreadCountX: lane i sums
    // GPartialRadiance[group * 9 + coeff] for group == i, i + ThreadCountX, ...
    // PartialIdx % 9 recovers the coefficient index from the flat layout.
    float3 SHRadiance[9] = (float3[9])0;

    for (uint PartialIdx = LaneIndex; PartialIdx < PartialCount * 9; PartialIdx += GParams.ThreadCountX)
    {
        SHRadiance[PartialIdx % 9] += GPartialRadiance[PartialIdx];
    }

    // Reduce across the whole wave.
    [unroll]
    for (int CoeffIndex = 0; CoeffIndex < 9; ++CoeffIndex)
    {
        SHRadiance[CoeffIndex] = WaveActiveSum(SHRadiance[CoeffIndex]);
    }

    if (LaneIndex == 0)
    {
        // Clamped cosine lobe convolved in SH space (Ramamoorthi 2001):
        // E(N) = sum_lm A_l * L_lm * y_lm(N), with L_lm = int L(s) y_lm(s) ds.
        const float IrradianceBandA[3] =
        {
            PI,               // A_0
            2.0f * PI / 3.0f, // A_1
            PI / 4.0f,        // A_2
        };

        // Band index of coefficient j (0, 0, 0, 1, 1, 1, 2, 2, 2).
        const int BandIndex[9] = {0, 0, 0, 1, 1, 1, 2, 2, 2};

        [unroll]
        for (int CoeffOut = 0; CoeffOut < 9; ++CoeffOut)
        {
            // Uniform sphere sampling has pdf 1/(4*pi), so the projection
            // integral int L(s)*y_j(s) ds = mean(L*y_j) * 4*pi.
            float3 c = SHRadiance[CoeffOut] * (4.0f * PI / float(GParams.TotalSampleCount));

            // Convolve with the clamped cosine lobe in SH space so that the
            // reconstruction sum_j (A_lj * c_j) * y_j(N) yields irradiance E(N).
            float3 IrradianceCoeff = c * IrradianceBandA[BandIndex[CoeffOut]];

            // Slots [34..42] follow the 33 SHE specular coefficients + 1 auto bias.
            GSHESHCoeff[34 + CoeffOut] = float4(IrradianceCoeff, 0.0f);
        }

        GSHESHCoeff[43] = float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}
