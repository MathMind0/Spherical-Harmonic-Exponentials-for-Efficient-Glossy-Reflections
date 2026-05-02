#include "../CPUAndGPUCommon.h"
#include "Common.hlsli"

#define GRootSignature \
    "RootFlags(0), " \
    "DescriptorTable(CBV(b0), SRV(t0), SRV(t1), UAV(u0), visibility = SHADER_VISIBILITY_ALL), " \

ConstantBuffer<FSHEReductionConstantData> GSHEReductionCB : register(b0);
Texture2D GSHEMatrixA : register(t0);
Texture2D GSHEMatrixb : register(t1);

RWStructuredBuffer<float4> GSHESHCoeff : register(u0);

[RootSignature(GRootSignature)]
[numthreads(1, 1, 1)]
void MainCS()
{
    float3 TargetEnergy = 0.0f;
    float3 ReconstructedEnergy = 0.0f;

    for (uint ElementIndex = 0; ElementIndex < GSHEReductionCB.ElementCount; ++ElementIndex)
    {
        const uint ViewIndex = ElementIndex % GSHEReductionCB.ViewCount;
        const uint Height = ElementIndex / GSHEReductionCB.ViewCount;
        const uint ABaseX = ViewIndex * GSHEReductionCB.SphericalHarmonicCount;

        float3 LogE0 = 0.0f;
        for (uint CoeffIndex = 0; CoeffIndex < GSHEReductionCB.SphericalHarmonicCount; ++CoeffIndex)
        {
            LogE0 += GSHEMatrixA[uint2(ABaseX + CoeffIndex, Height)].r * GSHESHCoeff[CoeffIndex].rgb;
        }

        const float3 TargetLogE0 = GSHEMatrixb[uint2(ViewIndex, Height)].rgb;
        TargetEnergy += exp(TargetLogE0);
        ReconstructedEnergy += exp(LogE0);
    }

    GSHESHCoeff[GSHEReductionCB.SphericalHarmonicCount] = float4(log(max(TargetEnergy, 1e-6f) / max(ReconstructedEnergy, 1e-6f)), 0.0f);
}
