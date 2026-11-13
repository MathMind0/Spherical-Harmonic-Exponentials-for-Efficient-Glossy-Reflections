#include "../CPUAndGPUCommon.h"
#include "Common.hlsli"

// SHE specular coefficients live in a FIXED SHE_SPEC_COEFFS-slot layout
// (P section + Q section). Unused P/Q slots are zero in the A matrix, so the
// normal equations stay (ridge-)positive-definite and the corresponding
// solved coefficients come out exactly 0. This keeps the Cholesky dimension a
// compile-time constant regardless of the UI-selected P/Q orders.

#define GRootSignature \
    "RootFlags(0), " \
    "DescriptorTable(SRV(t0), UAV(u0), visibility = SHADER_VISIBILITY_ALL), " \

Texture2D GSHEMatrixAT : register(t0); // float

RWStructuredBuffer<float4> GSHEBuffer : register(u0);

[RootSignature(GRootSignature)]
[numthreads(1, 1, 1)]
void MainCS()
{
    float L[SHE_SPEC_COEFFS][SHE_SPEC_COEFFS];
    float3 y[SHE_SPEC_COEFFS];
    float3 x[SHE_SPEC_COEFFS];

    // Cholesky decomposition
    for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            float Sum = 0.0f;
            for (int k = 0; k < j; ++k)
                Sum += L[i][k] * L[j][k];

            if (i == j)
            {
                float v = GSHEMatrixAT[uint2(i, i)].r - Sum;
                L[i][j] = sqrt(max(v, 1e-8f));
            }
            else
            {
                L[i][j] = (GSHEMatrixAT[uint2(i, j)].r - Sum) / max(L[j][j], 1e-8f);
            }
        }
    }

    // Forward substitution. ATb columns are stored just past the KxK block,
    // at columns SHE_SPEC_COEFFS .. SHE_SPEC_COEFFS+2 (R, G, B).
    for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
    {
        float3 Sum = 0;
        for (int j = 0; j < i; ++j)
            Sum += L[i][j] * y[j];

        float3 ATb_i = float3(
            GSHEMatrixAT[uint2(i, SHE_SPEC_COEFFS    )].r,
            GSHEMatrixAT[uint2(i, SHE_SPEC_COEFFS + 1)].r,
            GSHEMatrixAT[uint2(i, SHE_SPEC_COEFFS + 2)].r
        );

        y[i] = (ATb_i - Sum) / max(L[i][i], 1e-8f);
    }

    // Backward substitution
    for (int i = SHE_SPEC_COEFFS - 1; i >= 0; --i)
    {
        float3 Sum = 0;
        for (int j = i + 1; j < SHE_SPEC_COEFFS; ++j)
            Sum += L[j][i] * x[j];

        x[i] = (y[i] - Sum) / max(L[i][i], 1e-8f);
        GSHEBuffer[i] = float4(x[i], 0.0f);
    }
}
