#pragma once

#ifdef __cplusplus
#include "DirectXMath/DirectXMath.h"
typedef XMFLOAT4X4 float4x4;
typedef XMFLOAT4X3 float4x3;
typedef XMFLOAT2 float2;
typedef XMFLOAT3 float3;
typedef XMFLOAT4 float4;
#endif

#ifdef __cplusplus
#define SALIGN alignas(256)
#else
#define SALIGN
#endif

#define IBL_MODE_SPLIT_SUM_APPROXIMATION 0
#define IBL_MODE_SPHERICAL_HARMONICS_EXPONENTIAL 1
#define IBL_MODE_REFERENCE 2

#define MATERIAL_MODE_DIFFUSE_AND_SPECULAR 0
#define MATERIAL_MODE_DIFFUSE_ONLY 1
#define MATERIAL_MODE_SPECULAR_ONLY 2

struct SALIGN FPerDrawConstantData
{
	float4x4 ObjectToClip;
	float4x3 ObjectToWorld;
	float3 Albedo;
	float Metallic;
	float Roughness;
	float AO;
};

struct SALIGN FPerFrameConstantData
{
	float4 LightPositions[4];
	float4 LightColors[4];
	float4 ViewerPosition;
	float4 ViewDirection; // Unit vector from the focus point toward the camera eye. Used as the constant per-pixel view vector in orthographic mode.
	int MaterialMode;
	int IBLMode;
	uint32_t NumFrames;
	int bOrthographic; // 1 = orthographic (parallel view rays, use ViewDirection), 0 = perspective (rays converge at ViewerPosition).
};

struct SALIGN FSHEBuildConstantData
{
	int32_t ViewCount;
	int32_t NormalCount;
	int32_t ViewCountSqrt;
	int32_t NormalCountSqrt;
	int32_t RoughnessCount;
	int32_t SphericalHarmonicCount;
};

struct SALIGN FSHEReductionConstantData
{
	int32_t ViewCount;
	int32_t ElementCount;
	int32_t GroupCountZ;
	int32_t SphericalHarmonicCount;
};

struct SALIGN FSHEDiffuseParams
{
	uint32_t TotalSampleCount; // e.g. 65536
	uint32_t SamplesPerThread; // 16
	uint32_t ThreadCountX;     // group thread count == wave size (32 or 64)
};

struct SALIGN FSHECoefficientConstantData
{
	float4 Coeffs[44]; // [0..32] SHE specular coefficients, [33] auto bias, [34..42] 3-band SH diffuse irradiance, [43] unused.
};

#ifdef __cplusplus
#undef SALIGN
#endif
