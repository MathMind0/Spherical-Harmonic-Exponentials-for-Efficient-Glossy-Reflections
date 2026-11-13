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
#define IBL_MODE_CLASSIC_SH 3
#define IBL_MODE_SPHERICAL_GAUSSIAN 4

#define MATERIAL_MODE_DIFFUSE_AND_SPECULAR 0
#define MATERIAL_MODE_DIFFUSE_ONLY 1
#define MATERIAL_MODE_SPECULAR_ONLY 2

// SG (Spherical Gaussian) IBL: fit the environment radiance as a sum of
// isotropic SGs a_i * exp(lambda_i * (dot(mu_i, v) - 1)). The UI selects the
// lobe count (1..SG_MAX_LOBES) and the fitting algorithm: axes either stay at
// a fixed Fibonacci-uniform sphere distribution, or are free to move.
#define SG_MAX_LOBES 10
#define SG_FIT_FIXED_AXIS 0   // Axes pinned to a uniform Fibonacci sphere set.
#define SG_FIT_FREE_AXIS 1    // Axes refined to the radiance-weighted centroid.

// Classic (standard) SH IBL: the environment radiance is projected onto real
// spherical harmonics up to 9th order (l = 0..9, 100 coefficients). The
// forward shader truncates to the band count selected in the UI (order 2..10).
#define SH_CLASSIC_MAX_ORDER 10
#define SH_CLASSIC_MAX_BANDS (SH_CLASSIC_MAX_ORDER - 1)
#define SH_CLASSIC_COEFF_COUNT (SH_CLASSIC_MAX_ORDER * SH_CLASSIC_MAX_ORDER)

// =============================================================================
// SHE (Spherical Harmonic Exponentials): the P and Q SH orders are now
// INDEPENDENTLY configurable from the UI (P: 1..SHE_MAX_P, Q: 1..SHE_MAX_Q).
//
// The specular coefficient vector has a FIXED layout of SHE_SPEC_COEFFS slots
// so every bake / reduction / solve / render stage can use a compile-time
// constant dimension. The chosen P/Q orders only decide which slots carry
// active features (unused slots are zero and contribute nothing). Layout:
//   [0 .. SHE_P_COEFFS-1]            : P section  = Y_k(R),  k = 0 .. 99
//   [SHE_Q_OFFSET .. SPEC-1]         : Q section  = Y_k(Hr), k = 1 .. 24
//                                      (Q drops the DC term k=0)
// With P order p only k < p*p of the P section is active; with Q order q only
// k < q*q (k>=1) of the Q section is active. Both sections use the shared
// SHBasisEval basis (up to 10th order) and vMF band weights.
// Defaults P=5, Q=3 give 25 + 8 = 33 active coefficients, matching the
// original SH5/SH3 SHE exactly (only the flat index of the Q section changes).
// =============================================================================
#define SHE_MAX_P 10
#define SHE_MAX_Q 5
#define SHE_P_COEFFS (SHE_MAX_P * SHE_MAX_P)            // 100
#define SHE_Q_COEFFS (SHE_MAX_Q * SHE_MAX_Q - 1)        // 24 (k = 1..24)
#define SHE_Q_OFFSET SHE_P_COEFFS                       // 100
#define SHE_SPEC_COEFFS (SHE_P_COEFFS + SHE_Q_COEFFS)   // 124
#define SHE_DIFFUSE_COEFFS 9                            // fixed 3-band SH diffuse

// SHESHCoeff buffer slot layout (float4 each):
//   [0 .. SHE_SPEC_COEFFS-1]                     : specular coefficients
//   [SHE_SPEC_COEFFS]                            : auto bias (log-space energy)
//   [SHE_DIFFUSE_BASE .. +SHE_DIFFUSE_COEFFS-1] : 3-band diffuse irradiance SH
//   [last]                                       : spare (zeroed)
#define SHE_AUTOBIAS_SLOT SHE_SPEC_COEFFS                    // 124
#define SHE_DIFFUSE_BASE (SHE_SPEC_COEFFS + 1)               // 125
#define SHE_COEFF_BUFFER_SIZE (SHE_DIFFUSE_BASE + SHE_DIFFUSE_COEFFS + 1) // 135

// Per-group LUT / PCA stride: specular coefficients + auto bias.
#define SHE_LUT_STRIDE (SHE_SPEC_COEFFS + 1)                 // 125

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
	int bUseRoughnessLUT;            // 1 = render SHE with per-roughness grouped-fit coefficients (LUT).
	int SHECoeffLUTCount;            // Number of roughness groups stored in the LUT buffer.
	float SHECoeffLUTRoughnessStart; // Roughness value represented by LUT group 0.
	float SHECoeffLUTRoughnessEnd;   // Roughness value represented by the last LUT group.
	int bUsePCA;                     // 1 = reconstruct LUT coefficients from PCA basis + weights.
	int SHEPCABasisCount;            // Number of PCA basis vectors actually active (N).
	int ClassicSHOrder;              // Classic SH IBL band count (2..10), l = 0 .. ClassicSHOrder-1.
	int SGLobeCount;                 // SG IBL lobe count (1..SG_MAX_LOBES).
	int SGAxisMode;                  // SG_FIT_FIXED_AXIS or SG_FIT_FREE_AXIS (reflects the fitted data).
	int SHEPOrder;                   // SHE P band count used by the current bake (1..SHE_MAX_P).
	int SHEQOrder;                   // SHE Q band count used by the current bake (1..SHE_MAX_Q).
};

struct SALIGN FSHEBuildConstantData
{
	int32_t ViewCount;
	int32_t NormalCount;
	int32_t ViewCountSqrt;
	int32_t NormalCountSqrt;
	int32_t RoughnessCount;
	int32_t SphericalHarmonicCount; // = SHE_SPEC_COEFFS (fixed 124).
	float RoughnessStart; // Bake roughness range start.
	float RoughnessEnd;   // Bake roughness range end.
	int32_t bRandomRoughness; // 1 = jitter each bake row roughness within its group band.
	int32_t POrder;            // Active P band count (1..SHE_MAX_P).
	int32_t QOrder;            // Active Q band count (1..SHE_MAX_Q).
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
	float4 Coeffs[SHE_COEFF_BUFFER_SIZE];
};

// Classic SH IBL: SH_CLASSIC_COEFF_COUNT RGB radiance projection coefficients
// (projection L_lm = integral L(s) Y_lm(s) ds). Stored one float4 per coefficient.
struct SALIGN FClassicSHCoefficientConstantData
{
	float4 Coeffs[SH_CLASSIC_COEFF_COUNT];
};

#ifdef __cplusplus
#undef SALIGN
#endif
