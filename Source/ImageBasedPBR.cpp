#include "Library.h"
#include "CPUAndGPUCommon.h"
#include "Shaders/SHBasis.h"
#include "d3dx12.h"
#include "imgui/imgui.h"
#include "EAStdC/EAStdC.h"
#include "EAStdC/EASprintf.h"
#include "EAStdC/EABitTricks.h"
#include "EAStdC/EAString.h"
#include "stb_image.h"
#include "cgltf.h"

#include <math.h>
#include <string.h>

#define MESH_MAX_NUM_SECTIONS 4

// SHE build grid: 8x8 view directions, 8x8 normals, fixed spec-coefficient
// slots (SHE_SPEC_COEFFS), and 64 NormalCount rows per roughness slice.
static const uint32_t GSHEViewCountSqrt = 8;
static const uint32_t GSHENormalCountSqrt = 8;
static const uint32_t GSHEViewCount = GSHEViewCountSqrt * GSHEViewCountSqrt;
static const uint32_t GSHENormalCount = GSHENormalCountSqrt * GSHENormalCountSqrt;
static const uint32_t GSHESphericalHarmonicCount = SHE_SPEC_COEFFS; // fixed 124-slot layout.
static const uint32_t GSHECoeffLUTMaxGroups = 16; // Matches the UI max bake levels.
static const uint32_t GSHECoeffPerGroup = SHE_LUT_STRIDE; // specular + autoBias.
static const uint32_t GSHECoeffLUTMaxBasis = 16;   // Max PCA basis vectors (N).
static const uint32_t GSHECoeffLUTMaxFloat4 = 4096; // LUT capacity: raw G*S or PCA S + N*S + G*N.
static const int GSHEPCADimMax = SHE_LUT_STRIDE * 3; // PCA sample dimension (3 RGB per slot).

// The SHE specular coefficients grouped by SH band (order l). The P section
// covers l = 0..9 (1/3/5/7/9/11/13/15/17/19 coefficients), the Q section
// covers l = 1..4 (3/5/7/9 coefficients, DC dropped). Each band column lists
// the flat coefficient indices that belong to it (P bands use their natural
// k = l*l..l*l+2l, Q bands live at SHE_Q_OFFSET + (k-1)).
static const int GCoeffBandCount = (SHE_MAX_P + SHE_MAX_Q - 1);
static const int GCoeffBandMaxMembers = (2 * SHE_MAX_P - 1); // widest band (2l+1).

static const char *GCoeffBandLabels[GCoeffBandCount] = {
	"P l=0", "P l=1", "P l=2", "P l=3", "P l=4", "P l=5", "P l=6", "P l=7", "P l=8", "P l=9",
	"Q l=1", "Q l=2", "Q l=3", "Q l=4"
};
static const int GCoeffBandCoeffCount[GCoeffBandCount] = {
	1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
	3, 5, 7, 9
};
// BandIsP[b] and BandL[b] describe each band; indices are derived on the fly
// (see SHEBandIndicesFill) so the table works for any P/Q layout.
static const int GCoeffBandIsP[GCoeffBandCount] = {
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	0, 0, 0, 0
};
static const int GCoeffBandL[GCoeffBandCount] = {
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
	1, 2, 3, 4
};

// Fill the flat coefficient indices for band b into Out[0..count-1].
static int SHEBandIndicesFill(int b, int *Out)
{
	const bool bIsP = GCoeffBandIsP[b] != 0;
	const int l = GCoeffBandL[b];
	const int count = 2 * l + 1;
	for (int m = 0; m < count; ++m)
	{
		const int k = l * l + m; // flat SH index for this (l, m)
		if (bIsP)
		{
			Out[m] = k; // P section slot == flat k
		}
		else
		{
			// Q section stores flat k starting from k=1 at SHE_Q_OFFSET.
			Out[m] = SHE_Q_OFFSET + (k - 1);
		}
	}
	return count;
}

// CPU-side copy of the grouped-fit coefficients, used by both the render path
// (uploaded into the LUT buffer) and the ImGui variance table.
struct FSHECoeffTable
{
	int Valid = 0;                      // 1 when the table has been computed at least once.
	int BakeRoughnessCount = 0;         // Number of roughness groups baked when this table was built.
	int POrder = SHE_MAX_P;             // P band count used for this fit.
	int QOrder = SHE_MAX_Q;             // Q band count used for this fit.
	float RoughnessStart = 0.0f;        // Roughness of group 0.
	float RoughnessEnd = 1.0f;          // Roughness of the last group.
	// Index 0..N-1: per-roughness grouped fit. Index N: ungrouped (global) fit.
	// Each Coeff[group] holds SHE_LUT_STRIDE float4 (specular + autoBias), only .xyz used.
	float Coeff[GSHECoeffLUTMaxGroups + 1][GSHECoeffPerGroup][4];
	// Squared deviation of each grouped fit's channel value from the ungrouped
	// baseline, accumulated per band and in total. Computed for the UI table.
	float BandVariance[GSHECoeffLUTMaxGroups][GCoeffBandCount]; // [group][band]; variance vs ungrouped baseline.
	float TotalVariance[GSHECoeffLUTMaxGroups];    // [group]; variance over all specular coefficients.
	// PCA data (filled by UploadCoeffLUT when PCA is active), for the ImGui table:
	int PCABasisCount = 0;                          // number of active basis vectors (N).
	float PCABasisCoeff[GSHECoeffLUTMaxBasis][GSHECoeffPerGroup][4]; // [k][i][c] basis k value of parameter i.
	float PCABasisExplained[GSHECoeffLUTMaxBasis];  // [k] fraction of total variance captured by basis k.
	float PCABandVariance[GSHECoeffLUTMaxGroups][GCoeffBandCount]; // [g][b] var of original vs PCA reconstruction.
	float PCATotalVariance[GSHECoeffLUTMaxGroups];  // [g] total var of original vs PCA reconstruction.
};

static float HalfToFloat(uint16_t H)
{
	const uint32_t Sign = (uint32_t)(H & 0x8000u) << 16;
	uint32_t Exp = (uint32_t)((H >> 10) & 0x1Fu);
	uint32_t Mant = (uint32_t)(H & 0x03FFu);
	uint32_t F;
	if (Exp == 0)
	{
		if (Mant == 0)
		{
			F = Sign;
		}
		else
		{
			// Subnormal: normalize.
			Exp = 113;
			while ((Mant & 0x0400u) == 0)
			{
				Mant <<= 1;
				--Exp;
			}
			Mant &= 0x03FFu;
			F = Sign | (Exp << 23) | (Mant << 13);
		}
	}
	else if (Exp == 31)
	{
		F = Sign | 0x7F800000u | (Mant << 13); // Inf / NaN.
	}
	else
	{
		F = Sign | ((Exp + 112) << 23) | (Mant << 13);
	}
	float Result;
	memcpy(&Result, &F, sizeof(Result));
	return Result;
}

// Solve (A^T A + ridge*I) x = A^T b for a KxK symmetric positive system,
// for three right-hand sides (RGB). AtA is shared across channels; Atb has 3.
// K = SHE_SPEC_COEFFS is the fixed slot count; only the first N slots are
// active for the current P/Q orders (unused slots solve to exactly 0).
static void SolveGroupedFit(const float AtA[SHE_SPEC_COEFFS][SHE_SPEC_COEFFS], const float Atb[SHE_SPEC_COEFFS][3], float OutX[SHE_SPEC_COEFFS][3], int N)
{
	float L[SHE_SPEC_COEFFS][SHE_SPEC_COEFFS] = {};
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			float Sum = AtA[i][j];
			for (int k = 0; k < j; ++k)
			{
				Sum -= L[i][k] * L[j][k];
			}
			if (i == j)
			{
				// Small ridge keeps the Cholesky factorization stable even
				// for a single-roughness group (rank-deficient rows).
				L[i][j] = sqrtf(fmaxf(Sum, 1e-4f));
			}
			else
			{
				L[i][j] = Sum / fmaxf(L[j][j], 1e-4f);
			}
		}
	}
	for (int Ch = 0; Ch < 3; ++Ch)
	{
		float Y[SHE_SPEC_COEFFS];
		for (int i = 0; i < N; ++i)
		{
			float Sum = Atb[i][Ch];
			for (int j = 0; j < i; ++j)
			{
				Sum -= L[i][j] * Y[j];
			}
			Y[i] = Sum / fmaxf(L[i][i], 1e-4f);
		}
		for (int i = N - 1; i >= 0; --i)
		{
			float Sum = Y[i];
			for (int j = i + 1; j < N; ++j)
			{
				Sum -= L[j][i] * OutX[j][Ch];
			}
			OutX[i][Ch] = Sum / fmaxf(L[i][i], 1e-4f);
		}
	}
}


static const char *GEnvironmentMapNames[] =
{
	"Qwantani Sunset Puresky",
	"Newport Loft",
};

static const char *GEnvironmentMapPaths[] =
{
	"Data/Textures/qwantani_sunset_puresky_4k.hdr",
	"Data/Textures/Newport_Loft.hdr",
};

enum
{
	MESH_Cube,
	MESH_Sphere,
};

enum
{
	PSO_Test,
	PSO_SimpleForward,
	PSO_SampleEnvMap,
	PSO_EquirectangularToCube,
	PSO_GenerateIrradianceMap,
	PSO_PrefilterEnvMap,
	PSO_GenerateBRDFIntegrationMap,
	PSO_SHE_Build,
	PSO_SHE_Reduction,
	PSO_SHE_Reduction_Merge,
	PSO_SHE_Solve,
	PSO_SHE_Calibrate,
	PSO_SHE_Diffuse,                 // Placeholder index, unused directly.
	PSO_SHE_Diffuse_Wave32,          // Stage 1 partial projection, wave32 groups.
	PSO_SHE_Diffuse_Wave64,          // Stage 1 partial projection, wave64 groups.
	PSO_SHE_Diffuse_Finalize_Wave32, // Stage 2 partial sum + convolution writeback, wave32.
	PSO_SHE_Diffuse_Finalize_Wave64, // Stage 2 partial sum + convolution writeback, wave64.
};

struct FVertex
{
	XMFLOAT3 Position;
	XMFLOAT3 Normal;
};

struct FMeshSection
{
	uint32_t IndexCount;
	uint32_t StartIndexLocation;
	uint32_t BaseVertexLocation;
	uint32_t MaterialIndex;
};

struct FMesh
{
	FMeshSection Sections[MESH_MAX_NUM_SECTIONS];
	uint32_t NumSections;
};

struct FStaticMesh
{
	uint32_t IndexCount;
	uint32_t StartIndexLocation;
	uint32_t BaseVertexLocation;
};

// Vertical order of the IBL groups in the sphere grid. Row 0 is drawn at the
// BOTTOM of the screen (local +Y points up), so GIBLModeGroupOrder[0] is the
// bottom group and the last entry is the top group. 5 groups x 2 metallic
// rows = 10 rows.
// Screen top-to-bottom: SG / Split-Sum / Reference / SHE / Classic SH.
// Classic SH stays in the bottom two rows, directly below the SHE rows.
static const int GIBLModeGroupOrder[5] =
{
	IBL_MODE_CLASSIC_SH,
	IBL_MODE_SPHERICAL_HARMONICS_EXPONENTIAL,
	IBL_MODE_REFERENCE,
	IBL_MODE_SPLIT_SUM_APPROXIMATION,
	IBL_MODE_SPHERICAL_GAUSSIAN,
};
static const int GIBLModeGroupCount = 5;

// Mouse orbit camera control parameters.
static const float GCameraRotateSpeed = 0.005f;	   // Radians per pixel of mouse drag.
static const float GCameraMaxPitch = XM_PIDIV2 * 0.9f; // ~81 degrees, keeps the billboard basis non-degenerate.
static const float GCameraZoomSpeed = 0.1f;			   // Exponential zoom rate per wheel notch.
static const float GCameraMinViewScale = 0.2f;		   // Zoom-in limit (smaller = larger spheres on screen).
static const float GCameraMaxViewScale = 3.0f;		   // Zoom-out limit.

struct FStaticMeshInstance
{
	XMFLOAT3 Position;
	XMFLOAT3 Rotation;
	uint32_t MeshIndex;
	float Roughness;
	float RoughnessT;
	float Metallic;
	int IBLMode;
};

struct FDemoRoot
{
	FGraphicsContext Gfx;
	FUIContext UI;
	eastl::vector<FStaticMesh> StaticMeshes;
	eastl::vector<FStaticMeshInstance> StaticMeshInstances;
	eastl::vector<ID3D12PipelineState *> Pipelines;
	eastl::vector<ID3D12RootSignature *> RootSignatures;
	ID3D12Resource *StaticVB;
	ID3D12Resource *StaticIB;
	D3D12_VERTEX_BUFFER_VIEW StaticVBView;
	D3D12_INDEX_BUFFER_VIEW StaticIBView;
	XMFLOAT3 CameraPosition;
	XMFLOAT3 CameraFocusPosition;
	float CameraYaw = 0.0f;		 // Orbit angle around the Y axis, radians.
	float CameraPitch = 0.0f;	 // Orbit angle around the X axis, radians, clamped to +/-GCameraMaxPitch.
	float CameraDistance = 12.0f; // Distance from focus to camera eye.
	float CameraViewScale = 1.0f; // Zoom factor applied to the orthographic view size.
	bool bOrthographicProjection = true; // Use orthographic projection (parallel view rays). Default on.
	XMFLOAT3 AlbedoColor = XMFLOAT3(0.5f, 0.5f, 0.5f); // Customizable albedo color for spheres.
	bool bMouseDragging = false; // Left button held outside of ImGui UI.
	float LastMouseX = 0.0f;
	float LastMouseY = 0.0f;
	float PendingMouseWheel = 0.0f; // Wheel delta captured before NewFrame resets it.
	ID3D12Resource *EnvMap;
	ID3D12Resource *IrradianceMap;
	ID3D12Resource *PrefilteredEnvMap;
	ID3D12Resource *BRDFIntegrationMap;
	ID3D12Resource *SHEMatrixA;
	ID3D12Resource *SHEMatrixb;
	ID3D12Resource *SHEMatrixAT;
	ID3D12Resource *SHESHCoeff;
	ID3D12Resource *SHECoeffLUT;               // StructuredBuffer<float4>, flat [group*SHE_LUT_STRIDE + i].
	D3D12_CPU_DESCRIPTOR_HANDLE SHECoeffLUTSRV; // SRV bound at t5 in the SimpleForward descriptor table.
	void *SHECoeffLUTMapped = nullptr;          // Persistently mapped CPU write pointer (upload heap).
	ID3D12Resource *ClassicSHCoeff = nullptr;       // Upload buffer: StructuredBuffer<float4> of SH projection coefficients.
	D3D12_CPU_DESCRIPTOR_HANDLE ClassicSHCoeffSRV;  // SRV bound at t6 in the SimpleForward descriptor table.
	void *ClassicSHCoeffMapped = nullptr;           // Persistently mapped CPU write pointer.
	int ClassicSHOrder = 3;                         // Classic SH band count (order 2..10), default 3-band.
	ID3D12Resource *SGLobes = nullptr;              // Upload buffer: StructuredBuffer<float4>, 2 float4 per lobe (axis+lambda, amplitude).
	D3D12_CPU_DESCRIPTOR_HANDLE SGLobesSRV;         // SRV bound at t7 in the SimpleForward descriptor table.
	void *SGLobesMapped = nullptr;                  // Persistently mapped CPU write pointer.
	int SGLobeCount = 6;                            // SG IBL lobe count (1..SG_MAX_LOBES).
	int SGAxisMode = SG_FIT_FIXED_AXIS;             // SG_FIT_FIXED_AXIS / SG_FIT_FREE_AXIS.
	int SGDirty = 1;                                // 1 = re-run the CPU SG fit with the current settings.
	D3D12_CPU_DESCRIPTOR_HANDLE EnvMapSRV;
	D3D12_CPU_DESCRIPTOR_HANDLE IrradianceMapSRV;
	D3D12_CPU_DESCRIPTOR_HANDLE PrefilteredEnvMapSRV;
	D3D12_CPU_DESCRIPTOR_HANDLE BRDFIntegrationMapSRV;
	D3D12_CPU_DESCRIPTOR_HANDLE AccumulationBufferSRV;
	D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixASRV;
	D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixbSRV;
	D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixATSRV;
	D3D12_CPU_DESCRIPTOR_HANDLE SHESHCoeffCBV;
	ID3D12Resource *MSColorBuffer;
	ID3D12Resource *MSDepthBuffer;
	ID3D12Resource *AccumulationBuffer;
	D3D12_CPU_DESCRIPTOR_HANDLE MSColorBufferRTV;
	D3D12_CPU_DESCRIPTOR_HANDLE MSDepthBufferDSV;
	int LastMaterialMode;
	int LastIBLMode;
	int MaterialMode;
	int IBLMode;
	int EnvironmentMapIndex;
	int SelectedEnvironmentMapIndex;
	int PendingEnvironmentMapIndex;
	float RoughnessStart = 0.0f; // Display grid roughness sweep start (end fixed at 1.0).
	float BakeRoughnessStart = 0.0f; // SHE bake roughness range start.
	float BakeRoughnessEnd = 1.0f; // SHE bake roughness range end.
	int BakeRoughnessCount = 4; // SHE bake roughness levels.
	int SHEPOrder = 5;          // SHE P (R-direction) SH band count, 1..SHE_MAX_P.
	int SHEQOrder = 3;          // SHE Q (Hr-direction) SH band count, 1..SHE_MAX_Q.
	int PendingSHEBake = 0; // 1 = re-run the SHE precomputation with the current bake parameters.
	bool bUseRoughnessLUT = false;  // ImGui toggle: render SHE with per-roughness grouped-fit coefficients.
	bool bUsePCA = false;           // ImGui toggle: PCA-compress the Roughness LUT and reconstruct at runtime.
	int PCABasisCount = 4;          // Requested number of PCA basis vectors (N).
	int ActivePCABasisCount = 0;    // N actually used by the current LUT layout (clamped).
	int PCALUTDirty = 0;            // 1 = rebuild the LUT buffer from CoeffTable (no rebake needed).
	float PCAExplainedVariance = 0.0f; // Fraction of total variance kept by the top-N basis vectors.
	bool bRandomRoughness = false;  // ImGui toggle: jitter each bake row roughness within its group band.
	bool bShowCoeffTable = true;    // ImGui toggle: show the grouped-fit coefficient / variance table.
	FSHECoeffTable CoeffTable;      // CPU copy of the grouped-fit coefficients and their variance.
	uint32_t NumSamples;
	uint32_t NumFrames;
	uint32_t WaveSize = 32; // GPU wave size in lanes (32 NVIDIA/Intel, 64 AMD), used to select SHE diffuse PSO variants.
};

static void RebuildEnvironment(FDemoRoot &Root);
static void RebuildSHEBake(FDemoRoot &Root);
static void ComputeGroupedSHEFit(FDemoRoot &Root);
static void UploadCoeffLUT(FDemoRoot &Root);
static void DrawSHECoeffTable(FDemoRoot &Root);
static void CreateSGLobeResources(FDemoRoot &Root);
static void FitEnvMapToSG(FDemoRoot &Root);

static void UpdateUI(FDemoRoot &Root, float DeltaTime)
{
	ImGuiIO &IO = ImGui::GetIO();
	IO.KeyCtrl = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
	IO.KeyShift = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
	IO.KeyAlt = (GetKeyState(VK_MENU) & 0x8000) != 0;
	IO.DeltaTime = DeltaTime;

	// Capture the wheel delta before ImGui::NewFrame() resets MouseWheel each frame.
	Root.PendingMouseWheel = IO.MouseWheel;

	ImGui::NewFrame();

	// Our Custom UI Window
	ImGui::Begin("Image Based Lighting");

	{
		Root.LastMaterialMode = Root.MaterialMode;
		ImGui::Text("Material Mode");
		const char *items[] = {"Diffuse + Specular", "Diffuse Only", "Specular Only"};
		ImGui::Combo("##MaterialMode", &(Root.MaterialMode), items, IM_ARRAYSIZE(items));
	}

	{
		ImGui::Text("IBL Mode");
		ImGui::TextDisabled("Top: SG / Split-Sum / Ref / SHE / Classic SH :Bottom");
	}

	{
		ImGui::Text("SG Lobe Count");
		ImGui::TextDisabled("Spherical Gaussians used to fit the environment");
		if (ImGui::SliderInt("##SGLobeCount", &Root.SGLobeCount, 1, SG_MAX_LOBES, "Lobes %d"))
		{
			Root.SGDirty = 1;
			Root.NumFrames = 0;
		}
		const char *FitModes[] = {"Axis Fixed (uniform)", "Axis Free (fitted)"};
		ImGui::Text("SG Fitting");
		if (ImGui::Combo("##SGAxisMode", &Root.SGAxisMode, FitModes, IM_ARRAYSIZE(FitModes)))
		{
			Root.SGAxisMode = XMMin(XMMax(Root.SGAxisMode, SG_FIT_FIXED_AXIS), SG_FIT_FREE_AXIS);
			Root.SGDirty = 1;
			Root.NumFrames = 0;
		}
		ImGui::TextDisabled("Fixed: axes on a uniform Fibonacci sphere; Free: axes move to fit");
	}

	{
		ImGui::Text("Classic SH Order");
		ImGui::TextDisabled("Band count for the classic SH IBL rows (l = 0 .. order-1)");
		if (ImGui::SliderInt("##ClassicSHOrder", &Root.ClassicSHOrder, 2, SH_CLASSIC_MAX_ORDER, "Order %d"))
		{
			Root.NumFrames = 0;
		}
	}

	{
		ImGui::Text("Environment");
		if (ImGui::Combo("##Environment", &Root.SelectedEnvironmentMapIndex, GEnvironmentMapNames, IM_ARRAYSIZE(GEnvironmentMapNames)))
		{
			if (Root.SelectedEnvironmentMapIndex != Root.EnvironmentMapIndex)
			{
				Root.PendingEnvironmentMapIndex = Root.SelectedEnvironmentMapIndex;
			}
		}
	}

	{
		ImGui::Text("Display Roughness Start");
		ImGui::TextDisabled("Sphere grid sweep, up to 1.0");
		if (ImGui::SliderFloat("##RoughnessStart", &Root.RoughnessStart, 0.0f, 0.95f))
		{
			for (auto &Inst : Root.StaticMeshInstances)
			{
				Inst.Roughness = (1.0f - Inst.RoughnessT) * Root.RoughnessStart + Inst.RoughnessT * 1.0f;
			}
			Root.NumFrames = 0;
		}
	}

	{
		// SHE P/Q SH orders. P is the R-direction basis order (1..10), Q the
		// Hr-direction order (1..5). Changing either changes the baked feature
		// layout, so it must trigger a full SHE rebake (not just a LUT repack).
		ImGui::Text("SHE P / Q Orders");
		ImGui::TextDisabled("P: R-direction SH order  Q: Hr-direction SH order");
		if (ImGui::SliderInt("##SHEPOrder", &Root.SHEPOrder, 1, SHE_MAX_P, "P = %d"))
		{
			Root.PendingSHEBake = 1;
		}
		if (ImGui::SliderInt("##SHEQOrder", &Root.SHEQOrder, 1, SHE_MAX_Q, "Q = %d"))
		{
			Root.PendingSHEBake = 1;
		}
		ImGui::TextDisabled("Default P=5 Q=3 reproduces the original SH5 / SH3 SHE");
	}

	{
		// SHE bake roughness range and level count. The SHE precomputation is
		// re-run once the user releases a control (end of edit).
		ImGui::Text("SHE Bake Roughness Range");
		ImGui::TextDisabled("Range and levels baked into the SHE matrices");
		bool bBakeDirty = false;
		const float MinRange = 0.01f;

		ImGui::SliderFloat("##BakeRoughnessStart", &Root.BakeRoughnessStart, 0.0f, 1.0f - MinRange, "Start %.2f");
		Root.BakeRoughnessStart = XMMin(Root.BakeRoughnessStart, Root.BakeRoughnessEnd - MinRange);
		if (ImGui::IsItemDeactivatedAfterEdit())
		{
			bBakeDirty = true;
		}

		ImGui::SliderFloat("##BakeRoughnessEnd", &Root.BakeRoughnessEnd, MinRange, 1.0f, "End %.2f");
		Root.BakeRoughnessEnd = XMMax(Root.BakeRoughnessEnd, Root.BakeRoughnessStart + MinRange);
		if (ImGui::IsItemDeactivatedAfterEdit())
		{
			bBakeDirty = true;
		}

		ImGui::SliderInt("##BakeRoughnessCount", &Root.BakeRoughnessCount, 1, 16, "Levels %d");
		if (ImGui::IsItemDeactivatedAfterEdit())
		{
			bBakeDirty = true;
		}

		if (ImGui::Checkbox("Random Roughness", &Root.bRandomRoughness))
		{
			bBakeDirty = true;
		}
		ImGui::TextDisabled("Jitter each bake row roughness inside its group band");

		ImGui::TextDisabled("SHE rebake runs on slider release");

		if (bBakeDirty)
		{
			Root.PendingSHEBake = 1;
		}
	}

	{
		ImGui::Text("Roughness LUT");
		ImGui::Checkbox("Use per-group SHE fit (LUT)", &Root.bUseRoughnessLUT);
		ImGui::TextDisabled("Grouped-fit coefficients drive the SHE render");
		if (ImGui::Checkbox("PCA Compress LUT", &Root.bUsePCA))
		{
			Root.PCALUTDirty = 1;
		}
		ImGui::TextDisabled("Reconstruct coefficients from top-N PCA basis vectors");
		if (ImGui::SliderInt("##PCABasisCount", &Root.PCABasisCount, 1, 16, "PCA Basis %d"))
		{
			Root.PCALUTDirty = 1;
		}
		ImGui::TextDisabled("Basis vectors kept (ranked by variance / importance)");
		ImGui::Checkbox("Show coefficient table", &Root.bShowCoeffTable);
		ImGui::TextDisabled("Coefficients by band + variance vs ungrouped");
	}

	{
		ImGui::Text("Projection");
		if (ImGui::Checkbox("Orthographic", &Root.bOrthographicProjection))
		{
			Root.NumFrames = 0;
		}
	}

	{
		ImGui::Text("Albedo");
		if (ImGui::ColorEdit3("##Albedo", &Root.AlbedoColor.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel))
		{
			// Reset progressive accumulation so the Reference mode picks up the new albedo immediately.
			Root.NumFrames = 0;
		}
	}

	ImGui::End();

	if (Root.bShowCoeffTable)
	{
		DrawSHECoeffTable(Root);
	}
}

static void Update(FDemoRoot &Root)
{
	double Time;
	float DeltaTime;
	UpdateFrameStats(Root.Gfx.Window, "ImageBasedPBR", Time, DeltaTime, Root.NumFrames);
	UpdateUI(Root, DeltaTime);

	// Re-pack the LUT buffer when the PCA toggle / basis count changed. This
	// only rewrites the persistently-mapped buffer from the CPU coefficient
	// table; no rebake is needed. Rebuild paths already re-upload via
	// ComputeGroupedSHEFit, so clear the flag afterwards either way.
	if (Root.PCALUTDirty && Root.CoeffTable.Valid)
	{
		UploadCoeffLUT(Root);
	}
	Root.PCALUTDirty = 0;

	// Re-fit the SG representation when the lobe count or axis mode changed.
	// This is a pure CPU-side fit of the current HDR; no GPU rebake needed.
	if (Root.SGDirty)
	{
		Root.SGDirty = 0;
		FitEnvMapToSG(Root);
	}

	if (Root.PendingEnvironmentMapIndex >= 0)
	{
		// The environment rebuild already re-runs the SHE bake with the
		// current parameters, so drop any pending bake-only rebuild.
		Root.PendingSHEBake = 0;
		RebuildEnvironment(Root);
	}
	else if (Root.PendingSHEBake)
	{
		// Re-run the SHE precomputation when the bake roughness range or
		// level count changed in the UI.
		Root.PendingSHEBake = 0;
		RebuildSHEBake(Root);
	}

	// Mouse orbit camera control: left-drag to rotate, wheel to zoom.
	// MouseWheel is reset by ImGui::NewFrame() (called in UpdateUI above), so it was
	// captured into Root.PendingMouseWheel inside UpdateUI before NewFrame runs.
	if (Root.PendingMouseWheel != 0.0f && !ImGui::GetIO().WantCaptureMouse)
	{
		Root.CameraViewScale *= powf(GCameraMaxViewScale / GCameraMinViewScale, -Root.PendingMouseWheel * GCameraZoomSpeed);
		Root.CameraViewScale = XMMin(GCameraMaxViewScale, XMMax(GCameraMinViewScale, Root.CameraViewScale));
		Root.NumFrames = 0;
	}

	ImGuiIO &MouseIO = ImGui::GetIO();
	if (MouseIO.MouseDown[0] && !MouseIO.WantCaptureMouse)
	{
		if (Root.bMouseDragging)
		{
			Root.CameraYaw += (MouseIO.MousePos.x - Root.LastMouseX) * GCameraRotateSpeed;
			Root.CameraPitch += (MouseIO.MousePos.y - Root.LastMouseY) * GCameraRotateSpeed;
			Root.CameraPitch = XMMin(GCameraMaxPitch, XMMax(-GCameraMaxPitch, Root.CameraPitch));
			Root.NumFrames = 0;
		}
		Root.bMouseDragging = true;
		Root.LastMouseX = MouseIO.MousePos.x;
		Root.LastMouseY = MouseIO.MousePos.y;
	}
	else
	{
		Root.bMouseDragging = false;
	}

	// Update camera position from orbit angles (focus stays at CameraFocusPosition).
	{
		const float Yaw = Root.CameraYaw;
		const float Pitch = Root.CameraPitch;
		const XMVECTOR Direction = XMVectorSet(
			sinf(Yaw) * cosf(Pitch),
			sinf(Pitch),
			cosf(Yaw) * cosf(Pitch),
			0.0f);
		XMStoreFloat3(&Root.CameraPosition,
			XMLoadFloat3(&Root.CameraFocusPosition) + Direction * Root.CameraDistance);
	}
}

static void Draw(FDemoRoot &Root)
{
	FGraphicsContext &Gfx = Root.Gfx;
	ID3D12GraphicsCommandList2 *CmdList = GetAndInitCommandList(Gfx);

	CmdList->RSSetViewports(1, get_rvalue_ptr(CD3DX12_VIEWPORT(0.0f, 0.0f, (float)Gfx.Resolution[0], (float)Gfx.Resolution[1])));
	CmdList->RSSetScissorRects(1, get_rvalue_ptr(CD3DX12_RECT(0, 0, (LONG)Gfx.Resolution[0], (LONG)Gfx.Resolution[1])));

	CmdList->OMSetRenderTargets(1, &Root.MSColorBufferRTV, TRUE, &Root.MSDepthBufferDSV);
	CmdList->ClearRenderTargetView(Root.MSColorBufferRTV, XMVECTORF32{0.0f}, 0, nullptr);
	CmdList->ClearDepthStencilView(Root.MSDepthBufferDSV, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

	CmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	CmdList->IASetVertexBuffers(0, 1, &Root.StaticVBView);
	CmdList->IASetIndexBuffer(&Root.StaticIBView);

	const XMMATRIX ViewTransform = XMMatrixLookAtLH(XMLoadFloat3(&Root.CameraPosition), XMLoadFloat3(&Root.CameraFocusPosition), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));

	// Projection: orthographic fits the 11x10 sphere grid with margin (grid spans
	// 24.2 x 22 units). Perspective uses the orbit distance to frame the grid.
	XMMATRIX ProjectionTransform;
	if (Root.bOrthographicProjection)
	{
		const float AspectRatio = (float)Gfx.Resolution[0] / (float)Gfx.Resolution[1];
		const float GridWidth = 11 * 2.2f;
		const float GridHeight = 10 * 2.2f;
		float ViewWidth = GridWidth + 4.0f;
		float ViewHeight = ViewWidth / AspectRatio;
		if (ViewHeight < GridHeight + 2.0f)
		{
			ViewHeight = GridHeight + 2.0f;
			ViewWidth = ViewHeight * AspectRatio;
		}
		ViewWidth *= Root.CameraViewScale;
		ViewHeight *= Root.CameraViewScale;
		ProjectionTransform = XMMatrixOrthographicLH(ViewWidth, ViewHeight, 0.1f, 100.0f);
	}
	else
	{
		// Perspective FOV derived so that at the current orbit distance the grid
		// (with the same margin) fills the viewport similarly to the ortho view.
		const float AspectRatio = (float)Gfx.Resolution[0] / (float)Gfx.Resolution[1];
		const float GridWidth = 11 * 2.2f;
		const float GridHeight = 10 * 2.2f;
		float ViewWidth = GridWidth + 4.0f;
		float ViewHeight = ViewWidth / AspectRatio;
		if (ViewHeight < GridHeight + 2.0f)
		{
			ViewHeight = GridHeight + 2.0f;
			ViewWidth = ViewHeight * AspectRatio;
		}
		ViewWidth *= Root.CameraViewScale;
		ViewHeight *= Root.CameraViewScale;
		const float Distance = XMMax(Root.CameraDistance, 0.1f);
		// Vertical FOV half-angle that covers ViewHeight at Distance; use the
		// larger of the two half-extents mapped through the aspect ratio so the
		// whole grid stays visible.
		const float HalfHeight = ViewHeight * 0.5f;
		const float HalfWidth = ViewWidth * 0.5f;
		const float HalfAngleV = atanf(HalfHeight / Distance);
		const float HalfAngleH = atanf(HalfWidth / Distance);
		const float FovV = XMMin(HalfAngleV, HalfAngleH < 0.0001f ? HalfAngleV : atanf(tanf(HalfAngleH) / AspectRatio));
		ProjectionTransform = XMMatrixPerspectiveFovLH(FovV * 2.0f, AspectRatio, 0.1f, 100.0f);
	}

	// Clear accumulation buffer if needed.
	if (Root.LastMaterialMode != Root.MaterialMode || Root.LastIBLMode != Root.IBLMode)
	{
		Root.NumFrames = 0;

		D3D12_CPU_DESCRIPTOR_HANDLE AccumulationBufferUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);

		D3D12_UNORDERED_ACCESS_VIEW_DESC AccumulationBufferUAVDesc = {};
		AccumulationBufferUAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
		AccumulationBufferUAVDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		AccumulationBufferUAVDesc.Texture2D.MipSlice = 0;
		AccumulationBufferUAVDesc.Texture2D.PlaneSlice = 0;

		Gfx.Device->CreateUnorderedAccessView(Root.AccumulationBuffer, nullptr, &AccumulationBufferUAVDesc, AccumulationBufferUAV);

		CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.AccumulationBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)));

		float clearColor[4] = {0, 0, 0, 0};
		CmdList->ClearUnorderedAccessViewFloat(CopyDescriptorsToGPUHeap(Gfx, 1, AccumulationBufferUAV), AccumulationBufferUAV, Root.AccumulationBuffer, clearColor, 0, nullptr);

		CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.AccumulationBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
	}

	// Draw all static mesh instances.
	{
		CmdList->SetPipelineState(Root.Pipelines[PSO_SimpleForward]);
		CmdList->SetGraphicsRootSignature(Root.RootSignatures[PSO_SimpleForward]);

		const XMMATRIX WorldToClip = ViewTransform * ProjectionTransform;

		// Billboard basis: rotate the whole sphere grid (laid out on the local XY plane,
		// normal +Z) so it always faces the camera. Built from the view direction.
		const XMVECTOR Forward = XMVector3Normalize(
			XMLoadFloat3(&Root.CameraFocusPosition) - XMLoadFloat3(&Root.CameraPosition));
		const XMVECTOR UpRef = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
		const XMVECTOR Right = XMVector3Normalize(XMVector3Cross(UpRef, Forward));
		const XMVECTOR Up = XMVector3Cross(Forward, Right);
		XMFLOAT3 RightF, UpF, ForwardF;
		XMStoreFloat3(&RightF, Right);
		XMStoreFloat3(&UpF, Up);
		XMStoreFloat3(&ForwardF, Forward);
		const XMMATRIX Billboard = XMMatrixSet(
			RightF.x, RightF.y, RightF.z, 0.0f,
			UpF.x, UpF.y, UpF.z, 0.0f,
			ForwardF.x, ForwardF.y, ForwardF.z, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

		// Per-column roughness labels: project each column's top anchor (grid
		// local space, billboarded to face the camera) to screen space and draw
		// the value with ImGui's background draw list (rendered after the 3D
		// scene, but behind the ImGui control window).
		{
			ImDrawList *LabelDrawList = ImGui::GetBackgroundDrawList();
			const ImGuiIO &LabelIO = ImGui::GetIO();

			const int32_t NumRows = 10;
			const int32_t NumColumns = 11;
			const float CellSize = 2.2f;
			// Anchor slightly above the top row of spheres.
			const float LabelY = CellSize * (-NumRows * 0.5f + (NumRows - 1) + 0.5f) + CellSize * 0.65f;

			for (int32_t ColumnIdx = 0; ColumnIdx < NumColumns; ++ColumnIdx)
			{
				// Row 0 of each column (instances are laid out row-major).
				const FStaticMeshInstance &Inst = Root.StaticMeshInstances[ColumnIdx];

				char Label[16];
				snprintf(Label, sizeof(Label), "%.2f", Inst.Roughness);

				const XMVECTOR AnchorLocal = XMVectorSet(
					CellSize * (-NumColumns * 0.5f + ColumnIdx + 0.5f), LabelY, 0.0f, 1.0f);
				const XMVECTOR AnchorWorld = XMVector3Transform(AnchorLocal, Billboard);
				const XMVECTOR AnchorClip = XMVector3Transform(AnchorWorld, WorldToClip);

				const float NdcX = XMVectorGetX(AnchorClip) / XMVectorGetW(AnchorClip);
				const float NdcY = XMVectorGetY(AnchorClip) / XMVectorGetW(AnchorClip);

				// Skip labels that fall outside the viewport (e.g. when zoomed in).
				if (XMVectorGetW(AnchorClip) <= 0.0f || NdcX < -1.1f || NdcX > 1.1f || NdcY < -1.1f || NdcY > 1.1f)
				{
					continue;
				}

				const ImVec2 ScreenPos(
					(NdcX * 0.5f + 0.5f) * LabelIO.DisplaySize.x,
					(0.5f - NdcY * 0.5f) * LabelIO.DisplaySize.y);

				const ImVec2 TextSize = ImGui::CalcTextSize(Label);
				const ImVec2 TextPos(ScreenPos.x - TextSize.x * 0.5f, ScreenPos.y - TextSize.y * 0.5f);

				// Soft black shadow for readability on bright environments.
				LabelDrawList->AddText(ImVec2(TextPos.x + 1.0f, TextPos.y + 1.0f), IM_COL32(0, 0, 0, 180), Label);
				LabelDrawList->AddText(TextPos, IM_COL32(255, 255, 255, 255), Label);
			}
		}

		for (int32_t GroupIdx = 0; GroupIdx < GIBLModeGroupCount; ++GroupIdx)
		{
		const int GroupIBLMode = GIBLModeGroupOrder[GroupIdx];

		// Per-frame constant data (one per IBL mode group, only IBLMode differs).
		{
			D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
			auto *CPUAddress = (FPerFrameConstantData *)AllocateGPUMemory(Gfx, sizeof(FPerFrameConstantData), GPUAddress);

			CPUAddress->LightPositions[0] = XMFLOAT4(-10.0f, 10.0f, -10.0f, 1.0f);
			CPUAddress->LightPositions[1] = XMFLOAT4(10.0f, 10.0f, -10.0f, 1.0f);
			CPUAddress->LightPositions[2] = XMFLOAT4(-10.0f, -10.0f, -10.0f, 1.0f);
			CPUAddress->LightPositions[3] = XMFLOAT4(10.0f, -10.0f, -10.0f, 1.0f);

			CPUAddress->LightColors[0] = XMFLOAT4(300.0f, 300.0f, 300.0f, 1.0f);
			CPUAddress->LightColors[1] = XMFLOAT4(300.0f, 300.0f, 300.0f, 1.0f);
			CPUAddress->LightColors[2] = XMFLOAT4(300.0f, 300.0f, 300.0f, 1.0f);
			CPUAddress->LightColors[3] = XMFLOAT4(300.0f, 300.0f, 300.0f, 1.0f);

			const XMFLOAT3 P = Root.CameraPosition;
			CPUAddress->ViewerPosition = XMFLOAT4(P.x, P.y, P.z, 1.0f);

			// Unit vector from the focus point toward the camera eye. In
			// orthographic mode the shader uses -ViewDirection as the constant
			// per-pixel view vector (rays travel from the eye toward the focus).
			const XMVECTOR ViewDir = XMVector3Normalize(
				XMLoadFloat3(&Root.CameraPosition) - XMLoadFloat3(&Root.CameraFocusPosition));
			XMFLOAT3 ViewDirF;
			XMStoreFloat3(&ViewDirF, ViewDir);
			CPUAddress->ViewDirection = XMFLOAT4(ViewDirF.x, ViewDirF.y, ViewDirF.z, 0.0f);
			CPUAddress->bOrthographic = Root.bOrthographicProjection ? 1 : 0;

			CPUAddress->MaterialMode = Root.MaterialMode;
			CPUAddress->IBLMode = GroupIBLMode;
			CPUAddress->NumFrames = Root.NumFrames;
			CPUAddress->bUseRoughnessLUT = (Root.bUseRoughnessLUT && Root.CoeffTable.Valid) ? 1 : 0;
			CPUAddress->SHECoeffLUTCount = Root.CoeffTable.Valid ? Root.CoeffTable.BakeRoughnessCount : 0;
			CPUAddress->SHECoeffLUTRoughnessStart = Root.CoeffTable.RoughnessStart;
			CPUAddress->SHECoeffLUTRoughnessEnd = Root.CoeffTable.RoughnessEnd;
			CPUAddress->bUsePCA = (Root.bUsePCA && Root.CoeffTable.Valid) ? 1 : 0;
			CPUAddress->SHEPCABasisCount = (Root.bUsePCA && Root.CoeffTable.Valid) ? Root.ActivePCABasisCount : 0;
			CPUAddress->ClassicSHOrder = Root.ClassicSHOrder;
			CPUAddress->SGLobeCount = Root.SGLobeCount;
			CPUAddress->SGAxisMode = Root.SGAxisMode;
			CPUAddress->SHEPOrder = Root.CoeffTable.Valid ? Root.CoeffTable.POrder : Root.SHEPOrder;
			CPUAddress->SHEQOrder = Root.CoeffTable.Valid ? Root.CoeffTable.QOrder : Root.SHEQOrder;

			CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
			CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
			AllocateGPUDescriptors(Gfx, 10, TableBaseCPU, TableBaseGPU);

			D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
			CBVDesc.BufferLocation = GPUAddress;
			CBVDesc.SizeInBytes = (uint32_t)sizeof(FPerFrameConstantData);

			Gfx.Device->CreateConstantBufferView(&CBVDesc, TableBaseCPU);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.SHESHCoeffCBV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.IrradianceMapSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.PrefilteredEnvMapSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.EnvMapSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.BRDFIntegrationMapSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.AccumulationBufferSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.SHECoeffLUTSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.ClassicSHCoeffSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, Root.SGLobesSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
			TableBaseCPU.Offset(Gfx.DescriptorSize);

			CmdList->SetGraphicsRootDescriptorTable(1, TableBaseGPU);
		}

		const auto NumMeshInstances = (uint32_t)Root.StaticMeshInstances.size();

		D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
		auto *CPUAddress = (FPerDrawConstantData *)AllocateGPUMemory(Gfx, NumMeshInstances * sizeof(FPerDrawConstantData), GPUAddress);

		for (uint32_t MeshInstIdx = 0; MeshInstIdx < NumMeshInstances; ++MeshInstIdx)
		{
			const FStaticMeshInstance &MeshInst = Root.StaticMeshInstances[MeshInstIdx];
			if (MeshInst.IBLMode != GroupIBLMode)
			{
				continue;
			}
			const FStaticMesh &Mesh = Root.StaticMeshes[MeshInst.MeshIndex];

			const XMMATRIX ObjectToWorld =
				XMMatrixRotationRollPitchYaw(MeshInst.Rotation.x, MeshInst.Rotation.y, MeshInst.Rotation.z) *
				XMMatrixTranslation(MeshInst.Position.x, MeshInst.Position.y, MeshInst.Position.z) *
				Billboard;

			XMStoreFloat4x4(&CPUAddress->ObjectToClip, XMMatrixTranspose(ObjectToWorld * WorldToClip));
			{
				const XMMATRIX ObjectToWorldT = XMMatrixTranspose(ObjectToWorld);
				XMStoreFloat4((XMFLOAT4 *)&CPUAddress->ObjectToWorld, ObjectToWorldT.r[0]);
				XMStoreFloat4((XMFLOAT4 *)&CPUAddress->ObjectToWorld + 1, ObjectToWorldT.r[1]);
				XMStoreFloat4((XMFLOAT4 *)&CPUAddress->ObjectToWorld + 2, ObjectToWorldT.r[2]);
			}

			CPUAddress->Albedo = Root.AlbedoColor;
			CPUAddress->Metallic = MeshInst.Metallic;
			CPUAddress->Roughness = MeshInst.Roughness;
			CPUAddress->AO = 1.0f;

			CmdList->SetGraphicsRootConstantBufferView(0, GPUAddress);
			CmdList->DrawIndexedInstanced(Mesh.IndexCount, 1, Mesh.StartIndexLocation, Mesh.BaseVertexLocation, 0);

			GPUAddress += sizeof(FPerDrawConstantData);
			CPUAddress++;
		}
		} // for each IBL mode group
	}

	// Copy color buffer to accumulation buffer.
	{
		CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.AccumulationBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST)));
		CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.MSColorBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)));
		
		if (Root.NumSamples > 1)
			CmdList->ResolveSubresource(Root.AccumulationBuffer, 0, Root.MSColorBuffer, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
		else
			CmdList->CopyResource(Root.AccumulationBuffer, Root.MSColorBuffer);
		

		CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.AccumulationBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
		CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.MSColorBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)));
	}

	// Draw EnvMap.
	{
		CmdList->SetPipelineState(Root.Pipelines[PSO_SampleEnvMap]);
		CmdList->SetGraphicsRootSignature(Root.RootSignatures[PSO_SampleEnvMap]);

		D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
		auto *CPUAddress = (FPerDrawConstantData *)AllocateGPUMemory(Gfx, sizeof(FPerDrawConstantData), GPUAddress);

		XMMATRIX ViewTransformOrigin = ViewTransform;
		ViewTransformOrigin.r[3] = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);

		// Scale the unit sphere so its interior shell encloses the whole orthographic frustum.
		// Radius must cover the frustum half-diagonal and stay inside the far plane (100).
		const float SkySphereScale = 50.0f;
		const XMMATRIX ObjectToClip = XMMatrixScaling(SkySphereScale, SkySphereScale, SkySphereScale) * ViewTransformOrigin * ProjectionTransform;
		XMStoreFloat4x4(&CPUAddress->ObjectToClip, XMMatrixTranspose(ObjectToClip));

		const FStaticMesh &Mesh = Root.StaticMeshes[MESH_Sphere];

		CmdList->SetGraphicsRootConstantBufferView(0, GPUAddress);
		CmdList->SetGraphicsRootDescriptorTable(1, CopyDescriptorsToGPUHeap(Gfx, 1, Root.EnvMapSRV));
		CmdList->DrawIndexedInstanced(Mesh.IndexCount, 1, Mesh.StartIndexLocation, Mesh.BaseVertexLocation, 0);
	}

	DrawUI(Gfx, Root.UI);

	// Resolve MS color buffer and copy it to back buffer.
	{
		ID3D12Resource *BackBuffer;
		D3D12_CPU_DESCRIPTOR_HANDLE BackBufferRTV;
		GetBackBuffer(Gfx, BackBuffer, BackBufferRTV);

		const D3D12_RESOURCE_STATES BackBufferDestState = Root.NumSamples > 1 ? D3D12_RESOURCE_STATE_RESOLVE_DEST : D3D12_RESOURCE_STATE_COPY_DEST;
		const D3D12_RESOURCE_STATES ColorBufferSrcState = Root.NumSamples > 1 ? D3D12_RESOURCE_STATE_RESOLVE_SOURCE : D3D12_RESOURCE_STATE_COPY_SOURCE;
		D3D12_RESOURCE_BARRIER Barriers[2] =
			{
				CD3DX12_RESOURCE_BARRIER::Transition(BackBuffer, D3D12_RESOURCE_STATE_PRESENT, BackBufferDestState),
				CD3DX12_RESOURCE_BARRIER::Transition(Root.MSColorBuffer, D3D12_RESOURCE_STATE_RENDER_TARGET, ColorBufferSrcState)};
		CmdList->ResourceBarrier((UINT)eastl::size(Barriers), Barriers);

		if (Root.NumSamples > 1)
			CmdList->ResolveSubresource(BackBuffer, 0, Root.MSColorBuffer, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
		else
			CmdList->CopyResource(BackBuffer, Root.MSColorBuffer);

		eastl::swap(Barriers[0].Transition.StateBefore, Barriers[0].Transition.StateAfter);
		eastl::swap(Barriers[1].Transition.StateBefore, Barriers[1].Transition.StateAfter);
		CmdList->ResourceBarrier((UINT)eastl::size(Barriers), Barriers);
	}

	CmdList->Close();

	Gfx.CmdQueue->ExecuteCommandLists(1, CommandListCast(&CmdList));
}

static void AddGraphicsPipeline(FGraphicsContext &Gfx, D3D12_GRAPHICS_PIPELINE_STATE_DESC &PSODesc, const char *VSName, const char *PSName, eastl::vector<ID3D12PipelineState *> &OutPipelines, eastl::vector<ID3D12RootSignature *> &OutSignatures)
{
	char Path[MAX_PATH];

	EA::StdC::Snprintf(Path, sizeof(Path), "Data/Shaders/%s", VSName);
	eastl::vector<uint8_t> VSBytecode = LoadFile(Path);

	EA::StdC::Snprintf(Path, sizeof(Path), "Data/Shaders/%s", PSName);
	eastl::vector<uint8_t> PSBytecode = LoadFile(Path);

	ID3D12RootSignature *RootSignature;
	VHR(Gfx.Device->CreateRootSignature(0, VSBytecode.data(), VSBytecode.size(), IID_PPV_ARGS(&RootSignature)));

	PSODesc.pRootSignature = RootSignature;
	PSODesc.VS = {VSBytecode.data(), VSBytecode.size()};
	PSODesc.PS = {PSBytecode.data(), PSBytecode.size()};

	ID3D12PipelineState *Pipeline;
	VHR(Gfx.Device->CreateGraphicsPipelineState(&PSODesc, IID_PPV_ARGS(&Pipeline)));
	OutPipelines.push_back(Pipeline);
	OutSignatures.push_back(RootSignature);
}

static void AddComputePipeline(FGraphicsContext &Gfx, const char *CSName, eastl::vector<ID3D12PipelineState *> &OutPipelines, eastl::vector<ID3D12RootSignature *> &OutSignatures)
{
	char Path[MAX_PATH];

	EA::StdC::Snprintf(Path, sizeof(Path), "Data/Shaders/%s", CSName);
	eastl::vector<uint8_t> CSBytecode = LoadFile(Path);

	ID3D12RootSignature *RootSignature;
	VHR(Gfx.Device->CreateRootSignature(0, CSBytecode.data(), CSBytecode.size(), IID_PPV_ARGS(&RootSignature)));

	D3D12_COMPUTE_PIPELINE_STATE_DESC PSODesc = {};
	PSODesc.pRootSignature = RootSignature;
	PSODesc.CS = {CSBytecode.data(), CSBytecode.size()};

	ID3D12PipelineState *Pipeline;
	VHR(Gfx.Device->CreateComputePipelineState(&PSODesc, IID_PPV_ARGS(&Pipeline)));
	OutPipelines.push_back(Pipeline);
	OutSignatures.push_back(RootSignature);
}

static void CreatePipelines(FGraphicsContext &Gfx, uint32_t NumSamples, eastl::vector<ID3D12PipelineState *> &OutPipelines, eastl::vector<ID3D12RootSignature *> &OutSignatures, uint32_t &OutWaveSize)
{
	// Detect the GPU wave size via the adapter vendor: AMD GPUs execute
	// 64-wide waves, others (NVIDIA, Intel) execute 32-wide waves.
	OutWaveSize = 32;
	{
		LUID AdapterLuid = Gfx.Device->GetAdapterLuid();
		IDXGIFactory4 *Factory = nullptr;
		if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&Factory))))
		{
			IDXGIAdapter1 *Adapter = nullptr;
			for (uint32_t Idx = 0; Factory->EnumAdapters1(Idx, &Adapter) != DXGI_ERROR_NOT_FOUND; ++Idx)
			{
				DXGI_ADAPTER_DESC1 Desc = {};
				Adapter->GetDesc1(&Desc);
				if (memcmp(&Desc.AdapterLuid, &AdapterLuid, sizeof(LUID)) == 0)
				{
					// AMD GPUs execute 64-wide waves.
					if (Desc.VendorId == 0x1002)
					{
						OutWaveSize = 64;
					}
					break;
				}
				Adapter->Release();
			}
			SAFE_RELEASE(Adapter);
			Factory->Release();
		}
	}

	const D3D12_INPUT_ELEMENT_DESC InPositionNormal[] =
		{
			{"_Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
			{"_Normal", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
		};

	// Test pipeline.
	{
		D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc = {};
		PSODesc.InputLayout = {InPositionNormal, (UINT)eastl::size(InPositionNormal)};
		PSODesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		PSODesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		PSODesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
		PSODesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
		PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		PSODesc.NumRenderTargets = 1;
		PSODesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		PSODesc.SampleMask = UINT32_MAX;
		PSODesc.SampleDesc.Count = 1;
		EA_ASSERT(OutPipelines.size() == PSO_Test);
		AddGraphicsPipeline(Gfx, PSODesc, "Test.vs.cso", "Test.ps.cso", OutPipelines, OutSignatures);
	}
	// SimpleForward pipeline.
	{
		D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc = {};
		PSODesc.InputLayout = {InPositionNormal, (UINT)eastl::size(InPositionNormal)};
		PSODesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		PSODesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
		PSODesc.RasterizerState.MultisampleEnable = NumSamples > 1 ? TRUE : FALSE;
		PSODesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		PSODesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
		PSODesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
		PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		PSODesc.NumRenderTargets = 1;
		PSODesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		PSODesc.SampleMask = UINT32_MAX;
		PSODesc.SampleDesc.Count = NumSamples;
		EA_ASSERT(OutPipelines.size() == PSO_SimpleForward);
		AddGraphicsPipeline(Gfx, PSODesc, "SimpleForward.vs.cso", "SimpleForward.ps.cso", OutPipelines, OutSignatures);
	}
	// EnvMap pipeline.
	{
		D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc = {};
		PSODesc.InputLayout = {InPositionNormal, (UINT)eastl::size(InPositionNormal)};
		PSODesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		PSODesc.RasterizerState.CullMode = D3D12_CULL_MODE_FRONT;
		PSODesc.RasterizerState.MultisampleEnable = NumSamples > 1 ? TRUE : FALSE;
		PSODesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		PSODesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
		PSODesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
		PSODesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
		PSODesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
		PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		PSODesc.NumRenderTargets = 1;
		PSODesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
		PSODesc.SampleMask = UINT32_MAX;
		PSODesc.SampleDesc.Count = NumSamples;
		EA_ASSERT(OutPipelines.size() == PSO_SampleEnvMap);
		AddGraphicsPipeline(Gfx, PSODesc, "SampleEnvMap.vs.cso", "SampleEnvMap.ps.cso", OutPipelines, OutSignatures);
	}
	// EquirectangularToCube, GenerateIrradianceMap, PrefilterEnvMap pipelines.
	{
		D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc = {};
		PSODesc.InputLayout = {InPositionNormal, (UINT)eastl::size(InPositionNormal)};
		PSODesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		PSODesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
		PSODesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		PSODesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
		PSODesc.DepthStencilState.DepthEnable = FALSE;
		PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		PSODesc.NumRenderTargets = 1;
		PSODesc.RTVFormats[0] = DXGI_FORMAT_R16G16B16A16_FLOAT;
		PSODesc.SampleMask = UINT32_MAX;
		PSODesc.SampleDesc.Count = 1;

		EA_ASSERT(OutPipelines.size() == PSO_EquirectangularToCube);
		AddGraphicsPipeline(Gfx, PSODesc, "EquirectangularToCube.vs.cso", "EquirectangularToCube.ps.cso", OutPipelines, OutSignatures);

		EA_ASSERT(OutPipelines.size() == PSO_GenerateIrradianceMap);
		AddGraphicsPipeline(Gfx, PSODesc, "GenerateIrradianceMap.vs.cso", "GenerateIrradianceMap.ps.cso", OutPipelines, OutSignatures);

		EA_ASSERT(OutPipelines.size() == PSO_PrefilterEnvMap);
		AddGraphicsPipeline(Gfx, PSODesc, "PrefilterEnvMap.vs.cso", "PrefilterEnvMap.ps.cso", OutPipelines, OutSignatures);
	}

	EA_ASSERT(OutPipelines.size() == PSO_GenerateBRDFIntegrationMap);
	AddComputePipeline(Gfx, "GenerateBRDFIntegrationMap.cs.cso", OutPipelines, OutSignatures);

	EA_ASSERT(OutPipelines.size() == PSO_SHE_Build);
	AddComputePipeline(Gfx, "SHE_Build.cs.cso", OutPipelines, OutSignatures);

	EA_ASSERT(OutPipelines.size() == PSO_SHE_Reduction);
	AddComputePipeline(Gfx, "SHE_Reduction.cs.cso", OutPipelines, OutSignatures);

	EA_ASSERT(OutPipelines.size() == PSO_SHE_Reduction_Merge);
	AddComputePipeline(Gfx, "SHE_Reduction_Merge.cs.cso", OutPipelines, OutSignatures);

	EA_ASSERT(OutPipelines.size() == PSO_SHE_Solve);
	AddComputePipeline(Gfx, "SHE_Solve.cs.cso", OutPipelines, OutSignatures);

	EA_ASSERT(OutPipelines.size() == PSO_SHE_Calibrate);
	AddComputePipeline(Gfx, "SHE_Calibrate.cs.cso", OutPipelines, OutSignatures);

	EA_ASSERT(OutPipelines.size() == PSO_SHE_Diffuse);
	// Create both wave-size variants of the SH diffuse stage-1 and finalize
	// shaders; the CPU picks the matching variant at dispatch time based on
	// the detected GPU wave size (see the top of CreatePipelines).
	AddComputePipeline(Gfx, "SHE_Diffuse_Wave32.cs.cso", OutPipelines, OutSignatures);
	AddComputePipeline(Gfx, "SHE_Diffuse_Wave64.cs.cso", OutPipelines, OutSignatures);
	AddComputePipeline(Gfx, "SHE_Diffuse_Finalize_Wave32.cs.cso", OutPipelines, OutSignatures);
	AddComputePipeline(Gfx, "SHE_Diffuse_Finalize_Wave64.cs.cso", OutPipelines, OutSignatures);
}

static void CreateEnvMap(FGraphicsContext &Gfx, const FStaticMesh &Cube, const char *EnvironmentMapPath, ID3D12Resource *&OutEnvMap, D3D12_CPU_DESCRIPTOR_HANDLE &OutEnvMapSRV, eastl::vector<ID3D12Resource *> &OutTempResources)
{
	int Width, Height;
	D3D12_SUBRESOURCE_DATA ImageData = {};
	stbi_set_flip_vertically_on_load(1);
	ImageData.pData = stbi_loadf(EnvironmentMapPath, &Width, &Height, nullptr, 3);
	stbi_set_flip_vertically_on_load(0);
	ImageData.RowPitch = Width * sizeof(XMFLOAT3);
	EA_ASSERT(ImageData.pData);

	ID3D12Resource *TempHDRRectTexture;
	D3D12_CPU_DESCRIPTOR_HANDLE TempHDRRectTextureSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		const auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32_FLOAT, Width, Height, 1, 1);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&TempHDRRectTexture)));
		OutTempResources.push_back(TempHDRRectTexture);

		Gfx.Device->CreateShaderResourceView(TempHDRRectTexture, nullptr, TempHDRRectTextureSRV);
	}

	ID3D12Resource *StagingBuffer;
	{
		const auto BufferDesc = CD3DX12_RESOURCE_DESC::Buffer(GetRequiredIntermediateSize(TempHDRRectTexture, 0, 1));
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD)), D3D12_HEAP_FLAG_NONE, &BufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&StagingBuffer)));
		OutTempResources.push_back(StagingBuffer);
	}

	const uint32_t CubeMapResolution = 512;
	ID3D12Resource *TempCubeMap;
	const D3D12_CPU_DESCRIPTOR_HANDLE TempCubeMapRTVs = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 6);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R16G16B16A16_FLOAT, CubeMapResolution, CubeMapResolution, 6);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&OutEnvMap)));

		OutEnvMapSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);

		D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
		SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
		SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		SRVDesc.TextureCube.MipLevels = (uint32_t)-1;
		Gfx.Device->CreateShaderResourceView(OutEnvMap, &SRVDesc, OutEnvMapSRV);

		Desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&TempCubeMap)));
		OutTempResources.push_back(TempCubeMap);

		D3D12_CPU_DESCRIPTOR_HANDLE CPUHandle = TempCubeMapRTVs;

		for (uint32_t Idx = 0; Idx < 6; ++Idx)
		{
			D3D12_RENDER_TARGET_VIEW_DESC RTVDesc = {};
			RTVDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
			RTVDesc.Texture2DArray.ArraySize = 1;
			RTVDesc.Texture2DArray.FirstArraySlice = Idx;
			Gfx.Device->CreateRenderTargetView(TempCubeMap, &RTVDesc, CPUHandle);

			CPUHandle.ptr += Gfx.DescriptorSizeRTV;
		}
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	UpdateSubresources<1>(CmdList, TempHDRRectTexture, StagingBuffer, 0, 0, 1, &ImageData);

	stbi_image_free((void *)ImageData.pData);

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(TempHDRRectTexture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));

	CmdList->RSSetViewports(1, get_rvalue_ptr(CD3DX12_VIEWPORT(0.0f, 0.0f, (float)CubeMapResolution, (float)CubeMapResolution)));
	CmdList->RSSetScissorRects(1, get_rvalue_ptr(CD3DX12_RECT(0, 0, CubeMapResolution, CubeMapResolution)));

	CmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	const XMMATRIX ViewTransforms[6] =
		{
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(-1.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
		};
	const XMMATRIX ProjectionTransform = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.1f, 10.0f);

	D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
	auto *CPUAddress = (FPerDrawConstantData *)AllocateGPUMemory(Gfx, 6 * sizeof(FPerDrawConstantData), GPUAddress);

	D3D12_CPU_DESCRIPTOR_HANDLE RTV = TempCubeMapRTVs;

	for (uint32_t Idx = 0; Idx < 6; ++Idx)
	{
		CmdList->OMSetRenderTargets(1, &RTV, TRUE, nullptr);

		const XMMATRIX ObjectToClip = ViewTransforms[Idx] * ProjectionTransform;
		XMStoreFloat4x4(&CPUAddress->ObjectToClip, XMMatrixTranspose(ObjectToClip));

		CmdList->SetGraphicsRootConstantBufferView(0, GPUAddress);
		CmdList->SetGraphicsRootDescriptorTable(1, CopyDescriptorsToGPUHeap(Gfx, 1, TempHDRRectTextureSRV));
		CmdList->DrawIndexedInstanced(Cube.IndexCount, 1, Cube.StartIndexLocation, Cube.BaseVertexLocation, 0);

		RTV.ptr += Gfx.DescriptorSizeRTV;
		GPUAddress += sizeof(FPerDrawConstantData);
		CPUAddress++;
	}

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(TempCubeMap, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)));

	CmdList->CopyResource(OutEnvMap, TempCubeMap);

	Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(OutEnvMap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
}

static void CreateIrradianceMap(FGraphicsContext &Gfx, D3D12_CPU_DESCRIPTOR_HANDLE EnvMapSRV, const FStaticMesh &Cube, ID3D12Resource *&OutIrradianceMap, D3D12_CPU_DESCRIPTOR_HANDLE &OutIrradianceMapSRV, eastl::vector<ID3D12Resource *> &OutTempResources)
{
	const uint32_t CubeMapResolution = 64;
	ID3D12Resource *TempCubeMap;
	const D3D12_CPU_DESCRIPTOR_HANDLE TempCubeMapRTVs = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 6);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R16G16B16A16_FLOAT, CubeMapResolution, CubeMapResolution, 6);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&OutIrradianceMap)));

		OutIrradianceMapSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);

		D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
		SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
		SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		SRVDesc.TextureCube.MipLevels = (uint32_t)-1;
		Gfx.Device->CreateShaderResourceView(OutIrradianceMap, &SRVDesc, OutIrradianceMapSRV);

		Desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&TempCubeMap)));
		OutTempResources.push_back(TempCubeMap);

		D3D12_CPU_DESCRIPTOR_HANDLE CPUHandle = TempCubeMapRTVs;

		for (uint32_t Idx = 0; Idx < 6; ++Idx)
		{
			D3D12_RENDER_TARGET_VIEW_DESC RTVDesc = {};
			RTVDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
			RTVDesc.Texture2DArray.ArraySize = 1;
			RTVDesc.Texture2DArray.FirstArraySlice = Idx;
			Gfx.Device->CreateRenderTargetView(TempCubeMap, &RTVDesc, CPUHandle);

			CPUHandle.ptr += Gfx.DescriptorSizeRTV;
		}
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	CmdList->RSSetViewports(1, get_rvalue_ptr(CD3DX12_VIEWPORT(0.0f, 0.0f, (float)CubeMapResolution, (float)CubeMapResolution)));
	CmdList->RSSetScissorRects(1, get_rvalue_ptr(CD3DX12_RECT(0, 0, CubeMapResolution, CubeMapResolution)));

	CmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	const XMMATRIX ViewTransforms[6] =
		{
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(-1.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
		};
	const XMMATRIX ProjectionTransform = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.1f, 10.0f);

	D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
	auto *CPUAddress = (FPerDrawConstantData *)AllocateGPUMemory(Gfx, 6 * sizeof(FPerDrawConstantData), GPUAddress);

	D3D12_CPU_DESCRIPTOR_HANDLE RTV = TempCubeMapRTVs;

	for (uint32_t Idx = 0; Idx < 6; ++Idx)
	{
		CmdList->OMSetRenderTargets(1, &RTV, TRUE, nullptr);

		const XMMATRIX ObjectToClip = ViewTransforms[Idx] * ProjectionTransform;
		XMStoreFloat4x4(&CPUAddress->ObjectToClip, XMMatrixTranspose(ObjectToClip));

		CmdList->SetGraphicsRootConstantBufferView(0, GPUAddress);
		CmdList->SetGraphicsRootDescriptorTable(1, CopyDescriptorsToGPUHeap(Gfx, 1, EnvMapSRV));
		CmdList->DrawIndexedInstanced(Cube.IndexCount, 1, Cube.StartIndexLocation, Cube.BaseVertexLocation, 0);

		RTV.ptr += Gfx.DescriptorSizeRTV;
		GPUAddress += sizeof(FPerDrawConstantData);
		CPUAddress++;
	}

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(TempCubeMap, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)));

	CmdList->CopyResource(OutIrradianceMap, TempCubeMap);

	Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(OutIrradianceMap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
}

static void CreatePrefilteredEnvMap(FGraphicsContext &Gfx, D3D12_CPU_DESCRIPTOR_HANDLE EnvMapSRV, const FStaticMesh &Cube, ID3D12Resource *&OutPrefilteredEnvMap, D3D12_CPU_DESCRIPTOR_HANDLE &OutPrefilteredEnvMapSRV, eastl::vector<ID3D12Resource *> &OutTempResources)
{
	const uint32_t CubeMapResolution = 256;
	const uint32_t NumMipLevelsUsed = 6; // 256, 128, 64, 32, 16, 8

	ID3D12Resource *TempCubeMap;
	const D3D12_CPU_DESCRIPTOR_HANDLE TempCubeMapRTVs = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 6 * NumMipLevelsUsed);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R16G16B16A16_FLOAT, CubeMapResolution, CubeMapResolution, 6, NumMipLevelsUsed);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&OutPrefilteredEnvMap)));

		OutPrefilteredEnvMapSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);

		D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
		SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
		SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		SRVDesc.TextureCube.MipLevels = NumMipLevelsUsed;
		Gfx.Device->CreateShaderResourceView(OutPrefilteredEnvMap, &SRVDesc, OutPrefilteredEnvMapSRV);

		Desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&TempCubeMap)));
		OutTempResources.push_back(TempCubeMap);

		D3D12_CPU_DESCRIPTOR_HANDLE CPUHandle = TempCubeMapRTVs;

		for (uint32_t MipSliceIdx = 0; MipSliceIdx < NumMipLevelsUsed; ++MipSliceIdx)
		{
			for (uint32_t ArraySliceIdx = 0; ArraySliceIdx < 6; ++ArraySliceIdx)
			{
				D3D12_RENDER_TARGET_VIEW_DESC RTVDesc = {};
				RTVDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
				RTVDesc.Texture2DArray.ArraySize = 1;
				RTVDesc.Texture2DArray.FirstArraySlice = ArraySliceIdx;
				RTVDesc.Texture2DArray.MipSlice = MipSliceIdx;
				Gfx.Device->CreateRenderTargetView(TempCubeMap, &RTVDesc, CPUHandle);

				CPUHandle.ptr += Gfx.DescriptorSizeRTV;
			}
		}
	}

	const XMMATRIX ViewTransforms[6] =
		{
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(-1.0f, 0.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, -1.0f, 0.0f, 0.0f), XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
			XMMatrixLookToLH(XMVectorZero(), XMVectorSet(0.0f, 0.0f, -1.0f, 0.0f), XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f)),
		};
	const XMMATRIX ProjectionTransform = XMMatrixPerspectiveFovLH(XM_PIDIV2, 1.0f, 0.1f, 10.0f);

	D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
	auto *CPUAddress = (FPerDrawConstantData *)AllocateGPUMemory(Gfx, 6 * NumMipLevelsUsed * sizeof(FPerDrawConstantData), GPUAddress);

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	CmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

	uint32_t CurrentResolution = CubeMapResolution;
	D3D12_CPU_DESCRIPTOR_HANDLE RTV = TempCubeMapRTVs;

	for (uint32_t MipSliceIdx = 0; MipSliceIdx < NumMipLevelsUsed; ++MipSliceIdx)
	{
		CmdList->RSSetViewports(1, get_rvalue_ptr(CD3DX12_VIEWPORT(0.0f, 0.0f, (float)CurrentResolution, (float)CurrentResolution)));
		CmdList->RSSetScissorRects(1, get_rvalue_ptr(CD3DX12_RECT(0, 0, CurrentResolution, CurrentResolution)));

		for (uint32_t ArraySliceIdx = 0; ArraySliceIdx < 6; ++ArraySliceIdx)
		{
			CmdList->OMSetRenderTargets(1, &RTV, TRUE, nullptr);

			const XMMATRIX ObjectToClip = ViewTransforms[ArraySliceIdx] * ProjectionTransform;
			XMStoreFloat4x4(&CPUAddress->ObjectToClip, XMMatrixTranspose(ObjectToClip));
			CPUAddress->Roughness = (float)MipSliceIdx / (NumMipLevelsUsed - 1);

			CmdList->SetGraphicsRootConstantBufferView(0, GPUAddress);
			CmdList->SetGraphicsRootDescriptorTable(1, CopyDescriptorsToGPUHeap(Gfx, 1, EnvMapSRV));
			CmdList->DrawIndexedInstanced(Cube.IndexCount, 1, Cube.StartIndexLocation, Cube.BaseVertexLocation, 0);

			RTV.ptr += Gfx.DescriptorSizeRTV;
			GPUAddress += sizeof(FPerDrawConstantData);
			CPUAddress++;
		}

		CurrentResolution /= 2;
	}

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(TempCubeMap, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)));

	CmdList->CopyResource(OutPrefilteredEnvMap, TempCubeMap);

	Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(OutPrefilteredEnvMap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
}

static void CreateBRDFIntegrationMap(FGraphicsContext &Gfx, ID3D12Resource *&OutBRDFIntegrationMap, D3D12_CPU_DESCRIPTOR_HANDLE &OutBRDFIntegrationMapSRV, eastl::vector<ID3D12Resource *> &OutTempResources)
{
	const uint32_t MapResolution = 512;
	EA_ASSERT(EA::StdC::IsPowerOf2(MapResolution));

	ID3D12Resource *TempTexture;
	const D3D12_CPU_DESCRIPTOR_HANDLE TempTextureUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R16G16_FLOAT, MapResolution, MapResolution, 1, 1);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&OutBRDFIntegrationMap)));

		OutBRDFIntegrationMapSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);

		Gfx.Device->CreateShaderResourceView(OutBRDFIntegrationMap, nullptr, OutBRDFIntegrationMapSRV);

		Desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&TempTexture)));
		OutTempResources.push_back(TempTexture);

		Gfx.Device->CreateUnorderedAccessView(TempTexture, nullptr, nullptr, TempTextureUAV);
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	CmdList->SetComputeRootDescriptorTable(0, CopyDescriptorsToGPUHeap(Gfx, 1, TempTextureUAV));
	CmdList->Dispatch(MapResolution / 8, MapResolution / 8, 1);

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(TempTexture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)));

	CmdList->CopyResource(OutBRDFIntegrationMap, TempTexture);

	Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(OutBRDFIntegrationMap, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
}

static void SHEBuild(const FDemoRoot &Root, FGraphicsContext &Gfx, ID3D12Resource *&OutSHEMatrixA, ID3D12Resource *&OutSHEMatrixb, D3D12_CPU_DESCRIPTOR_HANDLE &OutSHEMatrixASRV, D3D12_CPU_DESCRIPTOR_HANDLE &OutSHEMatrixbSRV, D3D12_CPU_DESCRIPTOR_HANDLE EnvMapSRV)
{
	const uint32_t ViewCountSqrt = 8;
	const uint32_t NormalCountSqrt = 8;
	const uint32_t ViewCount = ViewCountSqrt * ViewCountSqrt;
	const uint32_t NormalCount = NormalCountSqrt * NormalCountSqrt;
	const uint32_t RoughnessCount = (uint32_t)Root.BakeRoughnessCount;
	const uint32_t SphericalHarmonicCount = SHE_SPEC_COEFFS; // fixed 124-slot P/Q layout.

	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHEMatrixAUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32_FLOAT, ViewCount * SphericalHarmonicCount, NormalCount * RoughnessCount, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&OutSHEMatrixA)));
		Gfx.Device->CreateUnorderedAccessView(OutSHEMatrixA, nullptr, nullptr, TempSHEMatrixAUAV);

		OutSHEMatrixASRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
		Gfx.Device->CreateShaderResourceView(OutSHEMatrixA, nullptr, OutSHEMatrixASRV);
	}

	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHEMatrixbUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R16G16B16A16_FLOAT, ViewCount, NormalCount * RoughnessCount, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&OutSHEMatrixb)));
		Gfx.Device->CreateUnorderedAccessView(OutSHEMatrixb, nullptr, nullptr, TempSHEMatrixbUAV);

		OutSHEMatrixbSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
		Gfx.Device->CreateShaderResourceView(OutSHEMatrixb, nullptr, OutSHEMatrixbSRV);
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
	auto *CPUAddress = (FSHEBuildConstantData *)AllocateGPUMemory(Gfx, sizeof(FSHEBuildConstantData), GPUAddress);

	CPUAddress->ViewCount = ViewCount;
	CPUAddress->NormalCount = NormalCount;
	CPUAddress->ViewCountSqrt = ViewCountSqrt;
	CPUAddress->NormalCountSqrt = NormalCountSqrt;
	CPUAddress->RoughnessCount = RoughnessCount;
	CPUAddress->SphericalHarmonicCount = SphericalHarmonicCount;
	CPUAddress->RoughnessStart = Root.BakeRoughnessStart;
	CPUAddress->RoughnessEnd = Root.BakeRoughnessEnd;
	CPUAddress->bRandomRoughness = Root.bRandomRoughness ? 1 : 0;
	CPUAddress->POrder = Root.SHEPOrder;
	CPUAddress->QOrder = Root.SHEQOrder;

	CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
	CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
	AllocateGPUDescriptors(Gfx, 4, TableBaseCPU, TableBaseGPU);

	D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
	CBVDesc.BufferLocation = GPUAddress;
	CBVDesc.SizeInBytes = sizeof(FSHEBuildConstantData);

	Gfx.Device->CreateConstantBufferView(&CBVDesc, TableBaseCPU);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, EnvMapSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, TempSHEMatrixAUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, TempSHEMatrixbUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);

	CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);
	CmdList->Dispatch(8, 8, RoughnessCount);

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
OutSHEMatrixA, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)));

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
OutSHEMatrixb, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)));
}

static void SHEReduction(FDemoRoot &Root, FGraphicsContext &Gfx, ID3D12Resource *&OutSHEMatrixAT, D3D12_CPU_DESCRIPTOR_HANDLE &OutSHEMatrixATSRV, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixASRV, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixbSRV,
						 eastl::vector<ID3D12Resource *> &OutTempResources)
{
	const uint32_t ViewCountSqrt = 8;
	const uint32_t NormalCountSqrt = 8;
	const uint32_t ViewCount = ViewCountSqrt * ViewCountSqrt;
	const uint32_t NormalCount = NormalCountSqrt * NormalCountSqrt;
	const uint32_t RoughnessCount = (uint32_t)Root.BakeRoughnessCount;
	const uint32_t SphericalHarmonicCount = SHE_SPEC_COEFFS; // fixed 124-slot P/Q layout.
	const uint32_t GroupCountZ = 64;

	ID3D12Resource *TempSHEMatrixAT3D;
	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHEMatrixAT3DUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHEMatrixAT3DSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex3D(
			DXGI_FORMAT_R32_FLOAT,
			SphericalHarmonicCount * 2, // Width
			SphericalHarmonicCount + 3, // Height
			GroupCountZ,				// Depth
			1,							// MipLevels
			D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		VHR(Gfx.Device->CreateCommittedResource(
			get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)),
			D3D12_HEAP_FLAG_NONE,
			&Desc,
			D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
			nullptr,
			IID_PPV_ARGS(&TempSHEMatrixAT3D)));
		OutTempResources.push_back(TempSHEMatrixAT3D);

		Gfx.Device->CreateUnorderedAccessView(TempSHEMatrixAT3D, nullptr, nullptr, TempSHEMatrixAT3DUAV);
		Gfx.Device->CreateShaderResourceView(TempSHEMatrixAT3D, nullptr, TempSHEMatrixAT3DSRV);
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
	auto *CPUAddress = (FSHEReductionConstantData *)AllocateGPUMemory(Gfx, sizeof(FSHEReductionConstantData), GPUAddress);

	CPUAddress->ViewCount = ViewCount;
	CPUAddress->ElementCount = ViewCount * NormalCount * RoughnessCount;
	CPUAddress->GroupCountZ = GroupCountZ;
	CPUAddress->SphericalHarmonicCount = SphericalHarmonicCount;

	CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
	CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
	AllocateGPUDescriptors(Gfx, 4, TableBaseCPU, TableBaseGPU);

	D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
	CBVDesc.BufferLocation = GPUAddress;
	CBVDesc.SizeInBytes = sizeof(FSHEReductionConstantData);

	Gfx.Device->CreateConstantBufferView(&CBVDesc, TableBaseCPU);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, SHEMatrixASRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, SHEMatrixbSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, TempSHEMatrixAT3DUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);

	CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);
	CmdList->Dispatch(1, 1, GroupCountZ);

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
TempSHEMatrixAT3D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)));

	// Merge
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Reduction_Merge]);
	Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Reduction_Merge]);

	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHEMatrixATUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32_FLOAT, SphericalHarmonicCount * 2, SphericalHarmonicCount + 3, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&OutSHEMatrixAT)));
		Gfx.Device->CreateUnorderedAccessView(OutSHEMatrixAT, nullptr, nullptr, TempSHEMatrixATUAV);

		OutSHEMatrixATSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
		Gfx.Device->CreateShaderResourceView(OutSHEMatrixAT, nullptr, OutSHEMatrixATSRV);
	}

	{
		D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
		auto *CPUAddress = (FSHEReductionConstantData *)AllocateGPUMemory(Gfx, sizeof(FSHEReductionConstantData), GPUAddress);

		CPUAddress->ViewCount = ViewCount;
		CPUAddress->ElementCount = ViewCount * NormalCount * RoughnessCount;
		CPUAddress->GroupCountZ = GroupCountZ;
		CPUAddress->SphericalHarmonicCount = SphericalHarmonicCount;

		CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
		CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
		AllocateGPUDescriptors(Gfx, 3, TableBaseCPU, TableBaseGPU);

		D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
		CBVDesc.BufferLocation = GPUAddress;
		CBVDesc.SizeInBytes = sizeof(FSHEReductionConstantData);

		Gfx.Device->CreateConstantBufferView(&CBVDesc, TableBaseCPU);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, TempSHEMatrixAT3DSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, TempSHEMatrixATUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);
		CmdList->Dispatch(1, 1, 1);

		CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
OutSHEMatrixAT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)));
	}
}

static void SHESolve(FGraphicsContext &Gfx, ID3D12Resource *&OutSHESHCoeff, D3D12_CPU_DESCRIPTOR_HANDLE &OutSHESHCoeffCBV, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixAT)
{
	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHESHCoeffUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		auto Desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(FSHECoefficientConstantData), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&OutSHESHCoeff)));
		D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
		UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		UAVDesc.Buffer.NumElements = SHE_LUT_STRIDE;
		UAVDesc.Buffer.StructureByteStride = sizeof(XMFLOAT4);
		UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		Gfx.Device->CreateUnorderedAccessView(OutSHESHCoeff, nullptr, &UAVDesc, TempSHESHCoeffUAV);

		OutSHESHCoeffCBV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
		D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
		CBVDesc.BufferLocation = OutSHESHCoeff->GetGPUVirtualAddress();
		CBVDesc.SizeInBytes = sizeof(FSHECoefficientConstantData);
		Gfx.Device->CreateConstantBufferView(&CBVDesc, OutSHESHCoeffCBV);
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
	CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
	AllocateGPUDescriptors(Gfx, 2, TableBaseCPU, TableBaseGPU);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, SHEMatrixAT, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, TempSHESHCoeffUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);
	CmdList->Dispatch(1, 1, 1);
}

static void SHECalibrate(const FDemoRoot &Root, FGraphicsContext &Gfx, ID3D12Resource *SHESHCoeff, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixASRV, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixbSRV)
{
	const uint32_t ViewCountSqrt = 8;
	const uint32_t NormalCountSqrt = 8;
	const uint32_t ViewCount = ViewCountSqrt * ViewCountSqrt;
	const uint32_t NormalCount = NormalCountSqrt * NormalCountSqrt;
	const uint32_t RoughnessCount = (uint32_t)Root.BakeRoughnessCount;
	const uint32_t SphericalHarmonicCount = SHE_SPEC_COEFFS;

	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHESHCoeffUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
		UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		UAVDesc.Buffer.NumElements = SHE_LUT_STRIDE;
		UAVDesc.Buffer.StructureByteStride = sizeof(XMFLOAT4);
		UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		Gfx.Device->CreateUnorderedAccessView(SHESHCoeff, nullptr, &UAVDesc, TempSHESHCoeffUAV);
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::UAV(SHESHCoeff)));

	D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
	auto *CPUAddress = (FSHEReductionConstantData *)AllocateGPUMemory(Gfx, sizeof(FSHEReductionConstantData), GPUAddress);

	CPUAddress->ViewCount = ViewCount;
	CPUAddress->ElementCount = ViewCount * NormalCount * RoughnessCount;
	CPUAddress->GroupCountZ = 1;
	CPUAddress->SphericalHarmonicCount = SphericalHarmonicCount;

	CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
	CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
	AllocateGPUDescriptors(Gfx, 4, TableBaseCPU, TableBaseGPU);

	D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
	CBVDesc.BufferLocation = GPUAddress;
	CBVDesc.SizeInBytes = sizeof(FSHEReductionConstantData);

	Gfx.Device->CreateConstantBufferView(&CBVDesc, TableBaseCPU);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, SHEMatrixASRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, SHEMatrixbSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, TempSHESHCoeffUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
	TableBaseCPU.Offset(Gfx.DescriptorSize);

	CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);
	CmdList->Dispatch(1, 1, 1);

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
									SHESHCoeff, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)));
}

static void SHEDiffuse(FDemoRoot &Root, FGraphicsContext &Gfx, ID3D12Resource *SHESHCoeff, D3D12_CPU_DESCRIPTOR_HANDLE EnvMapSRV, eastl::vector<ID3D12Resource *> &OutTempResources)
{
	// Two-stage parallel projection of the environment map onto 3-band SH:
	//   Stage 1: one group == one wave (WaveSize lanes), each thread integrates
	//            16 Hammersley samples, WaveActiveSum reduces the group, lane 0
	//            writes 9 float3 partials per group.
	//   Stage 2: a single group sums all partials and writes the final
	//            coefficients (with 4*pi/N normalization and Ramamoorthi
	//            cosine-band convolution) into SHESHCoeff slots
	//            [SHE_DIFFUSE_BASE .. SHE_DIFFUSE_BASE+8].
	const uint32_t TotalSampleCount = 65536;
	const uint32_t SamplesPerThread = 16;
	const uint32_t ThreadCountX = Root.WaveSize; // group size == wave size
	const uint32_t GroupCount = TotalSampleCount / (SamplesPerThread * ThreadCountX); // 128 (wave32) / 64 (wave64)

	// Whole-buffer UAV over SHESHCoeff: the finalize pass writes diffuse slots
	// with absolute indices, leaving the specular/autoBias slots untouched.
	const D3D12_CPU_DESCRIPTOR_HANDLE SHESHCoeffUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
		UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		UAVDesc.Buffer.NumElements = SHE_COEFF_BUFFER_SIZE;
		UAVDesc.Buffer.StructureByteStride = sizeof(XMFLOAT4);
		UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		Gfx.Device->CreateUnorderedAccessView(SHESHCoeff, nullptr, &UAVDesc, SHESHCoeffUAV);
	}

	// Stage-1 partial buffer: GroupCount x 9 x float3.
	ID3D12Resource *PartialRadiance;
	const D3D12_CPU_DESCRIPTOR_HANDLE PartialRadianceUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	const D3D12_CPU_DESCRIPTOR_HANDLE PartialRadianceSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		const D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(GroupCount * 9 * sizeof(XMFLOAT3), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&PartialRadiance)));
		OutTempResources.push_back(PartialRadiance);

		D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
		UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		UAVDesc.Buffer.NumElements = GroupCount * 9;
		UAVDesc.Buffer.StructureByteStride = sizeof(XMFLOAT3);
		UAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		Gfx.Device->CreateUnorderedAccessView(PartialRadiance, nullptr, &UAVDesc, PartialRadianceUAV);

		D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
		SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
		SRVDesc.Format = DXGI_FORMAT_UNKNOWN;
		SRVDesc.Buffer.NumElements = GroupCount * 9;
		SRVDesc.Buffer.StructureByteStride = sizeof(XMFLOAT3);
		SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		Gfx.Device->CreateShaderResourceView(PartialRadiance, &SRVDesc, PartialRadianceSRV);
	}

	// Upload the shared stage parameters.
	D3D12_GPU_VIRTUAL_ADDRESS GPUAddress;
	auto *CPUAddress = (FSHEDiffuseParams *)AllocateGPUMemory(Gfx, sizeof(FSHEDiffuseParams), GPUAddress);
	CPUAddress->TotalSampleCount = TotalSampleCount;
	CPUAddress->SamplesPerThread = SamplesPerThread;
	CPUAddress->ThreadCountX = ThreadCountX;

	const D3D12_CPU_DESCRIPTOR_HANDLE ParamsCBV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		D3D12_CONSTANT_BUFFER_VIEW_DESC CBVDesc = {};
		CBVDesc.BufferLocation = GPUAddress;
		CBVDesc.SizeInBytes = sizeof(FSHEDiffuseParams);
		Gfx.Device->CreateConstantBufferView(&CBVDesc, ParamsCBV);
	}

	ID3D12GraphicsCommandList2 *CmdList = Gfx.CmdList;

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
									SHESHCoeff, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)));

	// Stage 1: partial projection, one wave per group.
	{
		const uint32_t Stage1PSO = (Root.WaveSize == 64) ? PSO_SHE_Diffuse_Wave64 : PSO_SHE_Diffuse_Wave32;
		CmdList->SetPipelineState(Root.Pipelines[Stage1PSO]);
		CmdList->SetComputeRootSignature(Root.RootSignatures[Stage1PSO]);

		CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
		CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
		AllocateGPUDescriptors(Gfx, 3, TableBaseCPU, TableBaseGPU);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, ParamsCBV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, EnvMapSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, PartialRadianceUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);
		CmdList->Dispatch(GroupCount, 1, 1);
	}

	// Transition out of UAV: flushes all stage-1 writes and puts the buffer into a
	// shader-resource state so stage 2 can read it from a compute shader.
	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
		PartialRadiance, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)));

	// Stage 2: sum partials, convolve, write back.
	{
		const uint32_t Stage2PSO = (Root.WaveSize == 64) ? PSO_SHE_Diffuse_Finalize_Wave64 : PSO_SHE_Diffuse_Finalize_Wave32;
		CmdList->SetPipelineState(Root.Pipelines[Stage2PSO]);
		CmdList->SetComputeRootSignature(Root.RootSignatures[Stage2PSO]);

		CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
		CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
		AllocateGPUDescriptors(Gfx, 3, TableBaseCPU, TableBaseGPU);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, ParamsCBV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, PartialRadianceSRV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		Gfx.Device->CopyDescriptorsSimple(1, TableBaseCPU, SHESHCoeffUAV, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		TableBaseCPU.Offset(Gfx.DescriptorSize);

		CmdList->SetComputeRootDescriptorTable(0, TableBaseGPU);
		CmdList->Dispatch(1, 1, 1);
	}

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
									SHESHCoeff, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)));
}

// Create the persistently-mapped upload buffer that holds the per-roughness
// grouped-fit coefficients (StructuredBuffer<float4>, flat [group*SHE_LUT_STRIDE + i]).
static void CreateSHECoeffLUTResources(FDemoRoot &Root)
{
	FGraphicsContext &Gfx = Root.Gfx;
	// Enough for the raw layout (G*S) or the PCA layout (S + N*S + G*N),
	// with S=SHE_LUT_STRIDE; worst case 125 + 16*125 + 16*16 = 2321 < 4096.
	const uint32_t NumFloat4 = GSHECoeffLUTMaxFloat4;
	const UINT64 BufferSize = (UINT64)NumFloat4 * sizeof(float4);
	const D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(BufferSize);
	VHR(Gfx.Device->CreateCommittedResource(
		get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD)), D3D12_HEAP_FLAG_NONE,
		&Desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&Root.SHECoeffLUT)));
	CD3DX12_RANGE ReadRange(0, 0);
	VHR(Root.SHECoeffLUT->Map(0, &ReadRange, &Root.SHECoeffLUTMapped));
	memset(Root.SHECoeffLUTMapped, 0, (size_t)BufferSize);

	Root.SHECoeffLUTSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	D3D12_SHADER_RESOURCE_VIEW_DESC S = {};
	S.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	S.Format = DXGI_FORMAT_UNKNOWN;
	S.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	S.Buffer.NumElements = NumFloat4;
	S.Buffer.StructureByteStride = sizeof(float4);
	Gfx.Device->CreateShaderResourceView(Root.SHECoeffLUT, &S, Root.SHECoeffLUTSRV);
}

// Classic SH IBL: create the persistently-mapped upload buffer that holds the
// projected radiance coefficients (StructuredBuffer<float4> of length
// SH_CLASSIC_COEFF_COUNT). Bound as SRV at t6 in the SimpleForward table.
static void CreateClassicSHCoeffResources(FDemoRoot &Root)
{
	FGraphicsContext &Gfx = Root.Gfx;
	const UINT64 BufferSize = (UINT64)SH_CLASSIC_COEFF_COUNT * sizeof(float4);
	const D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(BufferSize);
	VHR(Gfx.Device->CreateCommittedResource(
		get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD)), D3D12_HEAP_FLAG_NONE,
		&Desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&Root.ClassicSHCoeff)));
	CD3DX12_RANGE ReadRange(0, 0);
	VHR(Root.ClassicSHCoeff->Map(0, &ReadRange, &Root.ClassicSHCoeffMapped));
	memset(Root.ClassicSHCoeffMapped, 0, (size_t)BufferSize);

	Root.ClassicSHCoeffSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	D3D12_SHADER_RESOURCE_VIEW_DESC S = {};
	S.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	S.Format = DXGI_FORMAT_UNKNOWN;
	S.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	S.Buffer.NumElements = SH_CLASSIC_COEFF_COUNT;
	S.Buffer.StructureByteStride = sizeof(float4);
	Gfx.Device->CreateShaderResourceView(Root.ClassicSHCoeff, &S, Root.ClassicSHCoeffSRV);
}

// Project the current equirectangular HDR environment onto real spherical
// harmonics up to l = SH_CLASSIC_MAX_BANDS. The direction convention matches
// EquirectangularToCube.hlsl / SampleSphericalMap: with stb flipped on load,
//   phi   = 2*pi * (u - 0.5)
//   theta = pi * (0.5 - v)
//   dir   = (sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi))
// and the SH basis (SHBasis.h) uses +z as polar axis, so we pass
// (dir.z, dir.x, dir.y) to align the basis with the world-space cubemap that
// the split-sum / SHE paths sample (their directions are x/y/z world axes).
static void ProjectEnvMapToSH(FDemoRoot &Root)
{
	int Width = 0, Height = 0;
	stbi_set_flip_vertically_on_load(1);
	float *HDR = stbi_loadf(GEnvironmentMapPaths[Root.EnvironmentMapIndex], &Width, &Height, nullptr, 3);
	stbi_set_flip_vertically_on_load(0);
	EA_ASSERT(HDR && Width > 0 && Height > 0);

	const double PI_D = 3.14159265358979323846;
	const double dPhi = 2.0 * PI_D / (double)Width;
	const double dElev = PI_D / (double)Height;
	double Acc[SH_CLASSIC_COEFF_COUNT][3] = {};
	for (int iy = 0; iy < Height; ++iy)
	{
		// Row 0 of the (vertically flipped) buffer is the GPU texture row at
		// v = 0. EquirectangularToCube.hlsl maps a world direction V to
		//   u   = atan2(V.z, V.x)/(2*pi) + 0.5
		//   v   = asin(V.y)/pi + 0.5      (=> V.y = sin(e), e = pi*(v-0.5))
		// Invert that here to recover the world direction for this texel.
		const double v = (iy + 0.5) / (double)Height;
		const double e = PI_D * (v - 0.5);   // asin(V.y)
		const double sinE = sin(e);           // V.y
		const double cosE = cos(e);           // horizontal magnitude >= 0
		const double dOmega = cosE * dPhi * dElev;
		for (int ix = 0; ix < Width; ++ix)
		{
			const double u = (ix + 0.5) / (double)Width;
			const double phi = 2.0 * PI_D * (u - 0.5); // atan2(V.z, V.x)
			const float vx = (float)(cosE * cos(phi));
			const float vy = (float)sinE;
			const float vz = (float)(cosE * sin(phi));

			// The SH basis in SHBasis.h uses +z as the polar axis; evaluate it
			// at the SAME world vector used to look up the environment. This
			// keeps projection consistent with the forward-shader reconstruction
			// (which evaluates the basis at world-space N / R).
			float Basis[SH_CLASSIC_COEFF_COUNT];
			SHBasisEval(vx, vy, vz, Basis);

			const float *Pixel = HDR + (size_t)(iy * Width + ix) * 3;
			for (int k = 0; k < SH_CLASSIC_COEFF_COUNT; ++k)
			{
				Acc[k][0] += (double)Pixel[0] * Basis[k] * dOmega;
				Acc[k][1] += (double)Pixel[1] * Basis[k] * dOmega;
				Acc[k][2] += (double)Pixel[2] * Basis[k] * dOmega;
			}
		}
	}
	stbi_image_free(HDR);

	auto *Dst = (float4 *)Root.ClassicSHCoeffMapped;
	for (int k = 0; k < SH_CLASSIC_COEFF_COUNT; ++k)
	{
		Dst[k] = {(float)Acc[k][0], (float)Acc[k][1], (float)Acc[k][2], 0.0f};
	}
}

// SG IBL: create the persistently-mapped upload buffer holding the fitted
// lobes. Layout (two float4 per lobe):
//   [2*i]   = (mu.xyz, lambda)   axis (unit) + sharpness
//   [2*i+1] = (a.rgb, unused)    RGB amplitude
static void CreateSGLobeResources(FDemoRoot &Root)
{
	FGraphicsContext &Gfx = Root.Gfx;
	const uint32_t NumFloat4 = 2 * SG_MAX_LOBES;
	const UINT64 BufferSize = (UINT64)NumFloat4 * sizeof(float4);
	const D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(BufferSize);
	VHR(Gfx.Device->CreateCommittedResource(
		get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD)), D3D12_HEAP_FLAG_NONE,
		&Desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&Root.SGLobes)));
	CD3DX12_RANGE ReadRange(0, 0);
	VHR(Root.SGLobes->Map(0, &ReadRange, &Root.SGLobesMapped));
	memset(Root.SGLobesMapped, 0, (size_t)BufferSize);

	Root.SGLobesSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	D3D12_SHADER_RESOURCE_VIEW_DESC S = {};
	S.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	S.Format = DXGI_FORMAT_UNKNOWN;
	S.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	S.Buffer.NumElements = NumFloat4;
	S.Buffer.StructureByteStride = sizeof(float4);
	Gfx.Device->CreateShaderResourceView(Root.SGLobes, &S, Root.SGLobesSRV);
}

// Fit the current HDR environment as a sum of isotropic Spherical Gaussians
//   L(v) ~= sum_i a_i * exp(lambda_i * (dot(mu_i, v) - 1)).
//
// Two cooperating stages:
//   (1) axes / sharpness: a von Mises-Fisher mixture EM on the luminance-
//       weighted directional distribution. In FIXED mode the axes are pinned
//       to a uniform Fibonacci sphere; in FREE mode they move to the radiance-
//       weighted centroid. Sharpness comes from the resultant length via the
//       Banerjee vMF concentration estimate.
//   (2) RGB amplitudes: a small linear least-squares solve (Cholesky) of the
//       normal equations sum_t W_t (L_t - sum_i a_i G_i(d_t))^2.
//
// Directions use the exact same world-space mapping as ProjectEnvMapToSH /
// EquirectangularToCube.hlsl, so the fit lives in the same space in which the
// forward shader evaluates SG lobes (world-space N and R).
static void FitEnvMapToSG(FDemoRoot &Root)
{
	if (!Root.SGLobesMapped)
	{
		return;
	}

	int Width = 0, Height = 0;
	stbi_set_flip_vertically_on_load(1);
	float *HDR = stbi_loadf(GEnvironmentMapPaths[Root.EnvironmentMapIndex], &Width, &Height, nullptr, 3);
	stbi_set_flip_vertically_on_load(0);
	EA_ASSERT(HDR && Width > 0 && Height > 0);

	const double PI_D = 3.14159265358979323846;
	const double GoldenAngle = PI_D * (3.0 - sqrt(5.0));
	const auto ClampD = [](double X, double Lo, double Hi) -> double
	{ return X < Lo ? Lo : (X > Hi ? Hi : X); };

	// Compact coarse sample set (<= 256x128) keeps slider changes interactive
	// even for 4k HDRIs; solid-angle weighting keeps the integral consistent.
	struct FSGSample
	{
		float Dir[3];
		float L[3];
		double W;    // solid angle (amplitude LSQ weight)
		double LumW; // luminance * solid angle (EM weight)
	};
	const int Cols = Width < 256 ? Width : 256;
	const int Rows = Height < 128 ? Height : 128;
	eastl::vector<FSGSample> Samples;
	Samples.reserve((size_t)Cols * Rows);
	for (int iy = 0; iy < Rows; ++iy)
	{
		const double v = (iy + 0.5) / (double)Rows;
		const double e = PI_D * (v - 0.5);
		const double sinE = sin(e), cosE = cos(e);
		const double dOmega = cosE * (2.0 * PI_D / Cols) * (PI_D / Rows);
		for (int ix = 0; ix < Cols; ++ix)
		{
			const double u = (ix + 0.5) / (double)Cols;
			const double phi = 2.0 * PI_D * (u - 0.5);
			const int fx = Width > 1 ? (int)(u * Width) : 0;
			const int fy = Height > 1 ? (int)(v * Height) : 0;
			const float *Texel = HDR + (size_t)(fy * Width + fx) * 3;

			FSGSample S;
			S.Dir[0] = (float)(cosE * cos(phi));
			S.Dir[1] = (float)sinE;
			S.Dir[2] = (float)(cosE * sin(phi));
			S.L[0] = Texel[0];
			S.L[1] = Texel[1];
			S.L[2] = Texel[2];
			S.W = dOmega;
			S.LumW = dOmega * (0.2126 * Texel[0] + 0.7152 * Texel[1] + 0.0722 * Texel[2]);
			Samples.push_back(S);
		}
	}
	stbi_image_free(HDR);

	int N = Root.SGLobeCount;
	if (N < 1) N = 1;
	if (N > SG_MAX_LOBES) N = SG_MAX_LOBES;
	const bool bFixedAxis = (Root.SGAxisMode == SG_FIT_FIXED_AXIS);

	double Mu[SG_MAX_LOBES][3];
	double Lambda[SG_MAX_LOBES];
	double Mix[SG_MAX_LOBES];
	double Amp[SG_MAX_LOBES][3];

	// Initialize lobes on a uniform Fibonacci sphere (used as-is by FIXED mode,
	// used as the starting point for FREE mode), with sharpness ~ lobe count.
	for (int i = 0; i < SG_MAX_LOBES; ++i)
	{
		const double y = 1.0 - 2.0 * (i + 0.5) / (double)N;
		const double r = sqrt(y < -1.0 || y > 1.0 ? 0.0 : 1.0 - y * y);
		const double th = GoldenAngle * i;
		Mu[i][0] = cos(th) * r;
		Mu[i][1] = y;
		Mu[i][2] = sin(th) * r;
		Lambda[i] = ClampD((double)N, 1.0, 64.0);
		Mix[i] = 1.0 / (double)N;
		Amp[i][0] = Amp[i][1] = Amp[i][2] = 0.0;
	}

	// von Mises-Fisher mixture EM on luminance-weighted directions.
	const int NumIters = bFixedAxis ? 8 : 24;
	for (int It = 0; It < NumIters; ++It)
	{
		double AccW[SG_MAX_LOBES] = {};
		double AccVec[SG_MAX_LOBES][3] = {};
		for (const FSGSample &S : Samples)
		{
			double Resp[SG_MAX_LOBES];
			double Sum = 0.0;
			for (int i = 0; i < N; ++i)
			{
				const double Dot = Mu[i][0] * S.Dir[0] + Mu[i][1] * S.Dir[1] + Mu[i][2] * S.Dir[2];
				const double E = Mix[i] * exp(Lambda[i] * (Dot - 1.0));
				Resp[i] = E;
				Sum += E;
			}
			if (Sum < 1e-30)
			{
				continue;
			}
			const double Inv = 1.0 / Sum;
			for (int i = 0; i < N; ++i)
			{
				const double RW = Resp[i] * Inv * S.LumW;
				AccW[i] += RW;
				AccVec[i][0] += RW * S.Dir[0];
				AccVec[i][1] += RW * S.Dir[1];
				AccVec[i][2] += RW * S.Dir[2];
			}
		}

		double TotalW = 0.0;
		for (int i = 0; i < N; ++i) TotalW += AccW[i];
		if (TotalW < 1e-30)
		{
			break;
		}

		for (int i = 0; i < N; ++i)
		{
			Mix[i] = AccW[i] / TotalW > 1e-6 ? AccW[i] / TotalW : 1e-6;
			if (AccW[i] < 1e-12)
			{
				continue;
			}

			double R;
			if (!bFixedAxis)
			{
				const double Len = sqrt(AccVec[i][0] * AccVec[i][0] + AccVec[i][1] * AccVec[i][1] + AccVec[i][2] * AccVec[i][2]);
				if (Len > 1e-12)
				{
					Mu[i][0] = AccVec[i][0] / Len;
					Mu[i][1] = AccVec[i][1] / Len;
					Mu[i][2] = AccVec[i][2] / Len;
				}
				R = Len / AccW[i];
			}
			else
			{
				// Concentration measured about the fixed, uniform axis.
				R = (AccVec[i][0] * Mu[i][0] + AccVec[i][1] * Mu[i][1] + AccVec[i][2] * Mu[i][2]) / AccW[i];
			}
			R = ClampD(R, 0.0, 0.98);
			// Banerjee et al. (2005) vML concentration approximation.
			const double Kappa = R * (3.0 - R * R) / (1.0 - R * R > 1e-4 ? 1.0 - R * R : 1e-4);
			Lambda[i] = ClampD(Kappa, 1.0, 64.0);
		}
	}

	// Linear least squares for RGB amplitudes: G_i(d) = exp(lambda_i*(dot(mu_i,d)-1)).
	double Gram[SG_MAX_LOBES][SG_MAX_LOBES] = {};
	double RHS[SG_MAX_LOBES][3] = {};
	double Gb[SG_MAX_LOBES];
	for (const FSGSample &S : Samples)
	{
		for (int i = 0; i < N; ++i)
		{
			const double Dot = Mu[i][0] * S.Dir[0] + Mu[i][1] * S.Dir[1] + Mu[i][2] * S.Dir[2];
			Gb[i] = exp(Lambda[i] * (Dot - 1.0));
		}
		for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				Gram[i][j] += S.W * Gb[i] * Gb[j];
			}
			for (int c = 0; c < 3; ++c)
			{
				RHS[i][c] += S.W * Gb[i] * S.L[c];
			}
		}
	}
	for (int i = 0; i < N; ++i)
	{
		for (int j = i + 1; j < N; ++j)
		{
			Gram[i][j] = Gram[j][i];
		}
	}
	double DiagSum = 0.0;
	for (int i = 0; i < N; ++i) DiagSum += Gram[i][i];
	const double Ridge = 1e-6 * (DiagSum / N) + 1e-9;
	for (int i = 0; i < N; ++i) Gram[i][i] += Ridge;

	// Cholesky: Gram = L * L^T.
	double Chol[SG_MAX_LOBES][SG_MAX_LOBES] = {};
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			double Sum = Gram[i][j];
			for (int k = 0; k < j; ++k) Sum -= Chol[i][k] * Chol[j][k];
			if (i == j)
			{
				Chol[i][j] = sqrt(Sum > 1e-12 ? Sum : 1e-12);
			}
			else
			{
				Chol[i][j] = Sum / (Chol[j][j] > 1e-12 ? Chol[j][j] : 1e-12);
			}
		}
	}
	for (int c = 0; c < 3; ++c)
	{
		double Y[SG_MAX_LOBES];
		for (int i = 0; i < N; ++i)
		{
			double Sum = RHS[i][c];
			for (int k = 0; k < i; ++k) Sum -= Chol[i][k] * Y[k];
			Y[i] = Sum / Chol[i][i];
		}
		double X[SG_MAX_LOBES];
		for (int i = N - 1; i >= 0; --i)
		{
			double Sum = Y[i];
			for (int k = i + 1; k < N; ++k) Sum -= Chol[k][i] * X[k];
			X[i] = Sum / Chol[i][i];
		}
		for (int i = 0; i < N; ++i) Amp[i][c] = X[i];
	}

	auto *Dst = (float4 *)Root.SGLobesMapped;
	for (int i = 0; i < SG_MAX_LOBES; ++i)
	{
		if (i < N)
		{
			Dst[2 * i] = {(float)Mu[i][0], (float)Mu[i][1], (float)Mu[i][2], (float)Lambda[i]};
			Dst[2 * i + 1] = {(float)Amp[i][0], (float)Amp[i][1], (float)Amp[i][2], 0.0f};
		}
		else
		{
			Dst[2 * i] = {0.0f, 0.0f, 1.0f, 1.0f};
			Dst[2 * i + 1] = {0.0f, 0.0f, 0.0f, 0.0f};
		}
	}
}

// Jacobi eigen-decomposition of a symmetric n x n matrix (Numerical Recipes).
// a[][] is overwritten; eigenvalues returned in d[], eigenvector j is the
// j-th column of v[][] (v[i][j] = component i of eigenvector j).
static void JacobiEigen(double a[16][16], int n, double d[16], double v[16][16])
{
	double b[16], z[16];
	for (int ip = 0; ip < n; ++ip)
	{
		for (int iq = 0; iq < n; ++iq) v[ip][iq] = 0.0;
		v[ip][ip] = 1.0;
	}
	for (int ip = 0; ip < n; ++ip) { b[ip] = d[ip] = a[ip][ip]; z[ip] = 0.0; }
	for (int iter = 1; iter <= 50; ++iter)
	{
		double sm = 0.0;
		for (int ip = 0; ip < n - 1; ++ip)
			for (int iq = ip + 1; iq < n; ++iq) sm += fabs(a[ip][iq]);
		if (sm < 1e-14) break;
		const double tresh = iter < 4 ? 0.2 * sm / (n * n) : 0.0;
		for (int ip = 0; ip < n - 1; ++ip)
		{
			for (int iq = ip + 1; iq < n; ++iq)
			{
				const double g = 100.0 * fabs(a[ip][iq]);
				if (iter > 4 && fabs(d[ip]) + g == fabs(d[ip]) && fabs(d[iq]) + g == fabs(d[iq]))
				{
					a[ip][iq] = 0.0;
				}
				else if (fabs(a[ip][iq]) > tresh)
				{
					const double h = d[iq] - d[ip];
					double t;
					if (fabs(h) + g == fabs(h))
					{
						t = a[ip][iq] / h;
					}
					else
					{
						const double theta = 0.5 * h / a[ip][iq];
						t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
						if (theta < 0.0) t = -t;
					}
					const double c = 1.0 / sqrt(1.0 + t * t);
					const double s = t * c;
					const double tau = s / (1.0 + c);
					const double hh = t * a[ip][iq];
					z[ip] -= hh; z[iq] += hh;
					d[ip] -= hh; d[iq] += hh;
					a[ip][iq] = 0.0;
					for (int j = 0; j < ip; ++j)
					{
						const double g0 = a[j][ip], h0 = a[j][iq];
						a[j][ip] = g0 - s * (h0 + g0 * tau);
						a[j][iq] = h0 + s * (g0 - h0 * tau);
					}
					for (int j = ip + 1; j < iq; ++j)
					{
						const double g0 = a[ip][j], h0 = a[j][iq];
						a[ip][j] = g0 - s * (h0 + g0 * tau);
						a[j][iq] = h0 + s * (g0 - h0 * tau);
					}
					for (int j = iq + 1; j < n; ++j)
					{
						const double g0 = a[ip][j], h0 = a[iq][j];
						a[ip][j] = g0 - s * (h0 + g0 * tau);
						a[iq][j] = h0 + s * (g0 - h0 * tau);
					}
					for (int j = 0; j < n; ++j)
					{
						const double g0 = v[j][ip], h0 = v[j][iq];
						v[j][ip] = g0 - s * (h0 + g0 * tau);
						v[j][iq] = h0 + s * (g0 - h0 * tau);
					}
				}
			}
		}
		for (int ip = 0; ip < n; ++ip) { b[ip] += z[ip]; d[ip] = b[ip]; z[ip] = 0.0; }
	}
}

// Pack the persistently-mapped LUT buffer from the CPU coefficient table.
//  - raw : [g*S + i].rgb = group g, parameter i (specular + autoBias), S = SHE_LUT_STRIDE.
//  - PCA : [0..S-1] mean, [S + k*S + i] basis k, [S+N*S + g*N + k].x weight.
// PCA uses the snapshot method: eigen-decompose the G x G group Gram matrix,
// keep the top-N eigenvectors by eigenvalue, build (3*S)-d basis vectors and
// each group weight vector. The shader reconstructs mean + sum_k w[k]*basis[k].
static void UploadCoeffLUT(FDemoRoot &Root)
{
	FSHECoeffTable &T = Root.CoeffTable;
	if (!T.Valid || !Root.SHECoeffLUTMapped) return;
	float4 *LUT = (float4 *)Root.SHECoeffLUTMapped;
	const int G = T.BakeRoughnessCount;
	const int S = SHE_LUT_STRIDE;
	memset(LUT, 0, sizeof(float4) * GSHECoeffLUTMaxFloat4);

	if (!Root.bUsePCA)
	{
		for (int g = 0; g < G; ++g)
			for (int i = 0; i < S; ++i)
				LUT[g * S + i] = XMFLOAT4(T.Coeff[g][i][0], T.Coeff[g][i][1], T.Coeff[g][i][2], 0.0f);
		Root.ActivePCABasisCount = 0;
		Root.PCAExplainedVariance = 1.0f;
		T.PCABasisCount = 0; // no PCA basis rows / PCA variance to show
		return;
	}

	const int D = S * 3; // 3*S dims per group (specular + autoBias, RGB).
	float x[GSHECoeffLUTMaxGroups][GSHEPCADimMax];
	float mean[GSHEPCADimMax] = {};
	for (int g = 0; g < G; ++g)
		for (int i = 0; i < S; ++i)
			for (int c = 0; c < 3; ++c)
			{
				const int d = i * 3 + c;
				x[g][d] = T.Coeff[g][i][c];
				mean[d] += x[g][d];
			}
	for (int d = 0; d < D; ++d) mean[d] /= (float)G;
	float y[GSHECoeffLUTMaxGroups][GSHEPCADimMax];
	for (int g = 0; g < G; ++g)
		for (int d = 0; d < D; ++d)
			y[g][d] = x[g][d] - mean[d];

	double K[16][16] = {};
	for (int g = 0; g < G; ++g)
		for (int h = 0; h < G; ++h)
		{
			double s = 0.0;
			for (int d = 0; d < D; ++d) s += (double)y[g][d] * y[h][d];
			K[g][h] = s;
		}
	double eigVal[16] = {}, eigVec[16][16] = {};
	JacobiEigen(K, G, eigVal, eigVec);

	double lam[16];
	int cols[16];
	for (int k = 0; k < G; ++k) { lam[k] = eigVal[k] > 0.0 ? eigVal[k] : 0.0; cols[k] = k; }
	for (int a = 1; a < G; ++a) // insertion sort, descending by eigenvalue
	{
		const double kl = lam[a]; const int kc = cols[a];
		int b = a - 1;
		while (b >= 0 && lam[b] < kl) { lam[b + 1] = lam[b]; cols[b + 1] = cols[b]; --b; }
		lam[b + 1] = kl; cols[b + 1] = kc;
	}

	int N = Root.PCABasisCount;
	if (N < 1) N = 1;
	if (N > G) N = G;
	if (N > (int)GSHECoeffLUTMaxBasis) N = GSHECoeffLUTMaxBasis;
	Root.ActivePCABasisCount = N;

	double totalVar = 0.0, keptVar = 0.0;
	for (int k = 0; k < G; ++k) totalVar += lam[k];
	for (int k = 0; k < N; ++k) keptVar += lam[k];
	Root.PCAExplainedVariance = totalVar > 1e-12 ? (float)(keptVar / totalVar) : 1.0f;
	T.PCABasisCount = N;

	for (int i = 0; i < S; ++i) // mean
		LUT[i] = XMFLOAT4(mean[i * 3 + 0], mean[i * 3 + 1], mean[i * 3 + 2], 0.0f);

	// Reconstruct every group from the retained top-N basis (mean + sum w_k*basis_k)
	// and mirror basis vectors into the coefficient table for the ImGui display.
	float recon[GSHECoeffLUTMaxGroups][GSHEPCADimMax];
	for (int g = 0; g < G; ++g)
		for (int d = 0; d < D; ++d)
			recon[g][d] = mean[d];

	const int WeightBase = S + N * S;
	for (int k = 0; k < N; ++k)
	{
		const int col = cols[k];
		const double invSqrt = lam[k] > 1e-12 ? 1.0 / sqrt(lam[k]) : 0.0;
		float basis[GSHEPCADimMax] = {};
		for (int g = 0; g < G; ++g)
		{
			const double coef = eigVec[g][col] * invSqrt;
			for (int d = 0; d < D; ++d) basis[d] += (float)(coef * y[g][d]);
		}
		for (int i = 0; i < S; ++i)
		{
			LUT[S + k * S + i] = XMFLOAT4(basis[i * 3 + 0], basis[i * 3 + 1], basis[i * 3 + 2], 0.0f);
			for (int c = 0; c < 3; ++c)
				T.PCABasisCoeff[k][i][c] = basis[i * 3 + c];
		}
		T.PCABasisExplained[k] = totalVar > 1e-12 ? (float)(lam[k] / totalVar) : 0.0f;
		for (int g = 0; g < G; ++g)
		{
			float w = 0.0f;
			for (int d = 0; d < D; ++d) w += y[g][d] * basis[d];
			LUT[WeightBase + g * N + k] = XMFLOAT4(w, 0.0f, 0.0f, 0.0f);
			for (int d = 0; d < D; ++d)
				recon[g][d] += w * basis[d];
		}
	}

	// Per-band / total variance of each original group vs its PCA reconstruction.
	// Only the specular coefficients (band members) are included; autoBias is
	// excluded. Band indices are derived from the shared P/Q band description.
	for (int g = 0; g < G; ++g)
	{
		float TotalSum = 0.0f;
		int TotalN = 0;
		for (int b = 0; b < GCoeffBandCount; ++b)
		{
			int IdxBuf[GCoeffBandMaxMembers];
			const int Cnt = SHEBandIndicesFill(b, IdxBuf);
			float BandSum = 0.0f;
			int BandN = 0;
			for (int k = 0; k < Cnt; ++k)
			{
				const int Idx = IdxBuf[k];
				for (int c = 0; c < 3; ++c)
				{
					const float diff = T.Coeff[g][Idx][c] - recon[g][Idx * 3 + c];
					BandSum += diff * diff;
					++BandN;
				}
			}
			T.PCABandVariance[g][b] = BandN > 0 ? BandSum / BandN : 0.0f;
			TotalSum += BandSum;
			TotalN += BandN;
		}
		T.PCATotalVariance[g] = TotalN > 0 ? TotalSum / TotalN : 0.0f;
	}
}

// Read back the baked SHE matrices and solve a separate fit for the
// SHE_SPEC_COEFFS specular slots for each roughness group, plus one
// ungrouped (global) fit used as the baseline.
// Results fill Root.CoeffTable and are uploaded into the LUT buffer.
// Height row layout (SHE_Build.hlsl): height = NormalIndex + RoughnessIndex*64,
// so group g owns rows [g*64, g*64+63].
static void ComputeGroupedSHEFit(FDemoRoot &Root)
{
	FGraphicsContext &Gfx = Root.Gfx;
	const uint32_t RoughnessCount = (uint32_t)(Root.BakeRoughnessCount > 0 ? Root.BakeRoughnessCount : 1);
	const uint32_t TotalHeight = GSHENormalCount * RoughnessCount;

	FSHECoeffTable &Table = Root.CoeffTable;
	memset(&Table, 0, sizeof(Table));
	Table.BakeRoughnessCount = (int)RoughnessCount;
	Table.POrder = Root.SHEPOrder;
	Table.QOrder = Root.SHEQOrder;
	Table.RoughnessStart = Root.BakeRoughnessStart;
	Table.RoughnessEnd = Root.BakeRoughnessEnd;

	// Footprints for copying the two textures into CPU-visible readback buffers.
	const D3D12_RESOURCE_DESC ADesc = Root.SHEMatrixA->GetDesc();
	const D3D12_RESOURCE_DESC bDesc = Root.SHEMatrixb->GetDesc();
	D3D12_PLACED_SUBRESOURCE_FOOTPRINT AFoot = {}, bFoot = {};
	UINT64 ASize = 0, bSize = 0;
	Gfx.Device->GetCopyableFootprints(&ADesc, 0, 1, 0, &AFoot, nullptr, nullptr, &ASize);
	Gfx.Device->GetCopyableFootprints(&bDesc, 0, 1, 0, &bFoot, nullptr, nullptr, &bSize);

	ID3D12Resource *ReadA = nullptr, *Readb = nullptr;
	VHR(Gfx.Device->CreateCommittedResource(
		get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK)), D3D12_HEAP_FLAG_NONE,
		get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(ASize)), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&ReadA)));
	VHR(Gfx.Device->CreateCommittedResource(
		get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK)), D3D12_HEAP_FLAG_NONE,
		get_rvalue_ptr(CD3DX12_RESOURCE_DESC::Buffer(bSize)), D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&Readb)));

	ID3D12GraphicsCommandList2 *CmdList = GetAndInitCommandList(Gfx);
	{
		const CD3DX12_RESOURCE_BARRIER ToCopy[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(Root.SHEMatrixA, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(Root.SHEMatrixb, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
		};
		CmdList->ResourceBarrier(2, ToCopy);

		const CD3DX12_TEXTURE_COPY_LOCATION SrcA(Root.SHEMatrixA, 0);
		const CD3DX12_TEXTURE_COPY_LOCATION Srcb(Root.SHEMatrixb, 0);
		const CD3DX12_TEXTURE_COPY_LOCATION DstA(ReadA, AFoot);
		const CD3DX12_TEXTURE_COPY_LOCATION Dstb(Readb, bFoot);
		CmdList->CopyTextureRegion(&DstA, 0, 0, 0, &SrcA, nullptr);
		CmdList->CopyTextureRegion(&Dstb, 0, 0, 0, &Srcb, nullptr);

		const CD3DX12_RESOURCE_BARRIER ToSRV[2] = {
			CD3DX12_RESOURCE_BARRIER::Transition(Root.SHEMatrixA, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
			CD3DX12_RESOURCE_BARRIER::Transition(Root.SHEMatrixb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
		};
		CmdList->ResourceBarrier(2, ToSRV);
	}
	CmdList->Close();
	Gfx.CmdQueue->ExecuteCommandLists(1, CommandListCast(&Gfx.CmdList));
	WaitForGPU(Gfx);

	void *MappedA = nullptr, *Mappedb = nullptr;
	ReadA->Map(0, nullptr, &MappedA);
	Readb->Map(0, nullptr, &Mappedb);
	const uint8_t *ABase = (const uint8_t *)MappedA + AFoot.Offset;
	const uint8_t *bBase = (const uint8_t *)Mappedb + bFoot.Offset;
	const UINT ARowPitch = AFoot.Footprint.RowPitch;
	const UINT bRowPitch = bFoot.Footprint.RowPitch;

	// Global (ungrouped) normal-equation accumulators, summed across every row.
	// All SHE_SPEC_COEFFS slots participate; unused P/Q slots have zero A
	// columns and (with ridge) solve to exactly 0.
	float GAtA[SHE_SPEC_COEFFS][SHE_SPEC_COEFFS] = {};
	float GAtb[SHE_SPEC_COEFFS][3] = {};

	for (uint32_t g = 0; g < RoughnessCount; ++g)
	{
		float AtA[SHE_SPEC_COEFFS][SHE_SPEC_COEFFS] = {};
		float Atb[SHE_SPEC_COEFFS][3] = {};
		for (uint32_t h = g * GSHENormalCount; h < (g + 1) * GSHENormalCount; ++h)
		{
			const float *ARow = (const float *)(ABase + (size_t)h * ARowPitch);
			const uint16_t *bRow = (const uint16_t *)(bBase + (size_t)h * bRowPitch);
			for (uint32_t v = 0; v < GSHEViewCount; ++v)
			{
				const float *a = ARow + v * GSHESphericalHarmonicCount;
				float blog[3];
				for (int c = 0; c < 3; ++c)
				{
					blog[c] = HalfToFloat(bRow[v * 4 + c]);
				}
				for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
				{
					for (int j = 0; j <= i; ++j)
					{
						const float p = a[i] * a[j];
						AtA[i][j] += p;
						GAtA[i][j] += p;
					}
					for (int c = 0; c < 3; ++c)
					{
						const float q = a[i] * blog[c];
						Atb[i][c] += q;
						GAtb[i][c] += q;
					}
				}
			}
		}
		for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
		{
			AtA[i][i] += 1e-2f; // ridge
		}
		float X[SHE_SPEC_COEFFS][3] = {};
		SolveGroupedFit(AtA, Atb, X, SHE_SPEC_COEFFS);

		// Per-group log-space energy calibration (autoBias), mirroring
		// SHE_Calibrate.hlsl but restricted to this group's rows.
		float Target[3] = {}, Recon[3] = {};
		for (uint32_t h = g * GSHENormalCount; h < (g + 1) * GSHENormalCount; ++h)
		{
			const float *ARow = (const float *)(ABase + (size_t)h * ARowPitch);
			const uint16_t *bRow = (const uint16_t *)(bBase + (size_t)h * bRowPitch);
			for (uint32_t v = 0; v < GSHEViewCount; ++v)
			{
				const float *a = ARow + v * GSHESphericalHarmonicCount;
				for (int c = 0; c < 3; ++c)
				{
					float LogRecon = 0.0f;
					for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
					{
						LogRecon += a[i] * X[i][c];
					}
					Target[c] += expf(HalfToFloat(bRow[v * 4 + c]));
					Recon[c] += expf(fmaxf(LogRecon, -20.0f));
				}
			}
		}
		for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
		{
			for (int c = 0; c < 3; ++c)
			{
				Table.Coeff[g][i][c] = X[i][c];
			}
		}
		for (int c = 0; c < 3; ++c)
		{
			Table.Coeff[g][SHE_AUTOBIAS_SLOT][c] = logf(fmaxf(Target[c], 1e-6f) / fmaxf(Recon[c], 1e-6f));
		}
	}

	// Ungrouped baseline fit (last table row).
	for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
	{
		GAtA[i][i] += 1e-2f;
	}
	float GX[SHE_SPEC_COEFFS][3] = {};
	SolveGroupedFit(GAtA, GAtb, GX, SHE_SPEC_COEFFS);
	{
		float Target[3] = {}, Recon[3] = {};
		for (uint32_t h = 0; h < TotalHeight; ++h)
		{
			const float *ARow = (const float *)(ABase + (size_t)h * ARowPitch);
			const uint16_t *bRow = (const uint16_t *)(bBase + (size_t)h * bRowPitch);
			for (uint32_t v = 0; v < GSHEViewCount; ++v)
			{
				const float *a = ARow + v * GSHESphericalHarmonicCount;
				for (int c = 0; c < 3; ++c)
				{
					float LogRecon = 0.0f;
					for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
					{
						LogRecon += a[i] * GX[i][c];
					}
					Target[c] += expf(HalfToFloat(bRow[v * 4 + c]));
					Recon[c] += expf(fmaxf(LogRecon, -20.0f));
				}
			}
		}
		for (int i = 0; i < SHE_SPEC_COEFFS; ++i)
		{
			for (int c = 0; c < 3; ++c)
			{
				Table.Coeff[RoughnessCount][i][c] = GX[i][c];
			}
		}
		for (int c = 0; c < 3; ++c)
		{
			Table.Coeff[RoughnessCount][SHE_AUTOBIAS_SLOT][c] = logf(fmaxf(Target[c], 1e-6f) / fmaxf(Recon[c], 1e-6f));
		}
	}

	ReadA->Unmap(0, nullptr);
	Readb->Unmap(0, nullptr);
	SAFE_RELEASE(ReadA);
	SAFE_RELEASE(Readb);

	// Mean squared deviation of each grouped row vs the ungrouped baseline,
	// per band (over RGB) and over all specular coefficients.
	for (uint32_t g = 0; g < RoughnessCount; ++g)
	{
		float TotalSum = 0.0f;
		int TotalN = 0;
		for (int b = 0; b < GCoeffBandCount; ++b)
		{
			int IdxBuf[GCoeffBandMaxMembers];
			const int Cnt = SHEBandIndicesFill(b, IdxBuf);
			float BandSum = 0.0f;
			int BandN = 0;
			for (int k = 0; k < Cnt; ++k)
			{
				const int Idx = IdxBuf[k];
				for (int c = 0; c < 3; ++c)
				{
					const float d = Table.Coeff[g][Idx][c] - Table.Coeff[RoughnessCount][Idx][c];
					BandSum += d * d;
					++BandN;
				}
			}
			Table.BandVariance[g][b] = BandN > 0 ? BandSum / BandN : 0.0f;
			TotalSum += BandSum;
			TotalN += BandN;
		}
		Table.TotalVariance[g] = TotalN > 0 ? TotalSum / TotalN : 0.0f;
	}

	Table.Valid = 1;

	// Pack the LUT buffer (raw coefficients or PCA mean/basis/weights,
	// depending on the current panel state).
	UploadCoeffLUT(Root);

	Root.NumFrames = 0;
}

// ImGui table: one row per roughness group (+ ungrouped baseline), SHE
// specular coefficients grouped by SH band, with per-band and total variance
// vs the ungrouped fit. Uses ImGui 1.72 legacy Columns API.
// ImGui table: one row per roughness group (+ ungrouped baseline + PCA basis),
// specular coefficients grouped by SH band (2*l + 1 per band), with per-band and
// total variance columns. Uses ImGui 1.72 legacy Columns API.
static void DrawSHECoeffTable(FDemoRoot &Root)
{
	ImGui::SetNextWindowSize(ImVec2(1180.0f, 460.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowContentWidth(1480.0f);
	if (!ImGui::Begin("SHE Grouped-Fit Coefficients", &Root.bShowCoeffTable, ImGuiWindowFlags_HorizontalScrollbar))
	{
		ImGui::End();
		return;
	}
	const FSHECoeffTable &T = Root.CoeffTable;
	if (!T.Valid)
	{
		ImGui::TextDisabled("No grouped fit available yet (waiting for first bake).");
		ImGui::End();
		return;
	}
	const int N = T.BakeRoughnessCount;
	const bool bPCA = Root.bUsePCA && T.PCABasisCount > 0;
	ImGui::Text("Groups: %d   Bake range: [%.2f, %.2f]   P=%d  Q=%d", N, T.RoughnessStart, T.RoughnessEnd, T.POrder, T.QOrder);
	if (Root.bUsePCA)
	{
		ImGui::Text("PCA: %d / %d basis vectors, explained variance %.1f%%",
			Root.ActivePCABasisCount, Root.PCABasisCount, Root.PCAExplainedVariance * 100.0f);
	}
	ImGui::TextDisabled("Cells: %d specular slots (luma) per SH band (2l+1). vU= MSE vs Ungrouped, vP= MSE vs PCA recon.", SHE_SPEC_COEFFS);

	const int NumCols = 1 + GCoeffBandCount + 1;
	ImGui::Columns(NumCols, "shecoefftable", true);
	ImGui::SetColumnWidth(0, 110.0f);
	// Each band column width scales with its (2l+1) parameter count so wider
	// bands get proportionally more room (TextWrapped still avoids clipping).
	for (int b = 0; b < GCoeffBandCount; ++b)
	{
		ImGui::SetColumnWidth(1 + b, 30.0f + 28.0f * (float)GCoeffBandCoeffCount[b]);
	}
	ImGui::SetColumnWidth(1 + GCoeffBandCount, 96.0f);

	ImGui::TextDisabled("Group");
	ImGui::NextColumn();
	for (int b = 0; b < GCoeffBandCount; ++b)
	{
		ImGui::TextDisabled("%s (%d)", GCoeffBandLabels[b], GCoeffBandCoeffCount[b]);
		ImGui::NextColumn();
	}
	ImGui::TextDisabled("TotalVar");
	ImGui::NextColumn();
	ImGui::Separator();

	char Buf[512];
	// Emit the GCoeffBandCount coefficient cells for one row. Optional variance
	// caption lines (vU = vs ungrouped, vP = vs PCA reconstruction) are appended
	// under each cell; the total column shows the corresponding total variance(s).
	auto EmitRow = [&](const float Coeff[GSHECoeffPerGroup][4],
		const float *VarU, float TotU, bool bShowU,
		const float *VarP, float TotP, bool bShowP,
		const char *NoVarCaption)
	{
		for (int b = 0; b < GCoeffBandCount; ++b)
		{
			int IdxBuf[GCoeffBandMaxMembers];
			const int Cnt = SHEBandIndicesFill(b, IdxBuf);
			char *p = Buf;
			size_t Rem = sizeof(Buf);
			for (int k = 0; k < Cnt; ++k)
			{
				const int Idx = IdxBuf[k];
				const float Luma = (Coeff[Idx][0] + Coeff[Idx][1] + Coeff[Idx][2]) / 3.0f;
				const int W = EA::StdC::Snprintf(p, Rem, "%.2f ", Luma);
				if (W < 0 || (size_t)W >= Rem) break;
				p += W;
				Rem -= W;
			}
			ImGui::TextWrapped("%s", Buf);
			if (bShowU && VarU) ImGui::TextDisabled("vU=%.3g", VarU[b]);
			if (bShowP && VarP) ImGui::TextDisabled("vP=%.3g", VarP[b]);
			if (!bShowU && !bShowP && NoVarCaption) ImGui::TextDisabled("%s", NoVarCaption);
			ImGui::NextColumn();
		}
		if (bShowU) ImGui::Text("U=%.3g", TotU);
		if (bShowP) ImGui::Text("P=%.3g", TotP);
		if (!bShowU && !bShowP) ImGui::TextDisabled("-");
		ImGui::NextColumn();
	};

	// ---- Group rows: coefficients + vs-ungrouped variance + (optional) PCA variance
	for (int g = 0; g < N; ++g)
	{
		const float Denom = (float)(N > 1 ? N - 1 : 1);
		const float R = T.RoughnessStart + (float)g / Denom * (T.RoughnessEnd - T.RoughnessStart);
		ImGui::Text("R=%.2f", R);
		ImGui::NextColumn();
		EmitRow(T.Coeff[g],
			T.BandVariance[g], T.TotalVariance[g], true,
			bPCA ? T.PCABandVariance[g] : nullptr, bPCA ? T.PCATotalVariance[g] : 0.0f, bPCA,
			nullptr);
	}

	// ---- Ungrouped baseline ----
	ImGui::Text("Ungrouped");
	ImGui::NextColumn();
	EmitRow(T.Coeff[N], nullptr, 0.0f, false, nullptr, 0.0f, false, "baseline");

	// ---- PCA basis rows (directly below Ungrouped), same band grid ----
	if (bPCA)
	{
		ImGui::Separator();
		for (int k = 0; k < T.PCABasisCount; ++k)
		{
			ImGui::Text("PCA B%d (%.1f%%)", k, T.PCABasisExplained[k] * 100.0f);
			ImGui::NextColumn();
			EmitRow(T.PCABasisCoeff[k], nullptr, 0.0f, false, nullptr, 0.0f, false, nullptr);
		}
	}

	ImGui::Columns(1);
	ImGui::End();
}

static void ReleaseEnvironmentResources(FDemoRoot &Root)
{
	SAFE_RELEASE(Root.EnvMap);
	SAFE_RELEASE(Root.IrradianceMap);
	SAFE_RELEASE(Root.PrefilteredEnvMap);
	SAFE_RELEASE(Root.SHEMatrixA);
	SAFE_RELEASE(Root.SHEMatrixb);
	SAFE_RELEASE(Root.SHEMatrixAT);
	SAFE_RELEASE(Root.SHESHCoeff);
}

static void RecordEnvironmentPrecomputation(FDemoRoot &Root, eastl::vector<ID3D12Resource *> &TempResources, eastl::vector<ID3D12Resource *> &TexturesThatNeedMipmaps)
{
	FGraphicsContext &Gfx = Root.Gfx;

	Gfx.CmdList->IASetVertexBuffers(0, 1, &Root.StaticVBView);
	Gfx.CmdList->IASetIndexBuffer(&Root.StaticIBView);

	// Create EnvMap.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_EquirectangularToCube]);
	Gfx.CmdList->SetGraphicsRootSignature(Root.RootSignatures[PSO_EquirectangularToCube]);
	CreateEnvMap(Root.Gfx, Root.StaticMeshes[MESH_Cube], GEnvironmentMapPaths[Root.EnvironmentMapIndex], Root.EnvMap, Root.EnvMapSRV, TempResources);
	TexturesThatNeedMipmaps.push_back(Root.EnvMap);

	// Create IrradianceMap.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_GenerateIrradianceMap]);
	Gfx.CmdList->SetGraphicsRootSignature(Root.RootSignatures[PSO_GenerateIrradianceMap]);
	CreateIrradianceMap(Root.Gfx, Root.EnvMapSRV, Root.StaticMeshes[MESH_Cube], Root.IrradianceMap, Root.IrradianceMapSRV, TempResources);

	// The SHE passes sample the environment map from compute shaders. Allow both
	// pixel and non-pixel SRV reads for the duration of the SHE section; restore the
	// pixel-only state afterwards because the mipmap generator requires it exactly.
	Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
		Root.EnvMap, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)));

	// SHE Build.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Build]);
	Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Build]);
	SHEBuild(Root, Gfx, Root.SHEMatrixA, Root.SHEMatrixb, Root.SHEMatrixASRV, Root.SHEMatrixbSRV, Root.EnvMapSRV);

	// SHE Reduction.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Reduction]);
	Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Reduction]);
	SHEReduction(Root, Gfx, Root.SHEMatrixAT, Root.SHEMatrixATSRV, Root.SHEMatrixASRV, Root.SHEMatrixbSRV, TempResources);

	// SHE Solve.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Solve]);
	Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Solve]);
	SHESolve(Gfx, Root.SHESHCoeff, Root.SHESHCoeffCBV, Root.SHEMatrixATSRV);

	// SHE log-space brightness calibration.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Calibrate]);
	Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Calibrate]);
	SHECalibrate(Root, Gfx, Root.SHESHCoeff, Root.SHEMatrixASRV, Root.SHEMatrixbSRV);

	// 3-band SH diffuse irradiance from the environment map (two-stage
	// dispatch; PSOs are set inside SHEDiffuse per stage).
	SHEDiffuse(Root, Gfx, Root.SHESHCoeff, Root.EnvMapSRV, TempResources);

	Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
		Root.EnvMap, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
}

static void FinishEnvironmentPrecomputation(FDemoRoot &Root, eastl::vector<ID3D12Resource *> &TempResources, eastl::vector<ID3D12Resource *> &TexturesThatNeedMipmaps)
{
	FGraphicsContext &Gfx = Root.Gfx;

	const DXGI_FORMAT Formats[] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM};
	FMipmapGenerator MipmapGenerators[eastl::size(Formats)];
	for (uint32_t Idx = 0; Idx < eastl::size(Formats); ++Idx)
	{
		CreateMipmapGenerator(Gfx, Formats[Idx], MipmapGenerators[Idx]);
	}

	for (ID3D12Resource *Texture : TexturesThatNeedMipmaps)
	{
		const D3D12_RESOURCE_DESC Desc = Texture->GetDesc();

		for (uint32_t Idx = 0; Idx < eastl::size(Formats); ++Idx)
		{
			if (Desc.Format == Formats[Idx])
			{
				GenerateMipmaps(Gfx, MipmapGenerators[Idx], Texture);
				break;
			}
		}
	}

	// Create PrefilteredEnvMap.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_PrefilterEnvMap]);
	Gfx.CmdList->SetGraphicsRootSignature(Root.RootSignatures[PSO_PrefilterEnvMap]);
	CreatePrefilteredEnvMap(Gfx, Root.EnvMapSRV, Root.StaticMeshes[MESH_Cube], Root.PrefilteredEnvMap, Root.PrefilteredEnvMapSRV, TempResources);

	Gfx.CmdList->Close();
	Gfx.CmdQueue->ExecuteCommandLists(1, CommandListCast(&Gfx.CmdList));
	WaitForGPU(Gfx);

	for (ID3D12Resource *Resource : TempResources)
	{
		SAFE_RELEASE(Resource);
	}
	for (uint32_t Idx = 0; Idx < eastl::size(MipmapGenerators); ++Idx)
	{
		DestroyMipmapGenerator(MipmapGenerators[Idx]);
	}

	// Build the grouped-fit LUT and coefficient table from the freshly baked
	// SHE matrices (GPU is idle here, so the readback is safe).
	ComputeGroupedSHEFit(Root);

	// Re-project the (possibly new) HDR environment onto classic SH coefficients.
	// This is a CPU-side integral over the equirectangular HDR, independent of
	// the GPU bake above, so run it here where all environment state is settled.
	ProjectEnvMapToSH(Root);

	// Re-fit the SG lobes for the (possibly new) HDR with the current settings.
	Root.SGDirty = 0;
	FitEnvMapToSG(Root);
}

static void RebuildEnvironment(FDemoRoot &Root)
{
	if (Root.PendingEnvironmentMapIndex < 0)
	{
		return;
	}

	WaitForGPU(Root.Gfx);
	ReleaseEnvironmentResources(Root);

	Root.EnvironmentMapIndex = Root.PendingEnvironmentMapIndex;
	Root.SelectedEnvironmentMapIndex = Root.EnvironmentMapIndex;
	Root.PendingEnvironmentMapIndex = -1;
	Root.NumFrames = 0;

	GetAndInitCommandList(Root.Gfx);

	eastl::vector<ID3D12Resource *> TempResources;
	eastl::vector<ID3D12Resource *> TexturesThatNeedMipmaps;
	RecordEnvironmentPrecomputation(Root, TempResources, TexturesThatNeedMipmaps);
	FinishEnvironmentPrecomputation(Root, TempResources, TexturesThatNeedMipmaps);
}

static void ReleaseSHEResources(FDemoRoot &Root)
{
	SAFE_RELEASE(Root.SHEMatrixA);
	SAFE_RELEASE(Root.SHEMatrixb);
	SAFE_RELEASE(Root.SHEMatrixAT);
	SAFE_RELEASE(Root.SHESHCoeff);
}

// Re-run only the SHE precomputation (build, reduction, solve, calibrate,
// diffuse irradiance) with the current bake roughness range and level count.
// The environment map and split-sum data stay untouched.
static void RebuildSHEBake(FDemoRoot &Root)
{
	WaitForGPU(Root.Gfx);
	ReleaseSHEResources(Root);
	Root.NumFrames = 0;

	GetAndInitCommandList(Root.Gfx);

	// The SHE passes share the graphics command list with static geometry
	// bindings; refresh them the same way the initial environment build does.
	Root.Gfx.CmdList->IASetVertexBuffers(0, 1, &Root.StaticVBView);
	Root.Gfx.CmdList->IASetIndexBuffer(&Root.StaticIBView);

	eastl::vector<ID3D12Resource *> TempResources;

	// The SHE passes sample the environment map from compute shaders. Allow both
	// pixel and non-pixel SRV reads for the rebake and restore the pixel-only state
	// below so repeated rebakes always start from a known state.
	Root.Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
		Root.EnvMap, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)));

	// SHE Build.
	Root.Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Build]);
	Root.Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Build]);
	SHEBuild(Root, Root.Gfx, Root.SHEMatrixA, Root.SHEMatrixb, Root.SHEMatrixASRV, Root.SHEMatrixbSRV, Root.EnvMapSRV);

	// SHE Reduction.
	Root.Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Reduction]);
	Root.Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Reduction]);
	SHEReduction(Root, Root.Gfx, Root.SHEMatrixAT, Root.SHEMatrixATSRV, Root.SHEMatrixASRV, Root.SHEMatrixbSRV, TempResources);

	// SHE Solve.
	Root.Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Solve]);
	Root.Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Solve]);
	SHESolve(Root.Gfx, Root.SHESHCoeff, Root.SHESHCoeffCBV, Root.SHEMatrixATSRV);

	// SHE log-space brightness calibration.
	Root.Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Calibrate]);
	Root.Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Calibrate]);
	SHECalibrate(Root, Root.Gfx, Root.SHESHCoeff, Root.SHEMatrixASRV, Root.SHEMatrixbSRV);

	// 3-band SH diffuse irradiance from the environment map.
	SHEDiffuse(Root, Root.Gfx, Root.SHESHCoeff, Root.EnvMapSRV, TempResources);

	Root.Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
		Root.EnvMap, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));

	// Close, execute and release temp resources (no mipmap work needed here).
	Root.Gfx.CmdList->Close();
	Root.Gfx.CmdQueue->ExecuteCommandLists(1, CommandListCast(&Root.Gfx.CmdList));
	WaitForGPU(Root.Gfx);

	for (ID3D12Resource *Resource : TempResources)
	{
		SAFE_RELEASE(Resource);
	}

	// Refresh the grouped-fit LUT / table for the new bake parameters.
	ComputeGroupedSHEFit(Root);
}

static void LoadGLTFMesh(const char *FileName, FMesh &OutMesh, eastl::vector<FVertex> &InOutVertices, eastl::vector<uint32_t> &InOutIndices)
{
	cgltf_options Options = {};
	cgltf_data *Data = nullptr;
	{
		cgltf_result R = cgltf_parse_file(&Options, FileName, &Data);
		EA_ASSERT(R == cgltf_result_success);
		R = cgltf_load_buffers(&Options, Data, FileName);
		EA_ASSERT(R == cgltf_result_success);
	}

	cgltf_mesh *Mesh = &Data->meshes[0];
	EA_ASSERT(Mesh->primitives_count <= MESH_MAX_NUM_SECTIONS);

	OutMesh.NumSections = (uint32_t)Mesh->primitives_count;

	uint32_t TotalNumVertices = 0;
	uint32_t TotalNumIndices = 0;

	for (uint32_t SectionIdx = 0; SectionIdx < Mesh->primitives_count; ++SectionIdx)
	{
		EA_ASSERT(Data->meshes[0].primitives[SectionIdx].indices);
		EA_ASSERT(Data->meshes[0].primitives[SectionIdx].attributes);

		TotalNumIndices += (uint32_t)Data->meshes[0].primitives[SectionIdx].indices->count;
		TotalNumVertices += (uint32_t)Data->meshes[0].primitives[SectionIdx].attributes[0].data->count;
	}

	InOutVertices.reserve(InOutVertices.size() + TotalNumVertices);
	InOutIndices.reserve(InOutIndices.size() + TotalNumIndices);

	eastl::vector<XMFLOAT3> Positions;
	eastl::vector<XMFLOAT3> Normals;
	Positions.reserve(TotalNumVertices);
	Normals.reserve(TotalNumVertices);

	for (uint32_t SectionIdx = 0; SectionIdx < Mesh->primitives_count; ++SectionIdx)
	{
		// Indices.
		{
			const cgltf_accessor *Accessor = Data->meshes[0].primitives[SectionIdx].indices;

			EA_ASSERT(Accessor->buffer_view);
			EA_ASSERT(Accessor->stride == Accessor->buffer_view->stride || Accessor->buffer_view->stride == 0);
			EA_ASSERT((Accessor->stride * Accessor->count) == Accessor->buffer_view->size);

			const auto DataAddr = (const uint8_t *)Accessor->buffer_view->buffer->data + Accessor->offset + Accessor->buffer_view->offset;

			OutMesh.Sections[SectionIdx].StartIndexLocation = (uint32_t)InOutIndices.size();
			OutMesh.Sections[SectionIdx].IndexCount = (uint32_t)Accessor->count;

			if (Accessor->stride == 1)
			{
				const uint8_t *DataU8 = (const uint8_t *)DataAddr;
				for (uint32_t Idx = 0; Idx < Accessor->count; ++Idx)
				{
					InOutIndices.push_back((uint32_t)*DataU8++);
				}
			}
			else if (Accessor->stride == 2)
			{
				const uint16_t *DataU16 = (const uint16_t *)DataAddr;
				for (uint32_t Idx = 0; Idx < Accessor->count; ++Idx)
				{
					InOutIndices.push_back((uint32_t)*DataU16++);
				}
			}
			else if (Accessor->stride == 4)
			{
				InOutIndices.resize(InOutIndices.size() + Accessor->count);
				memcpy(&InOutIndices[InOutIndices.size() - Accessor->count], DataAddr, Accessor->count * Accessor->stride);
			}
			else
			{
				EA_ASSERT(0);
			}
		}

		// Attributes.
		{
			const uint32_t NumAttribs = (uint32_t)Data->meshes[0].primitives[SectionIdx].attributes_count;

			for (uint32_t AttribIdx = 0; AttribIdx < NumAttribs; ++AttribIdx)
			{
				const cgltf_attribute *Attrib = &Data->meshes[0].primitives[SectionIdx].attributes[AttribIdx];
				const cgltf_accessor *Accessor = Attrib->data;

				EA_ASSERT(Accessor->buffer_view);
				EA_ASSERT(Accessor->stride == Accessor->buffer_view->stride || Accessor->buffer_view->stride == 0);
				EA_ASSERT((Accessor->stride * Accessor->count) == Accessor->buffer_view->size);

				const auto DataAddr = (const uint8_t *)Accessor->buffer_view->buffer->data + Accessor->offset + Accessor->buffer_view->offset;

				if (Attrib->type == cgltf_attribute_type_position)
				{
					EA_ASSERT(Accessor->type == cgltf_type_vec3);
					Positions.resize(Accessor->count);
					memcpy(Positions.data(), DataAddr, Accessor->count * Accessor->stride);
				}
				else if (Attrib->type == cgltf_attribute_type_normal)
				{
					EA_ASSERT(Accessor->type == cgltf_type_vec3);
					Normals.resize(Accessor->count);
					memcpy(Normals.data(), DataAddr, Accessor->count * Accessor->stride);
				}
			}

			EA_ASSERT(Positions.size() > 0 && Positions.size() == Normals.size());

			OutMesh.Sections[SectionIdx].BaseVertexLocation = (uint32_t)InOutVertices.size();

			for (uint32_t Idx = 0; Idx < Positions.size(); ++Idx)
			{
				FVertex Vertex;
				Vertex.Position = Positions[Idx];
				Vertex.Normal = Normals[Idx];
				InOutVertices.push_back(Vertex);
			}

			Positions.clear();
			Normals.clear();
		}
	}

	cgltf_free(Data);
}

static void OnWindowResize(FDemoRoot &Root, uint32_t Width, uint32_t Height)
{
	FGraphicsContext &Gfx = Root.Gfx;

	// Ignore no-op resizes (e.g. the initial WM_SIZE sent at window creation).
	if (Gfx.Resolution[0] == Width && Gfx.Resolution[1] == Height)
	{
		return;
	}

	// Rebuild the swap chain and its render-target views for the new size.
	ResizeSwapChain(Gfx, Width, Height);

	// Recreate all size-dependent render resources (MSAA color, MSAA depth,
	// accumulation). Their descriptor handles are stable, so only the resources
	// and views need to be refreshed in place.
	SAFE_RELEASE(Root.MSColorBuffer);
	SAFE_RELEASE(Root.MSDepthBuffer);
	SAFE_RELEASE(Root.AccumulationBuffer);

	{
		CD3DX12_RESOURCE_DESC DescColor = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, Gfx.Resolution[0], Gfx.Resolution[1], 1, 1, Root.NumSamples);
		DescColor.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &DescColor, D3D12_RESOURCE_STATE_RENDER_TARGET, get_rvalue_ptr(CD3DX12_CLEAR_VALUE(DXGI_FORMAT_R8G8B8A8_UNORM, XMVECTORF32{0.0f})), IID_PPV_ARGS(&Root.MSColorBuffer)));

		CD3DX12_RESOURCE_DESC DescDepth = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, Gfx.Resolution[0], Gfx.Resolution[1], 1, 1, Root.NumSamples);
		DescDepth.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &DescDepth, D3D12_RESOURCE_STATE_DEPTH_WRITE, get_rvalue_ptr(CD3DX12_CLEAR_VALUE(DXGI_FORMAT_D32_FLOAT, 1.0f, 0)), IID_PPV_ARGS(&Root.MSDepthBuffer)));

		Gfx.Device->CreateRenderTargetView(Root.MSColorBuffer, nullptr, Root.MSColorBufferRTV);
		Gfx.Device->CreateDepthStencilView(Root.MSDepthBuffer, nullptr, Root.MSDepthBufferDSV);
	}

	{
		CD3DX12_RESOURCE_DESC DescAccum = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, Gfx.Resolution[0], Gfx.Resolution[1],
																	   1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
												&DescAccum, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&Root.AccumulationBuffer)));

		D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
		SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		SRVDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		SRVDesc.Texture2D.MipLevels = 1;
		SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
		Gfx.Device->CreateShaderResourceView(Root.AccumulationBuffer, &SRVDesc, Root.AccumulationBufferSRV);
	}

	// Keep ImGui's coordinate space and the progressive accumulation in sync.
	ImGuiIO &IO = ImGui::GetIO();
	IO.DisplaySize = ImVec2((float)Gfx.Resolution[0], (float)Gfx.Resolution[1]);
	Root.NumFrames = 0;
}

static void Initialize(FDemoRoot &Root)
{
	FGraphicsContext &Gfx = Root.Gfx;

	eastl::vector<ID3D12Resource *> TempResources;
	eastl::vector<ID3D12Resource *> TexturesThatNeedMipmaps;

	Root.NumSamples = 1;
	CreateUIContext(Gfx, Root.NumSamples, Root.UI, TempResources);
	CreatePipelines(Gfx, Root.NumSamples, Root.Pipelines, Root.RootSignatures, Root.WaveSize);
	CreateSHECoeffLUTResources(Root);
	CreateClassicSHCoeffResources(Root);
	CreateSGLobeResources(Root);

	eastl::vector<FVertex> AllVertices;
	eastl::vector<uint32_t> AllIndices;
	{
		{
			FMesh Mesh = {};
			LoadGLTFMesh("Data/Meshes/Cube.gltf", Mesh, AllVertices, AllIndices);
			Root.StaticMeshes.push_back(FStaticMesh{Mesh.Sections[0].IndexCount, Mesh.Sections[0].StartIndexLocation, Mesh.Sections[0].BaseVertexLocation});
		}
		{
			FMesh Mesh = {};
			LoadGLTFMesh("Data/Meshes/Sphere.gltf", Mesh, AllVertices, AllIndices);
			Root.StaticMeshes.push_back(FStaticMesh{Mesh.Sections[0].IndexCount, Mesh.Sections[0].StartIndexLocation, Mesh.Sections[0].BaseVertexLocation});
		}

		const int32_t NumRows = 10;    // 5 IBL groups x 2 metallic rows.
		const int32_t NumColumns = 11; // Roughness from RoughnessStart to 1.
		const float CellSize = 2.2f;
		for (int32_t RowIdx = 0; RowIdx < NumRows; ++RowIdx)
		{
			const int32_t GroupIdx = RowIdx / 2;
			const int32_t MetallicIdx = RowIdx % 2;
			const float Metallic = (float)MetallicIdx;
			const int IBLMode = GIBLModeGroupOrder[GroupIdx];

			for (int32_t ColumnIdx = 0; ColumnIdx < NumColumns; ++ColumnIdx)
			{
				float RoughnessT = (float)ColumnIdx / (NumColumns - 1);
				float Roughness = (1 - RoughnessT) * Root.RoughnessStart + RoughnessT * 1.0f;

				FStaticMeshInstance Instance = {};
				float X = CellSize * (-NumColumns * 0.5f + ColumnIdx + 0.5f);
				float Y = CellSize * (-NumRows * 0.5f + RowIdx + 0.5f);
				Instance.Position = XMFLOAT3(X, Y, 0.0f);
				Instance.MeshIndex = 1;
				Instance.Roughness = Roughness;
				Instance.RoughnessT = RoughnessT;
				Instance.Metallic = Metallic;
				Instance.IBLMode = IBLMode;

				Root.StaticMeshInstances.push_back(Instance);
			}
		}
	}

	// Static geometry vertex buffer (single buffer for all static meshes).
	{
		const D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(AllVertices.size() * sizeof(FVertex));

		ID3D12Resource *StagingVB;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&StagingVB)));
		TempResources.push_back(StagingVB);

		void *Ptr;
		VHR(StagingVB->Map(0, get_rvalue_ptr(CD3DX12_RANGE(0, 0)), &Ptr));
		memcpy(Ptr, AllVertices.data(), AllVertices.size() * sizeof(FVertex));
		StagingVB->Unmap(0, nullptr);

		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&Root.StaticVB)));

		Root.StaticVBView.BufferLocation = Root.StaticVB->GetGPUVirtualAddress();
		Root.StaticVBView.StrideInBytes = sizeof(FVertex);
		Root.StaticVBView.SizeInBytes = (UINT)AllVertices.size() * sizeof(FVertex);

		Gfx.CmdList->CopyResource(Root.StaticVB, StagingVB);
		Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.StaticVB, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)));
	}

	// Static geometry index buffer (single buffer for all static meshes).
	{
		const D3D12_RESOURCE_DESC Desc = CD3DX12_RESOURCE_DESC::Buffer(AllIndices.size() * sizeof(uint32_t));

		ID3D12Resource *StagingIB;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&StagingIB)));
		TempResources.push_back(StagingIB);

		void *Ptr;
		VHR(StagingIB->Map(0, get_rvalue_ptr(CD3DX12_RANGE(0, 0)), &Ptr));
		memcpy(Ptr, AllIndices.data(), AllIndices.size() * sizeof(uint32_t));
		StagingIB->Unmap(0, nullptr);

		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &Desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&Root.StaticIB)));

		Root.StaticIBView.BufferLocation = Root.StaticIB->GetGPUVirtualAddress();
		Root.StaticIBView.Format = DXGI_FORMAT_R32_UINT;
		Root.StaticIBView.SizeInBytes = (UINT)AllIndices.size() * sizeof(uint32_t);

		Gfx.CmdList->CopyResource(Root.StaticIB, StagingIB);
		Gfx.CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(Root.StaticIB, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDEX_BUFFER)));
	}

	Gfx.CmdList->IASetVertexBuffers(0, 1, &Root.StaticVBView);
	Gfx.CmdList->IASetIndexBuffer(&Root.StaticIBView);

	// Create BRDFIntegrationMap.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_GenerateBRDFIntegrationMap]);
	Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_GenerateBRDFIntegrationMap]);
	CreateBRDFIntegrationMap(Gfx, Root.BRDFIntegrationMap, Root.BRDFIntegrationMapSRV, TempResources);

	Root.EnvironmentMapIndex = Root.SelectedEnvironmentMapIndex = 0;
	Root.PendingEnvironmentMapIndex = -1;

	RecordEnvironmentPrecomputation(Root, TempResources, TexturesThatNeedMipmaps);

	// Setup resources for MSAA.
	{
		CD3DX12_RESOURCE_DESC DescColor = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, Gfx.Resolution[0], Gfx.Resolution[1], 1, 1, Root.NumSamples);
		DescColor.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &DescColor, D3D12_RESOURCE_STATE_RENDER_TARGET, get_rvalue_ptr(CD3DX12_CLEAR_VALUE(DXGI_FORMAT_R8G8B8A8_UNORM, XMVECTORF32{0.0f})), IID_PPV_ARGS(&Root.MSColorBuffer)));

		CD3DX12_RESOURCE_DESC DescDepth = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, Gfx.Resolution[0], Gfx.Resolution[1], 1, 1, Root.NumSamples);
		DescDepth.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE, &DescDepth, D3D12_RESOURCE_STATE_DEPTH_WRITE, get_rvalue_ptr(CD3DX12_CLEAR_VALUE(DXGI_FORMAT_D32_FLOAT, 1.0f, 0)), IID_PPV_ARGS(&Root.MSDepthBuffer)));

		Root.MSColorBufferRTV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1);
		Root.MSDepthBufferDSV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1);

		Gfx.Device->CreateRenderTargetView(Root.MSColorBuffer, nullptr, Root.MSColorBufferRTV);
		Gfx.Device->CreateDepthStencilView(Root.MSDepthBuffer, nullptr, Root.MSDepthBufferDSV);
	}

	// Create accumulation buffer.
	{
		CD3DX12_RESOURCE_DESC DescAccum = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, Gfx.Resolution[0], Gfx.Resolution[1],
																	   1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

		VHR(Gfx.Device->CreateCommittedResource(get_rvalue_ptr(CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT)), D3D12_HEAP_FLAG_NONE,
												&DescAccum, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&Root.AccumulationBuffer)));

		Root.AccumulationBufferSRV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);

		D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc = {};
		SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
		SRVDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		SRVDesc.Texture2D.MipLevels = 1;
		SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

		Gfx.Device->CreateShaderResourceView(Root.AccumulationBuffer, &SRVDesc, Root.AccumulationBufferSRV);
	}

	// Execute "data upload" and "data generation" GPU commands, create mipmaps, destroy temp resources when GPU is done.
	FinishEnvironmentPrecomputation(Root, TempResources, TexturesThatNeedMipmaps);

	// Root.CameraPosition = XMFLOAT3(0.0f, 0.0f, -10.0f);
	Root.CameraPosition = XMFLOAT3(0.0f, 0.0f, 12.0f);
	Root.CameraFocusPosition = XMFLOAT3(0.0f, 0.0f, 0.0f);

	Root.LastIBLMode = Root.IBLMode = IBL_MODE_SPLIT_SUM_APPROXIMATION;
	Root.LastMaterialMode = Root.MaterialMode = MATERIAL_MODE_DIFFUSE_AND_SPECULAR;
	Root.NumFrames = 0;
}

static void Shutdown(FDemoRoot &Root)
{
	for (ID3D12RootSignature *Signature : Root.RootSignatures)
	{
		SAFE_RELEASE(Signature);
	}
	for (ID3D12PipelineState *Pipeline : Root.Pipelines)
	{
		SAFE_RELEASE(Pipeline);
	}
	SAFE_RELEASE(Root.StaticVB);
	SAFE_RELEASE(Root.StaticIB);
	SAFE_RELEASE(Root.EnvMap);
	SAFE_RELEASE(Root.IrradianceMap);
	SAFE_RELEASE(Root.PrefilteredEnvMap);
	SAFE_RELEASE(Root.BRDFIntegrationMap);
	SAFE_RELEASE(Root.SHEMatrixA);
	SAFE_RELEASE(Root.SHEMatrixb);
	SAFE_RELEASE(Root.SHEMatrixAT);
	SAFE_RELEASE(Root.SHESHCoeff);
	if (Root.SHECoeffLUT)
	{
		Root.SHECoeffLUT->Unmap(0, nullptr);
	}
	SAFE_RELEASE(Root.SHECoeffLUT);
	if (Root.ClassicSHCoeff)
	{
		Root.ClassicSHCoeff->Unmap(0, nullptr);
	}
	SAFE_RELEASE(Root.ClassicSHCoeff);
	if (Root.SGLobes)
	{
		Root.SGLobes->Unmap(0, nullptr);
	}
	SAFE_RELEASE(Root.SGLobes);
	SAFE_RELEASE(Root.MSColorBuffer);
	SAFE_RELEASE(Root.MSDepthBuffer);
	SAFE_RELEASE(Root.AccumulationBuffer);
	DestroyUIContext(Root.UI);
}

static int32_t Run(FDemoRoot &Root)
{
	EA::StdC::Init();
	ImGui::CreateContext();

	HWND Window = CreateSimpleWindow("ImageBasedPBR", 1920, 1080);
	CreateGraphicsContext(Window, /*bShouldCreateDepthBuffer*/ false, Root.Gfx);

	Initialize(Root);

	for (;;)
	{
		// Handle a pending window resize before drawing the next frame so the
		// swap chain and all size-dependent resources match the new client area.
		uint32_t ResizeWidth = 0;
		uint32_t ResizeHeight = 0;
		if (PollWindowResize(ResizeWidth, ResizeHeight))
		{
			OnWindowResize(Root, ResizeWidth, ResizeHeight);
		}

		MSG Message = {};
		if (PeekMessage(&Message, 0, 0, 0, PM_REMOVE))
		{
			DispatchMessage(&Message);
			if (Message.message == WM_QUIT)
			{
				break;
			}
		}
		else
		{
			Update(Root);
			Draw(Root);
			PresentFrame(Root.Gfx, 0);
		}
	}

	WaitForGPU(Root.Gfx);
	Shutdown(Root);
	DestroyGraphicsContext(Root.Gfx);
	ImGui::DestroyContext();
	EA::StdC::Shutdown();

	return 0;
}

int32_t CALLBACK WinMain(_In_ HINSTANCE, _In_opt_ HINSTANCE, _In_ LPSTR, _In_ int32_t)
{
	SetProcessDPIAware();
	FDemoRoot Root = {};
	return Run(Root);
}
