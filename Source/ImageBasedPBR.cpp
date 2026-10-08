#include "Library.h"
#include "CPUAndGPUCommon.h"
#include "d3dx12.h"
#include "imgui/imgui.h"
#include "EAStdC/EAStdC.h"
#include "EAStdC/EASprintf.h"
#include "EAStdC/EABitTricks.h"
#include "EAStdC/EAString.h"
#include "stb_image.h"
#include "cgltf.h"

#define MESH_MAX_NUM_SECTIONS 4

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

// Draw order of the IBL mode groups shown top-to-bottom in the sphere array.
static const int GIBLModeGroupOrder[3] =
{
	IBL_MODE_SPHERICAL_HARMONICS_EXPONENTIAL,
	IBL_MODE_REFERENCE,
	IBL_MODE_SPLIT_SUM_APPROXIMATION,
};

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
	float RoughnessStart = 0.4f;
	uint32_t NumSamples;
	uint32_t NumFrames;
	uint32_t WaveSize = 32; // GPU wave size in lanes (32 NVIDIA/Intel, 64 AMD), used to select SHE diffuse PSO variants.
};

static void RebuildEnvironment(FDemoRoot &Root);

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
		ImGui::TextDisabled("SHE / Reference / Split-Sum (shown as 3 groups)");
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
		ImGui::Text("Roughness Range Start");
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
}

static void Update(FDemoRoot &Root)
{
	double Time;
	float DeltaTime;
	UpdateFrameStats(Root.Gfx.Window, "ImageBasedPBR", Time, DeltaTime, Root.NumFrames);
	UpdateUI(Root, DeltaTime);

	if (Root.PendingEnvironmentMapIndex >= 0)
	{
		RebuildEnvironment(Root);
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

	// Projection: orthographic fits the 10x6 sphere grid with margin (grid spans
	// 22 x 13.2 units). Perspective uses the orbit distance to frame the grid.
	XMMATRIX ProjectionTransform;
	if (Root.bOrthographicProjection)
	{
		const float AspectRatio = 1.777f;
		const float GridWidth = 10 * 2.2f;
		const float GridHeight = 6 * 2.2f;
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
		const float AspectRatio = 1.777f;
		const float GridWidth = 10 * 2.2f;
		const float GridHeight = 6 * 2.2f;
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

		for (int32_t GroupIdx = 0; GroupIdx < 3; ++GroupIdx)
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

			CD3DX12_CPU_DESCRIPTOR_HANDLE TableBaseCPU;
			CD3DX12_GPU_DESCRIPTOR_HANDLE TableBaseGPU;
			AllocateGPUDescriptors(Gfx, 7, TableBaseCPU, TableBaseGPU);

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

static void SHEBuild(FGraphicsContext &Gfx, ID3D12Resource *&OutSHEMatrixA, ID3D12Resource *&OutSHEMatrixb, D3D12_CPU_DESCRIPTOR_HANDLE &OutSHEMatrixASRV, D3D12_CPU_DESCRIPTOR_HANDLE &OutSHEMatrixbSRV, D3D12_CPU_DESCRIPTOR_HANDLE EnvMapSRV)
{
	const uint32_t ViewCountSqrt = 8;
	const uint32_t NormalCountSqrt = 8;
	const uint32_t ViewCount = ViewCountSqrt * ViewCountSqrt;
	const uint32_t NormalCount = NormalCountSqrt * NormalCountSqrt;
	const uint32_t RoughnessCount = 4;
	const uint32_t SphericalHarmonicCount = 33; // 25(SH5) + 9(SH3) - 1(DC)

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
	CmdList->Dispatch(8, 8, 4);

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
									OutSHEMatrixA, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
									OutSHEMatrixb, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
}

static void SHEReduction(FDemoRoot &Root, FGraphicsContext &Gfx, ID3D12Resource *&OutSHEMatrixAT, D3D12_CPU_DESCRIPTOR_HANDLE &OutSHEMatrixATSRV, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixASRV, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixbSRV,
						 eastl::vector<ID3D12Resource *> &OutTempResources)
{
	const uint32_t ViewCountSqrt = 8;
	const uint32_t NormalCountSqrt = 8;
	const uint32_t ViewCount = ViewCountSqrt * ViewCountSqrt;
	const uint32_t NormalCount = NormalCountSqrt * NormalCountSqrt;
	const uint32_t RoughnessCount = 4;
	const uint32_t SphericalHarmonicCount = 33; // 25(SH5) + 9(SH3) - 1(DC)
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
	CmdList->Dispatch(1, 1, GroupCountZ);

	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::Transition(
									TempSHEMatrixAT3D, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));

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
										OutSHEMatrixAT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)));
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
		UAVDesc.Buffer.NumElements = 34;
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

static void SHECalibrate(FGraphicsContext &Gfx, ID3D12Resource *SHESHCoeff, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixASRV, D3D12_CPU_DESCRIPTOR_HANDLE SHEMatrixbSRV)
{
	const uint32_t ViewCountSqrt = 8;
	const uint32_t NormalCountSqrt = 8;
	const uint32_t ViewCount = ViewCountSqrt * ViewCountSqrt;
	const uint32_t NormalCount = NormalCountSqrt * NormalCountSqrt;
	const uint32_t RoughnessCount = 4;
	const uint32_t SphericalHarmonicCount = 33;

	const D3D12_CPU_DESCRIPTOR_HANDLE TempSHESHCoeffUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
		UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		UAVDesc.Buffer.NumElements = 34;
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
	//            cosine-band convolution) into SHESHCoeff slots [34..42].
	const uint32_t TotalSampleCount = 65536;
	const uint32_t SamplesPerThread = 16;
	const uint32_t ThreadCountX = Root.WaveSize; // group size == wave size
	const uint32_t GroupCount = TotalSampleCount / (SamplesPerThread * ThreadCountX); // 128 (wave32) / 64 (wave64)

	// Whole-buffer UAV over SHESHCoeff: the finalize pass writes slots [34..43]
	// with absolute indices, leaving the specular coefficients [0..33] untouched.
	const D3D12_CPU_DESCRIPTOR_HANDLE SHESHCoeffUAV = AllocateDescriptors(Gfx, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1);
	{
		D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc = {};
		UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
		UAVDesc.Buffer.NumElements = 44;
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

	// UAV barrier: all stage-1 writes must complete before stage 2 reads.
	CmdList->ResourceBarrier(1, get_rvalue_ptr(CD3DX12_RESOURCE_BARRIER::UAV(PartialRadiance)));

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

	// SHE Build.
	Gfx.CmdList->SetPipelineState(Root.Pipelines[PSO_SHE_Build]);
	Gfx.CmdList->SetComputeRootSignature(Root.RootSignatures[PSO_SHE_Build]);
	SHEBuild(Gfx, Root.SHEMatrixA, Root.SHEMatrixb, Root.SHEMatrixASRV, Root.SHEMatrixbSRV, Root.EnvMapSRV);

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
	SHECalibrate(Gfx, Root.SHESHCoeff, Root.SHEMatrixASRV, Root.SHEMatrixbSRV);

	// 3-band SH diffuse irradiance from the environment map (two-stage
	// dispatch; PSOs are set inside SHEDiffuse per stage).
	SHEDiffuse(Root, Gfx, Root.SHESHCoeff, Root.EnvMapSRV, TempResources);
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

static void Initialize(FDemoRoot &Root)
{
	FGraphicsContext &Gfx = Root.Gfx;

	eastl::vector<ID3D12Resource *> TempResources;
	eastl::vector<ID3D12Resource *> TexturesThatNeedMipmaps;

	Root.NumSamples = 1;
	CreateUIContext(Gfx, Root.NumSamples, Root.UI, TempResources);
	CreatePipelines(Gfx, Root.NumSamples, Root.Pipelines, Root.RootSignatures, Root.WaveSize);

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

		const int32_t NumRows = 6;    // 3 IBL groups x 2 metallic rows.
		const int32_t NumColumns = 10; // Roughness from RoughnessStart to 1.
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
