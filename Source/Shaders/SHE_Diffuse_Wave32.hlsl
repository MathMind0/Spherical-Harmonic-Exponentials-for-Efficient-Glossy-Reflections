// Wave32 variant (NVIDIA): one group == one wave of 32 lanes.
#define GROUP_SIZE_X 32
#define SAMPLES_PER_THREAD 16
#include "SHE_Diffuse.hlsl"
