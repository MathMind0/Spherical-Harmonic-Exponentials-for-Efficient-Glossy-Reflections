// Wave64 variant (AMD): one group == one wave of 64 lanes.
#define GROUP_SIZE_X 64
#define SAMPLES_PER_THREAD 16
#include "SHE_Diffuse.hlsl"
