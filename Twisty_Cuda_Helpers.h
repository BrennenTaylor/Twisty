#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <string>

namespace twisty
{
    void CudaSafeErrorCheck(cudaError_t error, std::string message);
    __device__ void Multiply3x3(float pFirst[9], float pSecond[9]);
    __device__ void UpdateRotationU(float pRotation[9], float curvature,
        float torsion, float ds);
}