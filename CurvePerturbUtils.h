#pragma once

#include <cstdint>

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#else
#define __device__ 
#define __host__ 
#endif

namespace twisty
{

    // A device curve will be of the form
    // For num segments M
    // Positions * (M + 1) followed by Tangents * M
    struct GpuDeviceVector3Aligned
    {
        float x;
        float y;
        float z;
        float w;
    };

    const int32_t PositionFloatCount = 3;
    const int32_t TangentFloatCount = 3;

        // Assumes pVector3f is an array of 3 floats
    __device__ __host__ void NormalizeVector3f(float* pVector3f);
    // This has an outparameter
    __device__ __host__ void RotationMatrixAroundAxis(float angle, float* pAxisVector3f, float* pMatrix3x3);
    __device__ __host__ float DotVector3fVector3f(float* lhs, float* rhs);
    __device__ __host__ float MagVector3f(float* pVec);
    __device__ __host__ void RotateVectorByMatrix(float* pRotationMatrix, float* pVector);
}