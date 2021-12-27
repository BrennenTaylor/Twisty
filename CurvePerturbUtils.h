#pragma once

#include <cstdint>

#include <FMath\Vector3.h>

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
    __device__ __host__ void RotationMatrixAroundAxis(const float angle, const float* pAxisVector3f, float* pMatrix3x3);
    __device__ __host__ float DotVector3fVector3f(float* lhs, float* rhs);
    __device__ __host__ float MagVector3f(float* pVec);
    __device__ __host__ void RotateVectorByMatrix(float* pRotationMatrix, float* pVector);
}

namespace twisty
{
    struct WeightingParameters;
}

namespace twisty
{
    namespace PerturbUtils
    {
        struct BoundaryConditions
        {
            Farlor::Vector3 m_startPos = Farlor::Vector3(0.0, 0.0, 0.0);
            Farlor::Vector3 m_startDir = Farlor::Vector3(1.0, 0.0, 0.0);
            Farlor::Vector3 m_endPos = Farlor::Vector3(0.0, 0.0, 0.0);
            Farlor::Vector3 m_endDir = Farlor::Vector3(1.0, 0.0, 0.0);
            float arclength = 0.0f;
        };

        struct BoundaryConditions_CudaSafe
        {
            float m_startPos[3] = { 0.0, 0.0, 0.0 };
            float m_startDir[3] = {0.0, 0.0, 0.0};
            float m_endPos[3] = {0.0, 0.0, 0.0};
            float m_endDir[3] = {0.0, 0.0, 0.0};
            float arclength = 0.0f;
        };

        void UpdateTangentsFromPos(Farlor::Vector3* pPositions, Farlor::Vector3* pTangents,
            const uint32_t numSegments, const BoundaryConditions& boundaryConditions);

        void UpdateCurvaturesFromTangents(Farlor::Vector3* pTangents, float* pCurvatures,
            const uint32_t numSegments, const BoundaryConditions& boundaryConditions, int32_t weightingMethod);

        __device__ __host__ void UpdateTangentsFromPosCudaSafe(float* pPositions, float* pTangents,
            const uint32_t numSegments, const BoundaryConditions_CudaSafe& boundaryConditions);

        __device__ __host__ void UpdateCurvaturesFromTangentsCudaSafe(float* pTangents, float* pCurvatures,
            const uint32_t numSegments, const BoundaryConditions_CudaSafe& boundaryConditions, int32_t weightingMethod);
    }
}