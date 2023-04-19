#pragma once

#include <cstdint>

#include <FMath/Vector3.h>

#if defined(USE_CUDA)
#include <cuda_occupancy.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#else
#define __device__
#define __host__
#endif

namespace twisty {
// A device curve will be of the form
// For num segments M
// Positions * (M + 1) followed by Tangents * M
struct GpuDeviceVector3Aligned {
    float x;
    float y;
    float z;
    float w;
};

const int32_t PositionFloatCount = 3;
const int32_t TangentFloatCount = 3;

// Assumes pVector3f is an array of 3 floats
__device__ __host__ void NormalizeVector3f(float *pVector3f);
// This has an outparameter

// Likely faster on cpu but slower on gpu
__device__ __host__ void RotationMatrixAroundAxis_AngleAsIs_CudaSafe(
      const float angle, const float *pAxisVector3f, float *pMatrix3x3);

// Likely slower on cpu but faster on GPU
__device__ __host__ void RotationMatrixAroundAxis_MultiplyAngleByPi_CudaSafe(
      const float angleNeedsPi, const float *pAxisVector3f, float *pMatrix3x3);


__device__ __host__ float DotVector3fVector3f(float *lhs, float *rhs);
__device__ __host__ float MagVector3f(float *pVec);
__device__ __host__ void RotateVectorByMatrix(float *pRotationMatrix, float *pVector);

__host__ __device__ void RotateQuatByQuat(
      float const *const pRotatedQuat, float const *const pRotateByQuat, float *const pResult);
__host__ __device__ void RotateVectorByQuaternion(float *pQuaternionRotation, float *pVector);

}  // namespace twisty

namespace twisty {
struct WeightingParameters;
}

namespace twisty {
namespace PerturbUtils {
    struct BoundaryConditions {
        Farlor::Vector3 m_startPos = Farlor::Vector3(0.0, 0.0, 0.0);
        Farlor::Vector3 m_startDir = Farlor::Vector3(1.0, 0.0, 0.0);
        Farlor::Vector3 m_endPos = Farlor::Vector3(0.0, 0.0, 0.0);
        Farlor::Vector3 m_endDir = Farlor::Vector3(1.0, 0.0, 0.0);
        float arclength = 0.0f;
    };

    struct BoundaryConditions_CudaSafe {
        float m_startPos[3] = { 0.0, 0.0, 0.0 };
        float m_startDir[3] = { 0.0, 0.0, 0.0 };
        float m_endPos[3] = { 0.0, 0.0, 0.0 };
        float m_endDir[3] = { 0.0, 0.0, 0.0 };
        float arclength = 0.0f;
    };

    void UpdateTangentsFromPos(Farlor::Vector3 *pPositions,
          Farlor::Vector3 *pTangents,
          const uint32_t numSegments,
          const BoundaryConditions &boundaryConditions);

    void UpdateCurvaturesFromTangents_RadiativeTransfer(Farlor::Vector3 *pTangents,
          float *pCurvatures,
          const uint32_t numSegments,
          const BoundaryConditions &boundaryConditions);

    void UpdateCurvaturesFromTangents_SimplifiedModel(Farlor::Vector3 *pTangents,
          float *pCurvatures,
          const uint32_t numSegments,
          const BoundaryConditions &boundaryConditions);

    __device__ __host__ void UpdateTangentsFromPos_CudaSafe(float *__restrict const pPositions,
          float *__restrict const pTangents, const uint32_t numSegments,
          const BoundaryConditions_CudaSafe &boundaryConditions);

    __device__ __host__ void UpdateCurvaturesFromTangents_RadiativeTransfer_CudaSafe(
          float *pTangents, float *pCurvatures, const uint32_t numSegments,
          const BoundaryConditions_CudaSafe &boundaryConditions);

    __device__ __host__ void UpdateCurvaturesFromTangents_SimplifiedModel_CudaSafe(float *pTangents,
          float *pCurvatures, const uint32_t numSegments,
          const BoundaryConditions_CudaSafe &boundaryConditions);
}  // namespace PerturbUtils
}  // namespace twisty


#ifdef ACTUAL_IMPLEMENTATION

#include "PathWeightUtils.h"

#include "MathConsts.h"

#include <cmath>
#include <assert.h>

// Cuda Functions
namespace twisty {
// Assumes pVector3f is an array of 3 floats
__host__ __device__ void NormalizeVector3f(float *pVector3f)
{
    float normalizer
          = pVector3f[0] * pVector3f[0] + pVector3f[1] * pVector3f[1] + pVector3f[2] * pVector3f[2];
    normalizer = 1.0f / sqrtf(normalizer);
    pVector3f[0] *= normalizer;
    pVector3f[1] *= normalizer;
    pVector3f[2] *= normalizer;
}

// This has an outparameter
__host__ __device__ void RotationMatrixAroundAxis_AngleAsIs_CudaSafe(
      const float angle, const float *pAxisVector3f, float *pMatrix3x3)
{
    const float cosAngle = cosf(angle);
    const float sinAngle = sinf(angle);

    // Ensure its normalized
    // TODO: Make assertion
    // NormalizeVector3f(pAxisVector3f);

    pMatrix3x3[0] = cosAngle + pAxisVector3f[0] * pAxisVector3f[0] * (1.0f - cosAngle);
    pMatrix3x3[1]
          = pAxisVector3f[0] * pAxisVector3f[1] * (1.0f - cosAngle) - pAxisVector3f[2] * sinAngle;
    pMatrix3x3[2]
          = pAxisVector3f[0] * pAxisVector3f[2] * (1.0f - cosAngle) + pAxisVector3f[1] * sinAngle;

    pMatrix3x3[3]
          = pAxisVector3f[1] * pAxisVector3f[0] * (1.0f - cosAngle) + pAxisVector3f[2] * sinAngle;
    pMatrix3x3[4] = cosAngle + pAxisVector3f[1] * pAxisVector3f[1] * (1 - cosAngle);
    pMatrix3x3[5]
          = pAxisVector3f[1] * pAxisVector3f[2] * (1 - cosAngle) - pAxisVector3f[0] * sinAngle;

    pMatrix3x3[6]
          = pAxisVector3f[2] * pAxisVector3f[0] * (1 - cosAngle) - pAxisVector3f[1] * sinAngle;
    pMatrix3x3[7]
          = pAxisVector3f[2] * pAxisVector3f[1] * (1 - cosAngle) + pAxisVector3f[0] * sinAngle;
    pMatrix3x3[8] = cosAngle + pAxisVector3f[2] * pAxisVector3f[2] * (1 - cosAngle);
}

__host__ void RotationMatrixAroundAxis_MultiplyAngleByPi_CudaSafe(
      const float angleNeedsPi, const float *pAxisVector3f, float *pMatrix3x3)
{
#ifndef __CUDA_ARCH__
    const float cosAngle = std::cos(angleNeedsPi * TwistyPi);
    const float sinAngle = std::sin(angleNeedsPi * TwistyPi);
#else
    const float cosAngle = cospi(angleNeedsPi);
    const float sinAngle = sinpi(angleNeedsPi);
#endif

    pMatrix3x3[0] = cosAngle + pAxisVector3f[0] * pAxisVector3f[0] * (1.0f - cosAngle);
    pMatrix3x3[1]
          = pAxisVector3f[0] * pAxisVector3f[1] * (1.0f - cosAngle) - pAxisVector3f[2] * sinAngle;
    pMatrix3x3[2]
          = pAxisVector3f[0] * pAxisVector3f[2] * (1.0f - cosAngle) + pAxisVector3f[1] * sinAngle;

    pMatrix3x3[3]
          = pAxisVector3f[1] * pAxisVector3f[0] * (1.0f - cosAngle) + pAxisVector3f[2] * sinAngle;
    pMatrix3x3[4] = cosAngle + pAxisVector3f[1] * pAxisVector3f[1] * (1 - cosAngle);
    pMatrix3x3[5]
          = pAxisVector3f[1] * pAxisVector3f[2] * (1 - cosAngle) - pAxisVector3f[0] * sinAngle;

    pMatrix3x3[6]
          = pAxisVector3f[2] * pAxisVector3f[0] * (1 - cosAngle) - pAxisVector3f[1] * sinAngle;
    pMatrix3x3[7]
          = pAxisVector3f[2] * pAxisVector3f[1] * (1 - cosAngle) + pAxisVector3f[0] * sinAngle;
    pMatrix3x3[8] = cosAngle + pAxisVector3f[2] * pAxisVector3f[2] * (1 - cosAngle);
}

__host__ __device__ float DotVector3fVector3f(float *lhs, float *rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

__host__ __device__ float MagVector3f(float *pVec)
{
    return sqrt(pVec[0] * pVec[0] + pVec[1] * pVec[1] + pVec[2] * pVec[2]);
}

__host__ __device__ void RotateQuatByQuat(
      float const *const pRotatedQuat, float const *const pRotateByQuat, float *const pResult)
{
    // https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html
    const float t0 = (pRotatedQuat[0] * pRotateByQuat[0] - pRotatedQuat[1] * pRotateByQuat[1]
          - pRotatedQuat[2] * pRotateByQuat[2] - pRotatedQuat[3] * pRotateByQuat[3]);

    const float t1 = (pRotatedQuat[0] * pRotateByQuat[1] + pRotatedQuat[1] * pRotateByQuat[0]
          - pRotatedQuat[2] * pRotateByQuat[3] + pRotatedQuat[3] * pRotateByQuat[2]);

    const float t2 = (pRotatedQuat[0] * pRotateByQuat[2] + pRotatedQuat[1] * pRotateByQuat[3]
          + pRotatedQuat[2] * pRotateByQuat[0] - pRotatedQuat[3] * pRotateByQuat[1]);

    const float t3 = (pRotatedQuat[0] * pRotateByQuat[3] - pRotatedQuat[1] * pRotateByQuat[2]
          + pRotatedQuat[2] * pRotateByQuat[1] + pRotatedQuat[3] * pRotateByQuat[0]);

    pResult[0] = t0;
    pResult[1] = t1;
    pResult[2] = t2;
    pResult[3] = t3;
}

__host__ __device__ void RotateVectorByQuaternion(float *pQuaternionRotation, float *pVector)
{
    float pointQuat[4] = { 0.0f, pVector[0], pVector[1], pVector[2] };

    float invQuat[4] = { pQuaternionRotation[0], -pQuaternionRotation[1], -pQuaternionRotation[2],
        -pQuaternionRotation[3] };

    RotateQuatByQuat(pointQuat, pQuaternionRotation, pointQuat);
    RotateQuatByQuat(invQuat, pointQuat, pointQuat);

    // Extract from quaternion
    pVector[0] = pointQuat[1];
    pVector[1] = pointQuat[2];
    pVector[2] = pointQuat[3];
}

__host__ __device__ void RotateVectorByMatrix(float *pRotationMatrix, float *pVector)
{
    float val[3];
    val[0] = DotVector3fVector3f(pRotationMatrix, pVector);
    val[1] = DotVector3fVector3f(pRotationMatrix + 3, pVector);
    val[2] = DotVector3fVector3f(pRotationMatrix + 6, pVector);

    // Write it back to pVector
    pVector[0] = val[0];
    pVector[1] = val[1];
    pVector[2] = val[2];
}
}


namespace twisty {
namespace PerturbUtils {
    // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
    __host__ void UpdateTangentsFromPos(Farlor::Vector3 *pPositions, Farlor::Vector3 *pTangents,
          const uint32_t numSegments, const BoundaryConditions &boundaryConditions)
    {
        BoundaryConditions_CudaSafe cs;
        cs.m_startPos[0] = boundaryConditions.m_startPos.m_data[0];
        cs.m_startPos[1] = boundaryConditions.m_startPos.m_data[1];
        cs.m_startPos[2] = boundaryConditions.m_startPos.m_data[2];

        cs.m_startDir[0] = boundaryConditions.m_startDir.m_data[0];
        cs.m_startDir[1] = boundaryConditions.m_startDir.m_data[1];
        cs.m_startDir[2] = boundaryConditions.m_startDir.m_data[2];

        cs.m_endPos[0] = boundaryConditions.m_endPos.m_data[0];
        cs.m_endPos[1] = boundaryConditions.m_endPos.m_data[1];
        cs.m_endPos[2] = boundaryConditions.m_endPos.m_data[2];

        cs.m_endDir[0] = boundaryConditions.m_endDir.m_data[0];
        cs.m_endDir[1] = boundaryConditions.m_endDir.m_data[1];
        cs.m_endDir[2] = boundaryConditions.m_endDir.m_data[2];

        cs.arclength = boundaryConditions.arclength;

        UpdateTangentsFromPos_CudaSafe((float *)pPositions, (float *)pTangents, numSegments, cs);
    }

    // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
    __host__ void UpdateCurvaturesFromTangents_RadiativeTransfer(Farlor::Vector3 *pTangents,
          float *pCurvatures, const uint32_t numSegments,
          const BoundaryConditions &boundaryConditions)
    {
        BoundaryConditions_CudaSafe cs;
        cs.m_startPos[0] = boundaryConditions.m_startPos.m_data[0];
        cs.m_startPos[1] = boundaryConditions.m_startPos.m_data[1];
        cs.m_startPos[2] = boundaryConditions.m_startPos.m_data[2];

        cs.m_startDir[0] = boundaryConditions.m_startDir.m_data[0];
        cs.m_startDir[1] = boundaryConditions.m_startDir.m_data[1];
        cs.m_startDir[2] = boundaryConditions.m_startDir.m_data[2];

        cs.m_endPos[0] = boundaryConditions.m_endPos.m_data[0];
        cs.m_endPos[1] = boundaryConditions.m_endPos.m_data[1];
        cs.m_endPos[2] = boundaryConditions.m_endPos.m_data[2];

        cs.m_endDir[0] = boundaryConditions.m_endDir.m_data[0];
        cs.m_endDir[1] = boundaryConditions.m_endDir.m_data[1];
        cs.m_endDir[2] = boundaryConditions.m_endDir.m_data[2];

        cs.arclength = boundaryConditions.arclength;

        UpdateCurvaturesFromTangents_RadiativeTransfer_CudaSafe(
              (float *)pTangents, pCurvatures, numSegments, cs);
    }

    // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
    __host__ void UpdateCurvaturesFromTangents_SimplifiedModel(Farlor::Vector3 *pTangents,
          float *pCurvatures, const uint32_t numSegments,
          const BoundaryConditions &boundaryConditions)
    {
        BoundaryConditions_CudaSafe cs;
        cs.m_startPos[0] = boundaryConditions.m_startPos.m_data[0];
        cs.m_startPos[1] = boundaryConditions.m_startPos.m_data[1];
        cs.m_startPos[2] = boundaryConditions.m_startPos.m_data[2];

        cs.m_startDir[0] = boundaryConditions.m_startDir.m_data[0];
        cs.m_startDir[1] = boundaryConditions.m_startDir.m_data[1];
        cs.m_startDir[2] = boundaryConditions.m_startDir.m_data[2];

        cs.m_endPos[0] = boundaryConditions.m_endPos.m_data[0];
        cs.m_endPos[1] = boundaryConditions.m_endPos.m_data[1];
        cs.m_endPos[2] = boundaryConditions.m_endPos.m_data[2];

        cs.m_endDir[0] = boundaryConditions.m_endDir.m_data[0];
        cs.m_endDir[1] = boundaryConditions.m_endDir.m_data[1];
        cs.m_endDir[2] = boundaryConditions.m_endDir.m_data[2];

        cs.arclength = boundaryConditions.arclength;

        UpdateCurvaturesFromTangents_SimplifiedModel_CudaSafe(
              (float *)pTangents, pCurvatures, numSegments, cs);
    }

    // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
    __host__ __device__ void UpdateTangentsFromPos_CudaSafe(float *__restrict const pPositions,
          float *__restrict const pTangents, const uint32_t numSegments,
          const BoundaryConditions_CudaSafe &csBoundaryConditions)
    {
        const float invDS = numSegments / csBoundaryConditions.arclength;

        // TODO: Is Forward Difference good enough?
        for (uint32_t i = 0; i < numSegments; ++i) {
            float diff_x = (pPositions[((i + 1) * 3) + 0] - pPositions[(i * 3) + 0]) * invDS;
            float diff_y = (pPositions[((i + 1) * 3) + 1] - pPositions[(i * 3) + 1]) * invDS;
            float diff_z = (pPositions[((i + 1) * 3) + 2] - pPositions[(i * 3) + 2]) * invDS;
            float mag = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
            assert(mag > 0.0f);
            mag = 1.0f / std::sqrt(mag);
            pTangents[i * 3 + 0] = diff_x * mag;
            pTangents[i * 3 + 1] = diff_y * mag;
            pTangents[i * 3 + 2] = diff_z * mag;
        }
    }

    // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
    __host__ __device__ void UpdateCurvaturesFromTangents_RadiativeTransfer_CudaSafe(
          float *pTangents, float *pCurvatures, const uint32_t numSegments,
          const BoundaryConditions_CudaSafe &boundaryConditions)
    {
        const float invDs = numSegments / boundaryConditions.arclength;
        // Update segments
        for (uint32_t i = 0; i < (numSegments - 1); ++i) {
            const float tanLeft_x = pTangents[i * 3 + 0];
            const float tanLeft_y = pTangents[i * 3 + 1];
            const float tanLeft_z = pTangents[i * 3 + 2];

            const float tanRight_x = pTangents[(i + 1) * 3 + 0];
            const float tanRight_y = pTangents[(i + 1) * 3 + 1];
            const float tanRight_z = pTangents[(i + 1) * 3 + 2];

            {
                const float scaledDiff_x = (tanRight_x - tanLeft_x) * invDs;
                const float scaledDiff_y = (tanRight_y - tanLeft_y) * invDs;
                const float scaledDiff_z = (tanRight_z - tanLeft_z) * invDs;

                pCurvatures[i] = sqrt(scaledDiff_x * scaledDiff_x + scaledDiff_y * scaledDiff_y
                      + scaledDiff_z * scaledDiff_z);
            }
        }
    }

    // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
    __host__ __device__ void UpdateCurvaturesFromTangents_SimplifiedModel_CudaSafe(float *pTangents,
          float *pCurvatures, const uint32_t numSegments,
          const BoundaryConditions_CudaSafe &boundaryConditions)
    {
        // Update segments
        for (uint32_t i = 0; i < (numSegments - 1); ++i) {
            const float tanLeft_x = pTangents[i * 3 + 0];
            const float tanLeft_y = pTangents[i * 3 + 1];
            const float tanLeft_z = pTangents[i * 3 + 2];

            const float tanRight_x = pTangents[(i + 1) * 3 + 0];
            const float tanRight_y = pTangents[(i + 1) * 3 + 1];
            const float tanRight_z = pTangents[(i + 1) * 3 + 2];

            {
                const float curvature = (tanLeft_x * tanRight_x) + (tanLeft_y * tanRight_y)
                      + (tanLeft_z * tanRight_z);
                pCurvatures[i]
                      = curvature;  // Negate curvature so that an increase in curvature leads to higher weight values
            }
        }
    }
}
}

#endif