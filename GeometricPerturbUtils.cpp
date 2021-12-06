#include "GeometricPerturbUtils.h"

#if defined(USE_CUDA)
#include <cuda_runtime.h>

#else

#include <cmath>

#endif

// Assumes pVector3f is an array of 3 floats
FunctionDecoration void NormalizeVector3f(float *pVector3f)
{
    float normalizer = pVector3f[0] * pVector3f[0] + pVector3f[1] * pVector3f[1] + pVector3f[2] * pVector3f[2];
    normalizer = 1.0 / sqrt(normalizer);
    pVector3f[0] *= normalizer;
    pVector3f[1] *= normalizer;
    pVector3f[2] *= normalizer;
}

// This has an out parameter
FunctionDecoration void RotationMatrixAroundAxis(float angle, float *pAxisVector3f, float *pMatrix3x3)
{
    // Ensure its normalized
    NormalizeVector3f(pAxisVector3f);

    pMatrix3x3[0] = cos(angle) + pAxisVector3f[0] * pAxisVector3f[0] * (1.0f - cos(angle));
    pMatrix3x3[1] = pAxisVector3f[0] * pAxisVector3f[1] * (1.0f - cos(angle)) - pAxisVector3f[2] * sin(angle);
    pMatrix3x3[2] = pAxisVector3f[0] * pAxisVector3f[2] * (1.0f - cos(angle)) + pAxisVector3f[1] * sin(angle);

    pMatrix3x3[3] = pAxisVector3f[1] * pAxisVector3f[0] * (1.0f - cos(angle)) + pAxisVector3f[2] * sin(angle);
    pMatrix3x3[4] = cos(angle) + pAxisVector3f[1] * pAxisVector3f[1] * (1 - cos(angle));
    pMatrix3x3[5] = pAxisVector3f[1] * pAxisVector3f[2] * (1 - cos(angle)) - pAxisVector3f[0] * sin(angle);

    pMatrix3x3[6] = pAxisVector3f[2] * pAxisVector3f[0] * (1 - cos(angle)) - pAxisVector3f[1] * sin(angle);
    pMatrix3x3[7] = pAxisVector3f[2] * pAxisVector3f[1] * (1 - cos(angle)) + pAxisVector3f[0] * sin(angle);
    pMatrix3x3[8] = cos(angle) + pAxisVector3f[2] * pAxisVector3f[2] * (1 - cos(angle));
}

FunctionDecoration float DotVector3fVector3f(float *lhs, float *rhs)
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

FunctionDecoration float MagVector3f(float *pVec)
{
    return sqrt(pVec[0] * pVec[0] + pVec[1] * pVec[1] + pVec[2] * pVec[2]);
}

FunctionDecoration void RotateVectorByMatrix(float *pRotationMatrix, float *pVector)
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