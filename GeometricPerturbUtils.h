#pragma once

#if defined(USE_CUDA)
#include <cuda_runtime.h>

#define FunctionDecoration __host__ __device__

#else

#define FunctionDecoration 

#endif

// Assumes pVector3f is an array of 3 floats
// FunctionDecoration void NormalizeVector3f(float *pVector3f);

// // This has an out parameter
// FunctionDecoration void RotationMatrixAroundAxis(float angle, float *pAxisVector3f, float *pMatrix3x3);

// FunctionDecoration float DotVector3fVector3f(float *lhs, float *rhs);

// FunctionDecoration float MagVector3f(float *pVec);

// FunctionDecoration void RotateVectorByMatrix(float *pRotationMatrix, float *pVector);