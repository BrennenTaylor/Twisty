#pragma once

#include <cstdint>

#include <FMath\Vector3.h>

#include <boost/multiprecision/cpp_dec_float.hpp>

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

// Currently implemented as #defines as there is no good way that I can find to make these a class specific constant.
// We can fit a maximum of 1e6 paths in a double as of now. Dont try more, it wont work yet.
#define MaxDoubleLog10 300
#define MaxNumberOfPathsLog10 6
#define MaxNumPathsPerCombinedWeight 1000000

namespace twisty {
// Initializer cant be used for cuda, thus the reset function!
struct CombinedWeightValues_C {
    double m_maxWeightLog10 = 0.0;
    double m_runningTotal = 0.0;
    double m_offset = 0.0;
    uint32_t m_numValues = 0;
};

__host__ __device__ void CombinedWeightValues_C_Reset(CombinedWeightValues_C &combinedWeightValue);
__host__ __device__ void CombinedWeightValues_C_AddValue(
      CombinedWeightValues_C &combinedWeightValue, double valueLog10);
__host__ __device__ CombinedWeightValues_C CombinedWeightValues_C_CombineValues(
      CombinedWeightValues_C firstCombinedWeightValue,
      CombinedWeightValues_C secondCombinedWeightValue);
boost::multiprecision::cpp_dec_float_100 ExtractFinalValue(
      const CombinedWeightValues_C &combinedWeightValue);
}