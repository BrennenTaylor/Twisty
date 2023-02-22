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

#ifdef ACTUAL_IMPLEMENTATION

namespace twisty {
__host__ __device__ void CombinedWeightValues_C_Reset(CombinedWeightValues_C &combinedWeightValue)
{
    combinedWeightValue.m_numValues = 0;
    combinedWeightValue.m_runningTotal = 0.0;
    combinedWeightValue.m_offset = 0.0;
    combinedWeightValue.m_maxWeightLog10 = 0.0;
}

__host__ __device__ void CombinedWeightValues_C_AddValue(
      CombinedWeightValues_C &combinedWeightValue, double valueLog10)
{
    assert(combinedWeightValue.m_numValues < MaxNumPathsPerCombinedWeight);

    // In the case we haven't added a value yet, we can early out
    if (combinedWeightValue.m_numValues == 0) {
        combinedWeightValue.m_maxWeightLog10 = valueLog10;
        combinedWeightValue.m_offset
              = MaxDoubleLog10 - (combinedWeightValue.m_maxWeightLog10 + MaxNumberOfPathsLog10);
        combinedWeightValue.m_runningTotal
              += pow(10.0, (valueLog10 + combinedWeightValue.m_offset));
        combinedWeightValue.m_numValues++;
        return;
    }


    // If we already have a value and its not larger than the current max, then throw it in.
    if (combinedWeightValue.m_maxWeightLog10 >= valueLog10) {
        combinedWeightValue.m_runningTotal
              += pow(10.0, (valueLog10 + combinedWeightValue.m_offset));
        combinedWeightValue.m_numValues++;
        return;
    }

    // If it is larger, we need to rescale everything around that new value

    // New difference
    double newOffset = MaxDoubleLog10 - (valueLog10 + MaxNumberOfPathsLog10);
    double offsetDelta = newOffset - combinedWeightValue.m_offset;

    // Adjust values already stored
    double log10RunningTotal = log10(combinedWeightValue.m_runningTotal);
    double adjustedLog10RunningTotal = log10RunningTotal + offsetDelta;
    combinedWeightValue.m_runningTotal = pow(10.0, adjustedLog10RunningTotal);

    // Update with new value
    combinedWeightValue.m_maxWeightLog10 = valueLog10;
    combinedWeightValue.m_offset = newOffset;

    combinedWeightValue.m_runningTotal += pow(10.0, (valueLog10 + combinedWeightValue.m_offset));
    combinedWeightValue.m_numValues++;
    assert(combinedWeightValue.m_numValues <= MaxNumPathsPerCombinedWeight);
}

__host__ __device__ CombinedWeightValues_C CombinedWeightValues_C_CombineValues(
      CombinedWeightValues_C firstCombinedWeightValue,
      CombinedWeightValues_C secondCombinedWeightValue)
{
    // printf("New combined number: %d\n", (firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues));
    assert((firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues)
          <= MaxNumPathsPerCombinedWeight);

    if (firstCombinedWeightValue.m_numValues > 0 && secondCombinedWeightValue.m_numValues == 0) {
        return firstCombinedWeightValue;
    }

    if (secondCombinedWeightValue.m_numValues > 0 && firstCombinedWeightValue.m_numValues == 0) {
        return secondCombinedWeightValue;
    }

    if (secondCombinedWeightValue.m_maxWeightLog10 > firstCombinedWeightValue.m_maxWeightLog10) {
        // Avoid recursive call by swaping combined weight values
        CombinedWeightValues_C temp = secondCombinedWeightValue;
        secondCombinedWeightValue = firstCombinedWeightValue;
        firstCombinedWeightValue = temp;
    }

    // We can assume that the first weight value is greater or equal.
    // If equal, we just combine.
    if (firstCombinedWeightValue.m_maxWeightLog10 == secondCombinedWeightValue.m_maxWeightLog10) {
        // printf("\tEqual Offset\n");
        // Basic combine
        CombinedWeightValues_C combined;
        combined.m_maxWeightLog10 = firstCombinedWeightValue.m_maxWeightLog10;
        combined.m_numValues
              = firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues;
        assert(combined.m_numValues <= MaxNumPathsPerCombinedWeight);
        combined.m_offset = firstCombinedWeightValue.m_offset;
        combined.m_runningTotal
              = firstCombinedWeightValue.m_runningTotal + secondCombinedWeightValue.m_runningTotal;
        return combined;
    }

    // printf("\tAdjusted Second\n");

    // New difference
    double offsetDelta = firstCombinedWeightValue.m_offset - secondCombinedWeightValue.m_offset;

    // Adjust values already stored
    double secongLog10RunningTotal = log10(secondCombinedWeightValue.m_runningTotal);
    double adjustedSecondLog10RunningTotal = secongLog10RunningTotal + offsetDelta;
    double newSecondRunningTotal = pow(10.0, adjustedSecondLog10RunningTotal);

    CombinedWeightValues_C combined;
    combined.m_maxWeightLog10 = firstCombinedWeightValue.m_maxWeightLog10;
    combined.m_numValues
          = firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues;
    assert(combined.m_numValues <= MaxNumPathsPerCombinedWeight);
    combined.m_offset = firstCombinedWeightValue.m_offset;
    combined.m_runningTotal = firstCombinedWeightValue.m_runningTotal + newSecondRunningTotal;
    return combined;
}

boost::multiprecision::cpp_dec_float_100 ExtractFinalValue(
      const CombinedWeightValues_C &combinedWeightValue)
{
    assert(combinedWeightValue.m_numValues <= MaxNumPathsPerCombinedWeight);

    // TODO: Do we need this, or can we simply compute using running total and offset?
    if (combinedWeightValue.m_numValues == 0) {
        return 0.0;
    }

    boost::multiprecision::cpp_dec_float_100 runningTotalLog10
          = std::log10(combinedWeightValue.m_runningTotal);
    runningTotalLog10 -= combinedWeightValue.m_offset;
    return boost::multiprecision::pow(10.0, runningTotalLog10);
}
}

#endif ACTUAL_IMPLEMENTATION