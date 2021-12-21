#include "CombinedWeightUtils.h"

namespace twisty
{
        __host__ __device__ void CombinedWeightValues_C_Reset(CombinedWeightValues_C& combinedWeightValue)
    {
        combinedWeightValue.m_numValues = 0;
        combinedWeightValue.m_runningTotal = 0.0;
        combinedWeightValue.m_offset = 0.0;
        combinedWeightValue.m_maxWeightLog10 = 0.0;
        combinedWeightValue.m_maxPossibleFinalWeightLog10 = 0.0;
    }

    __host__ __device__ void CombinedWeightValues_C_AddValue(CombinedWeightValues_C& combinedWeightValue, double valueLog10)
    {
        // In the case we haven't added a value yet, we can early out
        if (combinedWeightValue.m_numValues == 0)
        {
            combinedWeightValue.m_maxWeightLog10 = valueLog10;
            combinedWeightValue.m_maxPossibleFinalWeightLog10 = combinedWeightValue.m_maxWeightLog10 + MaxNumberOfPathsLog10;
            combinedWeightValue.m_offset = MaxDoubleLog10 - combinedWeightValue.m_maxPossibleFinalWeightLog10;
            combinedWeightValue.m_runningTotal += pow(10.0, (valueLog10 + combinedWeightValue.m_offset));
            combinedWeightValue.m_numValues++;
            return;
        }


        // If we already have a value and its not larger than the current max, then throw it in.
        if (combinedWeightValue.m_maxWeightLog10 > valueLog10)
        {
            combinedWeightValue.m_runningTotal += pow(10.0, (valueLog10 + combinedWeightValue.m_offset));
            combinedWeightValue.m_numValues++;
            return;
        }

        // If it is larger, we need to rescale everything around that new value

        // New difference
        double newMaxPossibleFinalWeightLog10 = valueLog10 + MaxNumberOfPathsLog10;
        double newOffset = MaxDoubleLog10 - newMaxPossibleFinalWeightLog10;

        double offsetDelta = newOffset - combinedWeightValue.m_offset;
        double log10RunningTotal = log10(combinedWeightValue.m_runningTotal);
        double adjustedLog10RunningTotal = log10RunningTotal + offsetDelta;
        combinedWeightValue.m_runningTotal = pow(10.0, adjustedLog10RunningTotal);

        // Update
        combinedWeightValue.m_maxWeightLog10 = valueLog10;
        combinedWeightValue.m_maxPossibleFinalWeightLog10 = newMaxPossibleFinalWeightLog10;
        combinedWeightValue.m_offset = newOffset;

        combinedWeightValue.m_runningTotal += pow(10.0, (valueLog10 + combinedWeightValue.m_offset));
        combinedWeightValue.m_numValues++;
    }

    boost::multiprecision::cpp_dec_float_100 ExtractFinalValue(const CombinedWeightValues_C& combinedWeightValue)
    {
        // TODO: Do we need this, or can we simply compute using running total and offset?
        if (combinedWeightValue.m_numValues == 0)
        {
            return 0.0;
        }

        boost::multiprecision::cpp_dec_float_100 runningTotalLog10 = std::log10(combinedWeightValue.m_runningTotal);
        runningTotalLog10 -= combinedWeightValue.m_offset;
        return boost::multiprecision::pow(10.0, runningTotalLog10);
    }
}