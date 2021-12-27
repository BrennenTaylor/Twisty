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
        assert(combinedWeightValue.m_numValues < MaxNumPathsPerCombinedWeight);

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
        if (combinedWeightValue.m_maxWeightLog10 >= valueLog10)
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

        // Adjust values already stored
        double log10RunningTotal = log10(combinedWeightValue.m_runningTotal);
        double adjustedLog10RunningTotal = log10RunningTotal + offsetDelta;
        combinedWeightValue.m_runningTotal = pow(10.0, adjustedLog10RunningTotal);

        // Update with new value
        combinedWeightValue.m_maxWeightLog10 = valueLog10;
        combinedWeightValue.m_maxPossibleFinalWeightLog10 = newMaxPossibleFinalWeightLog10;
        combinedWeightValue.m_offset = newOffset;

        combinedWeightValue.m_runningTotal += pow(10.0, (valueLog10 + combinedWeightValue.m_offset));
        combinedWeightValue.m_numValues++;
    }

    __host__ __device__ CombinedWeightValues_C CombinedWeightValues_C_CombineValues(const CombinedWeightValues_C& firstCombinedWeightValue,
        const CombinedWeightValues_C& secondCombinedWeightValue)
    {
        // printf("New combined number: %d\n", (firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues));
        assert((firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues) <= MaxNumPathsPerCombinedWeight);

        if (firstCombinedWeightValue.m_numValues > 0 && secondCombinedWeightValue.m_numValues == 0)
        {
            return firstCombinedWeightValue;
        }

        if (secondCombinedWeightValue.m_numValues > 0 && firstCombinedWeightValue.m_numValues == 0)
        {
            return secondCombinedWeightValue;
        }

        if (secondCombinedWeightValue.m_maxPossibleFinalWeightLog10 > firstCombinedWeightValue.m_maxPossibleFinalWeightLog10)
        {
            // printf("\tSwapping Order\n");
            // printf("\tFirst: %lf\n", firstCombinedWeightValue.m_maxPossibleFinalWeightLog10);
            // printf("\tSecond: %lf\n", secondCombinedWeightValue.m_maxPossibleFinalWeightLog10);
            return CombinedWeightValues_C_CombineValues(secondCombinedWeightValue, firstCombinedWeightValue);
        }

        // We can assume that the first weight value is greater or equal.
        // If equal, we just combine.
        if (firstCombinedWeightValue.m_maxPossibleFinalWeightLog10 == secondCombinedWeightValue.m_maxPossibleFinalWeightLog10)
        {

            // printf("\tEqual Offset\n");
            // Basic combine
            CombinedWeightValues_C combined;
            combined.m_maxPossibleFinalWeightLog10 = firstCombinedWeightValue.m_maxPossibleFinalWeightLog10;
            combined.m_maxWeightLog10 = firstCombinedWeightValue.m_maxWeightLog10;
            combined.m_numValues = firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues;
            combined.m_offset = firstCombinedWeightValue.m_offset;
            combined.m_runningTotal = firstCombinedWeightValue.m_runningTotal + secondCombinedWeightValue.m_runningTotal;
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
        combined.m_maxPossibleFinalWeightLog10 = firstCombinedWeightValue.m_maxPossibleFinalWeightLog10;
        combined.m_maxWeightLog10 = firstCombinedWeightValue.m_maxWeightLog10;
        combined.m_numValues = firstCombinedWeightValue.m_numValues + secondCombinedWeightValue.m_numValues;
        combined.m_offset = firstCombinedWeightValue.m_offset;
        combined.m_runningTotal = firstCombinedWeightValue.m_runningTotal + newSecondRunningTotal;
        return combined;
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