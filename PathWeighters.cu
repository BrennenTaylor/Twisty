#include "PathWeighters.h"
#include <stdint.h>

namespace twisty {
namespace PathWeighting {
    // Assume we have good pointers
    double WeightCurveViaCurvatureLog10(float *pCurvatureStart, uint32_t numCurvatures,
          const twisty::PathWeighting::BaseWeightLookupTable &weightIntegral)
    {
        return WeightCurveViaCurvatureLog10_CudaSafe(pCurvatureStart, numCurvatures,
              weightIntegral.AccessLookupTable().data(), weightIntegral.AccessLookupTable().size(),
              weightIntegral.GetDs(), weightIntegral.GetMinCurvature(),
              weightIntegral.GetMaxCurvature(), weightIntegral.GetCurvatureStepSize());
    }

    __host__ __device__ double WeightCurveViaCurvatureLog10_CudaSafe(float *pCurvatureStart,
          uint32_t numCurvatures, const double *pWeightLookupTable,
          const int32_t weightLookupTableSize, const double ds, const double minCurvature,
          const double maxCurvature, const double curvatureStepSize)
    {
        if (!pCurvatureStart || (numCurvatures == 0)) {
            return 0.0;
        }

        // Calculate value
        double runningPathWeightLog10 = 0.0;
        for (int segIdx = 0; segIdx < numCurvatures; ++segIdx) {
            // Extract curvature
            double curvature = pCurvatureStart[segIdx];

            if (curvature < minCurvature) {
                printf("Error: curvature less than min curvature: %lf < %lf\n", curvature,
                      minCurvature);
                printf("Forcing to min curvature\n");
                curvature = minCurvature;
            }

            if (curvature > maxCurvature) {
                printf("Error: curvature greater than max curvature: %lf > %lf\n", curvature,
                      maxCurvature);
                printf("Forcing to max curvature\n");
                curvature = maxCurvature;
            }

            double distanceFromMin = curvature - minCurvature;
            double realIdx = distanceFromMin / curvatureStepSize;
            int32_t leftIdx = (int32_t)floor(realIdx);
            int32_t rightIdx = leftIdx + 1;

            if (leftIdx == (weightLookupTableSize - 1)) {
                rightIdx--;  // Bump it left 1, it doesnt really matter anymore anyways.
            }

            double leftLookup = pWeightLookupTable[leftIdx];
            double rightLookup = pWeightLookupTable[rightIdx];

            double interpDist = distanceFromMin - (leftIdx * curvatureStepSize);

            double interpolatedResult
                  = leftLookup * (1.0f - interpDist) + (rightLookup * interpDist);
            // Take the natural log of the interpolated results
            double interpolatedResultLog10 = std::log10(interpolatedResult);
            // Lets do weights as doubles for now
            double segmentWeightLog10 = interpolatedResultLog10;

            // Update the running path weight. We also want to cache the segment weights
            runningPathWeightLog10 += segmentWeightLog10;
        }
        return runningPathWeightLog10;
    }
}
}