#include <ExperimentRunner.h>
#include <PathWeighters.h>
#include "CombinedWeightUtils.h"

#include "MathConsts.h"
#include "boost/multiprecision/detail/default_ops.hpp"

namespace twisty {
namespace ExperimentBase {

    struct Result {
        uint64_t numValidPaths = 0;
        uint64_t numPathsTotal = 0;
        boost::multiprecision::cpp_dec_float_100 totalWeight;
        double minPathWeightLog10;
        double maxPathWeightLog10;
    };

    Result FiveSegmentAngleIntegration(const uint32_t numPhi1Vals, const uint32_t numTheta1Vals,
          const uint32_t numTheta2Vals,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);

    Result FiveSegmentAngleSpaceMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);

    Result SixSegmentAngleSpaceMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);

    Result FiveSegmentPathGenerationMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);

    Result SixSegmentPathGenerationMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);

    Result MSegmentPathGenerationMC(const int64_t numExperimentPaths, const int numSegmentsPerCurve,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);

    Result MSegmentPathGenerationMC(const uint64_t seed, const int64_t numExperimentPaths,
          const int numSegmentsPerCurve,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &cachedWeightLookupTable,
          const float maxDs);

    Result MSegmentPathGenerationMC(const int64_t numExperimentPaths, const int numSegmentsPerCurve,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &environmentCachedWeightLookupTable,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &objectCachedWeightLookupTable,
          const float maxDs);
}
}