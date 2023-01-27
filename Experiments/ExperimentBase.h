#include <ExperimentRunner.h>
#include <PathWeighters.h>
#include "CombinedWeightUtils.h"

#include "MathConsts.h"
#include "boost/multiprecision/detail/default_ops.hpp"

namespace twisty {
namespace ExperimentBase {

    struct FiveSegmentAngleIntegrationResult {
        uint64_t numValidPaths = 0;
        uint64_t numPathsTotal = 0;
        boost::multiprecision::cpp_dec_float_100 totalWeight;
        double minPathWeightLog10;
        double maxPathWeightLog10;
    };

    FiveSegmentAngleIntegrationResult FiveSegmentAngleIntegration(const uint32_t numPhi1Vals,
          const uint32_t numTheta1Vals, const uint32_t numTheta2Vals,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);
}
}