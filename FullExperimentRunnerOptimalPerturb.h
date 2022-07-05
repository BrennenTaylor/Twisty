/**
 * @file ExperimentRunnerCpu.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-03-18
 *
 * @copyright Copyright (c) 2019
 *
 */

#include "ExperimentRunner.h"
#include "PathWeightUtils.h"
#include "PerturbUtils.h"

#include "CombinedWeightUtils.h"

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <optional>
#include <random>

namespace twisty {
class FullExperimentRunnerOptimalPerturb : public ExperimentRunner {
   public:
    FullExperimentRunnerOptimalPerturb(
          ExperimentRunner::ExperimentParameters &experimentParams, Bootstrapper &bootstrapper);
    virtual ~FullExperimentRunnerOptimalPerturb();

    virtual ExperimentRunner::RunnerSpecificResults RunnerSpecificRunExperiment() override;

   private:
    void GeometryRandom(int64_t threadIdx,
          uint64_t numExperimentPaths,
          uint64_t numPathsPerThread,
          uint32_t numPathsToSkipPerThread,
          uint32_t numSegmentsPerCurve,
          std::vector<std::mt19937_64> &rngGenerators,
          std::vector<Farlor::Vector3> &globalPos,
          std::vector<Farlor::Vector3> &globalTans,
          std::vector<float> &globalCurvatures,
          std::vector<CombinedWeightValues_C> &perThreadCombinedWeightValues,
          float segmentLength,
          const float *pWeightLookupTable,
          const uint32_t weightLookupTableSize,
          const float ds,
          const float minCurvature,
          const float maxCurvature,
          const float curvatureStepSize,
          const twisty::PerturbUtils::BoundaryConditions &boundaryConditions);


    void GeometryRandom_ExportPaths(int64_t threadIdx,
          uint64_t numExperimentPaths,
          uint64_t numPathsPerThread,
          uint32_t numPathsToSkipPerThread,
          uint32_t numSegmentsPerCurve,
          std::vector<std::mt19937_64> &rngGenerators,
          std::vector<Farlor::Vector3> &globalPos,
          std::vector<Farlor::Vector3> &globalTans,
          std::vector<float> &globalCurvatures,
          std::vector<CombinedWeightValues_C> &perThreadCombinedWeightValues,
          float segmentLength,
          const float *pWeightLookupTable,
          const uint32_t weightLookupTableSize,
          const float ds,
          const float minCurvature,
          const float maxCurvature,
          const float curvatureStepSize,
          const twisty::PerturbUtils::BoundaryConditions &boundaryConditions);
};
}