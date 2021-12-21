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

namespace twisty
{
    class FullExperimentRunnerOptimalPerturb : public ExperimentRunner
    {
    public:
        FullExperimentRunnerOptimalPerturb(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~FullExperimentRunnerOptimalPerturb();

        virtual ExperimentRunner::RunnerSpecificResults RunnerSpecificRunExperiment() override;

    private:

        void GeometryRandom(
            int64_t threadIdx,
            int64_t numExperimentPaths,
            int64_t numPathsPerThread,
            int64_t numPathsToSkipPerThread,
            int64_t numSegmentsPerCurve,
            std::vector<std::mt19937_64>& rngGenerators,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<double>& globalPathWeights,
            std::vector<double>& cachedSegmentWeights,
            std::vector<CombinedWeightValues_C>& perThreadCombinedWeightValues,
            float segmentLength,
            twisty::PathWeighting::BaseWeightLookupTable* weightingIntegralPtr,
            const twisty::PerturbUtils::BoundaryConditions& boundaryConditions,
            const PathWeighting::NormalizerStuff::BaseNormalizer& pathNormalizer
        );

        void GeometryCombined(
            int64_t threadIdx,
            int64_t numExperimentPaths,
            int64_t numPathsPerThread,
            int64_t numPathsToSkipPerThread,
            int64_t numSegmentsPerCurve,
            std::vector<std::mt19937_64>& rngGenerators,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<Farlor::Vector3>& scratchPositionSpaceLeft,
            std::vector<Farlor::Vector3>& scratchTangentSpaceLeft,
            std::vector<float>& scratchCurvatureSpaceLeft,
            std::vector<Farlor::Vector3>& scratchPositionSpaceRight,
            std::vector<Farlor::Vector3>& scratchTangentSpaceRight,
            std::vector<float>& scratchCurvatureSpaceRight,
            std::vector<double>& globalPathWeights,
            std::vector<double>& cachedSegmentWeights,
            float segmentLength,
            twisty::PathWeighting::BaseWeightLookupTable* weightingIntegralPtr,
            const twisty::PerturbUtils::BoundaryConditions& boundaryConditions,
            const PathWeighting::NormalizerStuff::BaseNormalizer& pathNormalizer
        );

        void WeightCombineThreadKernel(const int64_t threadIdx, int64_t numWeights, int64_t numWeightsPerThread,
            float arclength, int64_t numSegmentsPerCurve, const twisty::WeightingMethod weightingMethod,
            const std::vector<double>& compressedWeights, std::vector<boost::multiprecision::cpp_dec_float_100>& bigFloatWeightsLog10,
            std::vector<boost::multiprecision::cpp_dec_float_100>& threadScatterWeights, boost::multiprecision::cpp_dec_float_100 pathNormalizer,
            boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10);
    };
}