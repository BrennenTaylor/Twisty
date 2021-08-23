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

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <optional>
#include <random>

namespace twisty
{
    /**
     * @brief Implements the ExperimentRunner interface for running the experiment on the CPU
     *
     */
    class FullExperimentRunner : public ExperimentRunner
    {
    public:
        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         * @param kdsRange Range of allowed curvature * ds values
         * @param tdsRange Range of allowed torsion * ds values
         */
        FullExperimentRunner(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~FullExperimentRunner();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

    private:
        /*

        flag parameter is set to 0 for no errors.
        If the root solve fails, its set to 1
        If the root solve succeeds, but we dont accept the path due to calculated path error, we set to 2
        */

        void GeometryPerturb(
            int64_t threadIdx,
            int64_t numExperimentPaths,
            int64_t numPathsPerThread,
            int64_t numPathsToSkipPerThread,
            int64_t numSegmentsPerCurve,
            std::vector<std::mt19937_64>& rngGenerators,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<Farlor::Vector3>& scratchPositionSpace,
            std::vector<Farlor::Vector3>& scratchTangentSpace,
            std::vector<float>& scratchCurvatureSpace,
            std::vector<double>& globalPathWeights,
            std::vector<double>& cachedSegmentWeights,
            float segmentLength,
            const twisty::PathWeighting::WeightLookupTableIntegral& weightingIntegral,
            const twisty::PerturbUtils::BoundrayConditions& boundaryConditions,
            const PathWeighting::NormalizerStuff::FN& fn
        );

        //void SpringBasedPerturb(
        //    int64_t threadIdx,
        //    int64_t numExperimentPaths,
        //    int64_t numPathsPerThread,
        //    int64_t numPathsToSkipPerThread,
        //    int64_t numSegmentsPerCurve,
        //    std::vector<std::mt19937_64>& rngGenerators,
        //    std::vector<Farlor::Vector3>& globalPos,
        //    std::vector<Farlor::Vector3>& globalTans,
        //    std::vector<float>& globalCurvatures,
        //    std::vector<double>& globalPathWeights,
        //    std::vector<double>& cachedSegmentWeights,
        //    float segmentLength,
        //    float scattering,
        //    float absorbtion,
        //    const std::vector<double>& lookupTable,
        //    float minCurvature,
        //    float maxCurvature,
        //    float curvatureStepSize
        //);

        //void HybridMethod(
        //    int64_t threadIdx,
        //    int64_t numExperimentPaths,
        //    int64_t numPathsPerThread,
        //    int64_t numPathsToSkipPerThread,
        //    int64_t numSegmentsPerCurve,
        //    std::vector<std::mt19937>& rngGenerators,
        //    std::vector<Farlor::Vector3>& globalPos,
        //    std::vector<Farlor::Vector3>& globalTans,
        //    std::vector<float>& globalCurvatures,
        //    std::vector<double>& globalPathWeights,
        //    std::vector<double>& cachedSegmentWeights,
        //    float segmentLength,
        //    float scattering,
        //    float absorbtion,
        //    const std::vector<double>& lookupTable,
        //    float minCurvature,
        //    float maxCurvature,
        //    float curvatureStepSize
        //);

        void WeightCombineThreadKernel(const int64_t threadIdx, int64_t numWeights, int64_t numWeightsPerThread, float arclength, int64_t numSegmentsPerCurve,
            const std::vector<double>& compressedWeights, std::vector<boost::multiprecision::cpp_dec_float_100>& bigFloatWeightsLog10,
            boost::multiprecision::cpp_dec_float_100& threadWeight, boost::multiprecision::cpp_dec_float_100 pathNormalizer);

    private:
        std::unique_ptr<PathWeighting::RegularizedIntegral> m_upRegIntEvaluator;
    };
}