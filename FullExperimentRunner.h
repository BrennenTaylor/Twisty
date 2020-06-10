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
#include "Range.h"
#include "PathWeightUtils.h"

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
        FullExperimentRunner(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper, Range kdsRange, Range tdsRange);
        virtual ~FullExperimentRunner();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

    private:
        ExperimentResults RunExperimentLogWeightTable();

    private:
        /*

        flag parameter is set to 0 for no errors.
        If the root solve fails, its set to 1
        If the root solve succeeds, but we dont accept the path due to calculated path error, we set to 2
        */
        //std::unique_ptr<Curve> PurturbCurve(const Curve& curve, uint32_t& flag);

        //std::unique_ptr<Curve> SimpleGeometryCurvePerturb(const Curve& curve, uint32_t& flag);

        void LogWeightThreadFunction(
            uint32_t threadIdx,
            int32_t numExperimentPaths,
            int32_t numPathsPerThread,
            int32_t numPathsToSkipPerThread,
            int32_t numSegmentsPerCurve,
            std::vector<std::mt19937>& rngGenerators,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<double>& globalPathWeights,
            std::vector<double>& cachedSegmentWeights,
            float segmentLength,
            float scattering,
            float absorbtion,
            const std::vector<double>& lookupTable,
            float minCurvature,
            float maxCurvature,
            float curvatureStepSize
        );

        void SpringBasedPerturb(
            uint32_t threadIdx,
            int32_t numExperimentPaths,
            int32_t numPathsPerThread,
            int32_t numPathsToSkipPerThread,
            int32_t numSegmentsPerCurve,
            std::vector<std::mt19937>& rngGenerators,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<double>& globalPathWeights,
            std::vector<double>& cachedSegmentWeights,
            float segmentLength,
            float scattering,
            float absorbtion,
            const std::vector<double>& lookupTable,
            float minCurvature,
            float maxCurvature,
            float curvatureStepSize
        );

        void HybridMethod(
            uint32_t threadIdx,
            int32_t numExperimentPaths,
            int32_t numPathsPerThread,
            int32_t numPathsToSkipPerThread,
            int32_t numSegmentsPerCurve,
            std::vector<std::mt19937>& rngGenerators,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<double>& globalPathWeights,
            std::vector<double>& cachedSegmentWeights,
            float segmentLength,
            float scattering,
            float absorbtion,
            const std::vector<double>& lookupTable,
            float minCurvature,
            float maxCurvature,
            float curvatureStepSize
        );

        void WeightCombineThreadKernel(const uint32_t threadIdx, uint32_t numWeights, uint32_t numWeightsPerThread, float arclength, uint32_t numSegmentsPerCurve,
            const std::vector<double>& compressedWeights, std::vector<boost::multiprecision::cpp_dec_float_100>& bigFloatWeights, twisty::BigFloat& threadWeight);

    private:
        std::unique_ptr<PathSpaceUtils::RegularizedIntegral> m_upRegIntEvaluator;
    };
}