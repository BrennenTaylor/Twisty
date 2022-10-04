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
    class FullExperimentRunnerOldMethodBridge : public ExperimentRunner
    {
    public:
        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         */
        FullExperimentRunnerOldMethodBridge(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~FullExperimentRunnerOldMethodBridge();

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
        //std::unique_ptr<Curve> PurturbCurve(const Curve& curve, int64_t& flag);

        //std::unique_ptr<Curve> SimpleGeometryCurvePerturb(const Curve& curve, int64_t& flag);

        void LogWeightThreadFunction(
            int64_t threadIdx,
            int64_t dispatchIdx,
            int64_t numPerturbThreads,
            int64_t numExperimentPaths,
            int64_t numPathsPerThread,
            int64_t numPathsToSkipPerThread,
            int64_t numSegmentsPerCurve,
            std::vector<std::mt19937_64>& rngGenerators,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<double>& cachedSegmentWeights,
            std::vector<double>& finalThreadWeight,
            std::vector<double>& finalThreadDifference,
            float segmentLength,
            float scattering,
            float absorbtion,
            const std::vector<double>& lookupTable,
            float minCurvature,
            float maxCurvature,
            float curvatureStepSize,
            std::string pathToRawBinary,
            std::vector<double>& cachedWeights
        );
    };
}