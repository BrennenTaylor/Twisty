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

#include <optional>
#include <random>

#include <curand.h>
#include <curand_kernel.h>

namespace twisty
{
    /**
     * @brief Implements the ExperimentRunner interface for running the experiment on the CPU
     *
     */
    class GpuFullExperimentRunnerGeneral : public ExperimentRunner
    {

    public:
        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         * @param kdsRange Range of allowed curvature * ds values
         * @param tdsRange Range of allowed torsion * ds values
         */
        GpuFullExperimentRunnerGeneral(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper, Range kdsRange, Range tdsRange);
        virtual ~GpuFullExperimentRunnerGeneral();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

       // Cuda Stuff
    private:
        bool SetupCudaDevice();
        bool SetupKernelDispatchParameters(uint32_t numPathWeightsInShared, uint32_t& numGlobalPerturbThreads, uint32_t& perturbBlockSize, uint32_t& perturbGridSize);
        bool SetupCuRandStates(uint32_t numGlobalPerturbThreads);
        bool SetupCrossDispatchCurveData(uint32_t numGlobalPerturbThreads);
        bool SetupWeightLookupTexture(const twisty::PathWeighting::WeightLookupTableIntegral& lookupEvaluator);

        void WeightCombineThreadKernel(const uint32_t threadIdx, uint32_t numWeights, uint32_t numWeightsPerThread,
            float arclength, uint32_t numSegmentsPerCurve, const std::vector<double>& compressedWeights,
            std::vector<boost::multiprecision::cpp_dec_float_100>& bigFloatWeights, boost::multiprecision::cpp_dec_float_100& threadWeight);

        void CleanupCudaMemory();

    private:
        std::mt19937 m_rng;

        std::unique_ptr<PathWeighting::RegularizedIntegral> m_upRegIntEvaluator;

        // Cuda Stuff
    private:
        int32_t m_numSMs;
        int32_t m_warpSize;
        int32_t m_maxThreadsPerMultiprocessor;

        // Device Memory - Unique Per Thread
        curandState_t* m_pPerGlobalThreadRandStates = nullptr;

        double* m_pWeightLookupTable = nullptr;

        // Curve Positions Per Thread
        float* m_pPerThreadPositions = nullptr;
        // Curve Tangents Per Thread
        float* m_pPerThreadTangents = nullptr;
        // Curve Tangents Per Thread
        float* m_pPerThreadCurvatures = nullptr;

        // Values to read back from gpu
        double* m_pPerThreadSegmentWeightCache = nullptr;
        double* m_pPerPathCompressedWeightGlobal = nullptr;
    };
}