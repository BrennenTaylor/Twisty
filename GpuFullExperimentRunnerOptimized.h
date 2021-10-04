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
    class GpuFullExperimentRunnerOptimized : public ExperimentRunner
    {

    public:

        // A device curve will be of the form
        // For num segments M
        // Positions * (M + 1) followed by Tangents * M
        struct GpuDeviceVector3Aligned
        {
            float x;
            float y;
            float z;
            // TODO: Do we need this?
            float _pad;
        };

    public:
        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         */
        GpuFullExperimentRunnerOptimized(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~GpuFullExperimentRunnerOptimized();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

       // Cuda Stuff
    private:
        bool SetupCudaDevice();
        bool SetupKernelDispatchParameters(uint32_t numPathWeightsInShared, uint32_t& numGlobalPerturbThreads, uint32_t& perturbBlockSize, uint32_t& perturbGridSize);
        bool SetupCuRandStates(uint32_t numGlobalPerturbThreads);
        bool SetupCurveDataStructures();
        bool SetupWeightLookupTexture(const twisty::PathSpaceUtils::LogWeightLookupTableIntegral& lookupEvaluator);

        void CleanupWeightLookupTexture();

        void WeightCombineThreadKernel(const uint32_t threadIdx, uint32_t numWeights, uint32_t numWeightsPerThread, const std::vector<float>& compressedWeights, twisty::BigFloat& threadWeight);

    private:
        // This really should be generalized or better thought out for experimentation sake
        __global__ void InitializeCurandState(uint32_t seed, curandState_t* pStates, uint32_t maxNumStates);

        // Dispatch which atually runs the purtibation algorithm on the GPU
        __global__ void GeneratePathBatchPutrubations(
            uint32_t numExperimentPaths,
            uint32_t numPathsPerThread,
            uint32_t numCachedPathWeightsPerThread,
            uint32_t numSegmentsPerCurve,
            curandState_t* pRandStates,
            float* pGlobalPathWeights,
            float segmentLength,
            float scattering,
            float absorbtion,
            cudaTextureObject_t texObj,
            float minCurvature,
            float maxCurvature);

    private:
        std::mt19937 m_rng;

        // Cuda Stuff
    private:
        int32_t m_numSMs;
        int32_t m_warpSize;
        int32_t m_maxThreadsPerMultiprocessor;

        // Device Memory - Unique Per Thread
        curandState_t* m_pPerGlobalThreadRandStates = nullptr;

        // Values to read back from gpu
        float* m_pPerPathCompressedWeightGlobal = nullptr;


        // Curve Positions Per Thread
        float* m_pPerThreadPositions = nullptr;

        // Curve Tangents Per Thread
        float* m_pPerThreadTangents = nullptr;

        // Lookup texture stuff
        cudaArray* m_pWeightLookupArray = nullptr;
        cudaTextureObject_t m_weightTextureObj = 0;
    };
}