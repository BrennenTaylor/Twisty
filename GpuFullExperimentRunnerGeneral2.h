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
    __global__ void GpuFullExperimentRunnerGeneral2_InitializeCurandState(uint32_t seed, curandState_t *pStates, uint32_t maxNumStates);
    __device__ void GpuFullExperimentRunnerGeneral2_GeneratePathBatchPerturbations(
        double* pFinalThreadWeightsShared,
        double* pFinalThreadWeightsDifferences,
        double* pFinalThreadMaxes,
        int32_t numExperimentPaths,
        int32_t numPathsPerThread,
        int32_t numPathsToSkipPerThread,
        uint32_t numSegmentsPerCurve,
        curandState_t* pRandStates,
        float* pPerThreadPositions,
        float* pPerThreadTangents,
        float* pPerThreadCurvatures,

        double* pCachedSegmentWeights,

        double* pPerBlockFinalWeights,
        double* pPerBlockFinalDifferences,

        float segmentLength,
        float scattering,
        float absorbtion,
        double* pLookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize
    );

    __global__ void GpuFullExperimentRunnerGeneral2_PerturbControl(
        int32_t numExperimentPaths,
        int32_t numPathsPerThread,
        int32_t numPathsToSkipPerThread,
        uint32_t numSegmentsPerCurve,
        curandState_t* pRandStates,
        float* pPerThreadPositions,
        float* pPerThreadTangents,
        float* pPerThreadCurvatures,

        double* pCachedSegmentWeights,

        double* pPerBlockFinalWeights,
        double* pPerBlockFinalDifferences,

        float segmentLength,
        float scattering,
        float absorbtion,
        double* pLookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize
    );

    /**
     * @brief Implements the ExperimentRunner interface for running the experiment on the CPU
     *
     */
    class GpuFullExperimentRunnerGeneral2 : public ExperimentRunner
    {

    public:
        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         */
        GpuFullExperimentRunnerGeneral2(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~GpuFullExperimentRunnerGeneral2();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

       // Cuda Stuff
    private:
        bool SetupCudaDevice();
        bool SetupCuRandStates(uint32_t numGlobalPerturbThreads);
        bool SetupCrossDispatchCurveData(uint32_t gridSize, uint32_t blockSize);
        bool SetupWeightLookupTexture(const twisty::PathWeighting::WeightLookupTableIntegral& lookupEvaluator);
        void CleanupCudaMemory();

    private:
        std::mt19937 m_rng;

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

        // Values to use on gpu
        double* m_pPerThreadSegmentWeightCache = nullptr;

        // Values read back from gpu
        double* m_pPerBlockFinalWeights = nullptr;
        double* m_pPerBlockFinalDifferences = nullptr;
    };
}