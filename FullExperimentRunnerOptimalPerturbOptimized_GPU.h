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

#include <cuda_runtime_api.h>

#include "CurvePerturbUtils.h"

#include "CombinedWeightUtils.h"

#include "ExperimentRunner.h"
#include "PathWeightUtils.h"
#include "PerturbUtils.h"

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <curand.h>
#include <curand_kernel.h>

#include <optional>
#include <random>

namespace twisty
{
    /**
     * @brief Implements the ExperimentRunner interface for running the experiment on the CPU
     *
     */

    class FullExperimentRunnerOptimalPerturbOptimized_GPU : public ExperimentRunner
    {
    public:

        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         */
        FullExperimentRunnerOptimalPerturbOptimized_GPU(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~FullExperimentRunnerOptimalPerturbOptimized_GPU();

        virtual ExperimentRunner::RunnerSpecificResults RunnerSpecificRunExperiment() override;

    private:
        /*
            flag parameter is set to 0 for no errors.
            If the root solve fails, its set to 1
            If the root solve succeeds, but we dont accept the path due to calculated path error, we set to 2
        */

    private:
        bool SetupCudaDevice();
        void CleanupCudaDevice();

        bool SetupCuRandStates(uint32_t numGlobalPerturbThreads, uint32_t seed);
        void CleanupCudaRandStates();

        bool SetupCudaPerturb(uint32_t numGlobalPerturbThreads, uint32_t numCombinedWeightValues, const std::vector<double>& weightTable);
        void CleanupCudaPerturb();


    private:
        // CUDA STUFF
    private:
        int32_t m_numSMs = 0;
        int32_t m_warpSize = 0;
        int32_t m_maxThreadsPerMultiprocessor = 0;

        uint64_t m_usedDeviceMemoryInBytes = 0;

        curandState_t* m_pPerGlobalThreadRandStates = nullptr;
        
        float* m_pPerGlobalThreadScratchSpacePositions = nullptr;
        float* m_pPerGlobalThreadScratchSpaceTangents = nullptr;
        float* m_pPerGlobalThreadScratchSpaceCurvatures = nullptr;

        CombinedWeightValues_C* m_pPerThreadCombinedWeightValues = nullptr;
        CombinedWeightValues_C* m_pFinalCombinedValues = nullptr;
        double* m_pDeviceWeightLookupTable = nullptr;
    };

    __global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState(uint32_t seed, curandState_t* pStates, uint32_t maxNumStates);
}