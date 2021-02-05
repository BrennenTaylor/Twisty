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

#include "ExperimentRunner.h"
#include "Range.h"
#include "PathWeightUtils.h"
#include "PerturbUtils.h"

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <curand.h>
#include <curand_kernel.h>

#include <optional>
#include <random>

// Currently implemented as #defines as there is no good way that I can find to make these a class specific constant.
// TODO: Keep looking, there has to be something better
#define MaxDoubleLog10 300
#define MaxNumberOfPathsLog10 6.0
#define MaxNumberOfPaths 1000000

namespace twisty
{
    /**
     * @brief Implements the ExperimentRunner interface for running the experiment on the CPU
     *
     */

    class FullExperimentRunnerOptimalPerturbOptimized_GPU : public ExperimentRunner
    {
    public:
        struct BoundaryConditions_CUDA
        {
            float m_startPos[3] = { 0.0f, 0.0f, 0.0f };
            float m_startDir[3] = { 1.0f, 0.0f, 0.0f };
            float m_endPos[3] = { 0.0f, 0.0f, 0.0f };
            float m_endDir[3] = { 1.0f, 0.0f, 0.0f };
            float arclength = 0.0f;
        };


        struct CombinedWeightValues_C
        {
            uint32_t m_numValues = 0;
            double m_runningTotal = 0.0;
            double m_offset = 0.0;
            double m_maxWeightLog10 = 0.0;
            double m_maxPossibleFinalWeightLog10 = 0.0;
        };

        __host__ __device__ void static CombinedWeightValues_C_Reset(CombinedWeightValues_C* pData)
        {
            pData->m_numValues = 0;
            pData->m_runningTotal = 0.0;
            pData->m_offset = 0.0;
            pData->m_maxWeightLog10 = 0.0;
            pData->m_maxPossibleFinalWeightLog10 = 0.0;
        }

        __host__ __device__ void static CombinedWeightValues_C_AddValue(CombinedWeightValues_C* pData, double valueLog10)
        {
            // In the case we haven't added a value yet, we can early out
            if (pData->m_numValues == 0)
            {
                pData->m_maxWeightLog10 = valueLog10;
                pData->m_maxPossibleFinalWeightLog10 = pData->m_maxWeightLog10 + MaxNumberOfPathsLog10;
                pData->m_offset = MaxDoubleLog10 - pData->m_maxPossibleFinalWeightLog10;
                pData->m_runningTotal += pow(10.0, (valueLog10 + pData->m_offset));
                pData->m_numValues++;
                return;
            }


            // If we already have a value and its not larger than the current max, then throw it in.
            if (pData->m_maxWeightLog10 > valueLog10)
            {
                pData->m_runningTotal += pow(10.0, (valueLog10 + pData->m_offset));
                pData->m_numValues++;
                return;
            }

            // If it is larger, we need to rescale everything around that new value

            // New difference
            double newMaxPossibleFinalWeightLog10 = valueLog10 + MaxNumberOfPathsLog10;
            double newOffset = MaxDoubleLog10 - newMaxPossibleFinalWeightLog10;

            double offsetDelta = newOffset - pData->m_offset;
            double log10RunningTotal = log10(pData->m_runningTotal);
            double adjustedLog10RunningTotal = log10RunningTotal + offsetDelta;
            pData->m_runningTotal = pow(10.0, adjustedLog10RunningTotal);

            // Update
            pData->m_maxWeightLog10 = valueLog10;
            pData->m_maxPossibleFinalWeightLog10 = newMaxPossibleFinalWeightLog10;
            pData->m_offset = newOffset;

            pData->m_runningTotal += pow(10.0, (valueLog10 + pData->m_offset));
            pData->m_numValues++;
        }

        // Responsable for storing up to 10^6 big float values as double internally, while maintaining a significant amount of precision
        class CombinedWeightValues
        {

        public:
            __host__ __device__ void AddValue(double valueLog10)
            {
                CombinedWeightValues_C_AddValue(&m_data, valueLog10);
            }

            boost::multiprecision::cpp_dec_float_100 ExtractFinalValue()
            {
                // TODO: Do we need this, or can we simply compute using running total and offset?
                if (m_data.m_numValues == 0)
                {
                    return 0.0;
                }

                boost::multiprecision::cpp_dec_float_100 runningTotalLog10 = std::log10(m_data.m_runningTotal);
                runningTotalLog10 -= m_data.m_offset;
                return boost::multiprecision::pow(10.0, runningTotalLog10);
            }

        public:
            CombinedWeightValues_C m_data;
        };


    public:

        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         * @param kdsRange Range of allowed curvature * ds values
         * @param tdsRange Range of allowed torsion * ds values
         */
        FullExperimentRunnerOptimalPerturbOptimized_GPU(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~FullExperimentRunnerOptimalPerturbOptimized_GPU();

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
            int64_t numCombinedWeightValuesTotal,
            int64_t numCombinedWeightValuesPerThread,
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
            std::vector<CombinedWeightValues>& combinedWeightValues,
            std::vector<double>& cachedSegmentWeights,
            float segmentLength,
            const twisty::PathWeighting::WeightLookupTableIntegral& weightingIntegral,
            const twisty::PerturbUtils::BoundrayConditions& boundaryConditions,
            const PathWeighting::NormalizerStuff::FN& fn
        );

    private:
        bool SetupCudaDevice();
        void CleanupCudaDevice();

        bool SetupCuRandStates(uint32_t numGlobalPerturbThreads, uint32_t seed);
        void CleanupCudaRandStates();

        bool SetupCudaPerturb(uint32_t numGlobalPerturbThreads, uint32_t numCombinedWeightValues);
        void CleanupCudaPerturb();


    private:
        std::unique_ptr<PathWeighting::RegularizedIntegral> m_upRegIntEvaluator;

        // CUDA STUFF
    private:
        int32_t m_numSMs = 0;
        int32_t m_warpSize = 0;
        int32_t m_maxThreadsPerMultiprocessor = 0;

        uint64_t m_usedDeviceMemoryInBytes = 0;

        curandState_t* m_pPerGlobalThreadRandStates = nullptr;
        
        
        float* m_pPerGlobalThreadLeftScratchSpacePositions = nullptr;
        float* m_pPerGlobalThreadRightScratchSpacePositions = nullptr;
        float* m_pPerGlobalThreadWorkingScratchSpacePositions = nullptr;
        
        float* m_pPerGlobalThreadLeftScratchSpaceTangents = nullptr;
        float* m_pPerGlobalThreadRightScratchSpaceTangents = nullptr;
        float* m_pPerGlobalThreadWorkingScratchSpaceTangents = nullptr;

        float* m_pPerGlobalThreadLeftScratchSpaceCurvatures = nullptr;
        float* m_pPerGlobalThreadRightScratchSpaceCurvatures = nullptr;
        float* m_pPerGlobalThreadWorkingScratchSpaceCurvatures = nullptr;

        CombinedWeightValues_C* m_pPerThreadCombinedWeightValues = nullptr;
        CombinedWeightValues_C* m_pFinalCombinedValues = nullptr;
    };

    __global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState(uint32_t seed, curandState_t* pStates, uint32_t maxNumStates);
    __global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryPerturbKernel(
        int64_t numCombinedWeightValuesTotal,
        int64_t numCombinedWeightValuesPerWarp,
        int64_t numCombinedWeightValuesPerThread,
        int64_t numPathsToSkipPerThread,
        int64_t numSegmentsPerCurve,
        curandState_t* pCurandStates,
        float* pPerGlobalThreadLeftScratchSpacePositions,
        float* pPerGlobalThreadRightScratchSpacePositions,
        float* pPerGlobalThreadWorkingScratchSpacePositions,
        float* pPerGlobalThreadLeftScratchSpaceTangents, 
        float* pPerGlobalThreadRightScratchSpaceTangents,
        float* pPerGlobalThreadWorkingScratchSpaceTangents,
        float* pPerGlobalThreadLeftScratchSpaceCurvatures,
        float* pPerGlobalThreadRightScratchSpaceCurvatures,
        float* pPerGlobalThreadWorkingScratchSpaceCurvatures,
        FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C* pPerThreadCombinedWeightValues,
        FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C* pFinalCombinedValues,
        //float segmentLength,
        //const twisty::PathWeighting::WeightLookupTableIntegral& weightingIntegral,
        const twisty::PerturbUtils::BoundrayConditions& boundaryConditions,
        double* pLookupTable
        //const PathWeighting::NormalizerStuff::FN& fn
    );
}