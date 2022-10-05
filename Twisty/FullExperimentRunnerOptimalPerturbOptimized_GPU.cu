#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"

// Include these specifically to have all localized linking
#include "CombinedWeightUtils.cpp"
#include "CurvePerturbUtils.cpp"
#include "PathWeighters.cpp"

#include <boost\multiprecision\cpp_dec_float.hpp>

#include "CombinedWeightUtils.h"
#include "CurvePerturbUtils.h"
#include "CurveUtils.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "PathWeighters.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include <curand.h>

#include <omp.h>

#include <assert.h>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <limits>

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdlib.h>
#include <memory>

/*
      Notes, the GPU should not involve numbers larger than 4 billion
      as this is the maximum size that can fit in a unsigned 32 bit in
*/

#define PerturbGridSize 32 * 4
#define PerturbBlockSize 64
#define PathsPerGPUThread (MaxNumPathsPerCombinedWeight / PerturbBlockSize)

namespace twisty {
static void CudaSafeErrorCheck(cudaError_t error, std::string message)
{
    if (error != cudaSuccess) {
        std::string errorString(cudaGetErrorString(error));
        fprintf(stdout, "ERROR: %s : %s\n", message.c_str(), errorString.c_str());
    }
}

template<class T>
class ManagedPinnedMemory {
   public:
    ManagedPinnedMemory(size_t numType) { Alloc(numType); }

    void Alloc(size_t numType)
    {
        if (m_pData) {
            Free(m_pData);
        }
        m_numElements = numType;
        cudaMallocHost(&m_pData, numType * sizeof(T), 0);
    }

    ~ManagedPinnedMemory() { cudaFreeHost(m_pData); }

    T *Data() const { return m_pData; }
    size_t NumElements() { return m_numElements; }
    size_t NumBytes() { return m_numElements * sizeof(T); }

   private:
    void Free(void *ptr) { cudaFreeHost(ptr); }

   private:
    T *m_pData = nullptr;
    size_t m_numElements = 0;
};

__global__ void __launch_bounds__(PerturbBlockSize, PerturbGridSize)
      FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_RadiativeTransfer(
            const uint32_t numCombinedWeightValuesPerWarp,
            const uint32_t numSegmentsPerCurve,
            curandState_t *const pCurandStates,
            float *const pPerGlobalThreadScratchSpacePositions,
            float *const pPerGlobalThreadScratchSpaceTangents,
            float *const pPerGlobalThreadScratchSpaceCurvatures,
            CombinedWeightValues_C *const pFinalCombinedValues,
            const twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions,
            const float *const pWeightLookupTable,
            const uint32_t weightLookupTableSize,
            const float ds,
            const float minCurvature,
            const float maxCurvature,
            const float curvatureStepSize);

// __global__ void __launch_bounds__(PerturbBlockSize, PerturbGridSize)
//       FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_SimplifiedModel(
//             const int64_t numCombinedWeightValuesTotal,
//             const int64_t numCombinedWeightValuesPerWarp,
//             const uint32_t numSegmentsPerCurve,
//             curandState_t *const pCurandStates,
//             float *const pPerGlobalThreadScratchSpacePositions,
//             float *const pPerGlobalThreadScratchSpaceTangents,
//             float *const pPerGlobalThreadScratchSpaceCurvatures,
//             CombinedWeightValues_C *const pFinalCombinedValues,
//             const twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions,
//             const float *const pWeightLookupTable,
//             const uint32_t weightLookupTableSize,
//             const float ds,
//             const float minCurvature,
//             const float maxCurvature,
//             const float curvatureStepSize);

FullExperimentRunnerOptimalPerturbOptimized_GPU::FullExperimentRunnerOptimalPerturbOptimized_GPU(
      ExperimentRunner::ExperimentParameters &experimentParams, Bootstrapper &bootstrapper)
    : ExperimentRunner(experimentParams, bootstrapper)
{
}

FullExperimentRunnerOptimalPerturbOptimized_GPU::~FullExperimentRunnerOptimalPerturbOptimized_GPU()
{
}

ExperimentRunner::RunnerSpecificResults
FullExperimentRunnerOptimalPerturbOptimized_GPU::RunnerSpecificRunExperiment()
{
    /* --------------------- */
    auto setupTimeStart = std::chrono::high_resolution_clock::now();
    /* --------------------- */

    std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;
    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::SimplifiedModel) {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_ds);
    } else {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_ds);
    }
    lookupEvaluator->ExportValues(m_experimentDirPath.string());

    const twisty::PerturbUtils::BoundaryConditions boundaryConditions
          = m_upInitialCurve->GetBoundaryConditions();

    bool result = SetupCudaDevice();
    if (!result) {
        printf("Failed to setup cuda device\n");
        return {};
    }

    // Calculate number of paths needed to generate
    // Each warp is responsible for calculating a combined weight at a time
    const uint64_t numGlobalPerturbThreads = PerturbGridSize * PerturbBlockSize;

    const uint32_t numDispatchedWarps = PerturbGridSize;
    const uint64_t numPathsPerFullGrid
          = (uint64_t)numDispatchedWarps * MaxNumPathsPerCombinedWeight;

    const uint32_t numCombinedWeightValuesPerBlock
          = (m_experimentParams.numPathsInExperiment + numPathsPerFullGrid - 1)
          / numPathsPerFullGrid;
    const uint64_t numCombinedWeightValuesTotal
          = (uint64_t)numCombinedWeightValuesPerBlock * (uint64_t)numDispatchedWarps;

    const uint64_t actualNumberOfPaths
          = (uint64_t)numCombinedWeightValuesTotal * MaxNumPathsPerCombinedWeight;

    std::cout << "Requested number of paths: " << m_experimentParams.numPathsInExperiment
              << std::endl;
    std::cout << "Actual number of paths: " << actualNumberOfPaths << std::endl;
    std::cout << "Number of added paths: "
              << actualNumberOfPaths - m_experimentParams.numPathsInExperiment << std::endl;
    std::cout << "Num Global Perturb Threads: " << numGlobalPerturbThreads << std::endl;
    std::cout << "Num Combined Weight Values Total: " << numCombinedWeightValuesTotal << std::endl;
    std::cout << "Num Combined weights per warp: " << numCombinedWeightValuesPerBlock << std::endl;
    std::cout << "Perturb Grid Size required: " << PerturbGridSize << std::endl;
    std::cout << "Perturb Block Size required: " << PerturbBlockSize << std::endl;

    int32_t seed = m_experimentParams.curvePurturbSeed;
    if (seed == 0) {
        seed = time(0);
    }

    auto setupCuRandTimeStart = std::chrono::high_resolution_clock::now();
    {
        result = SetupCuRandStates(numGlobalPerturbThreads, seed);
        if (!result) {
            printf("Failed to setup CuRand states\n");
            return {};
        }
    }
    auto setupCuRandTimeEnd = std::chrono::high_resolution_clock::now();

    auto setupCudaPerturbStart = std::chrono::high_resolution_clock::now();
    {
        result = SetupCudaPerturb(numGlobalPerturbThreads, numCombinedWeightValuesTotal,
              m_experimentParams.numSegmentsPerCurve, lookupEvaluator->AccessLookupTable());
        if (!result) {
            printf("Failed to setup Cuda Perturb\n");
            return {};
        }
    }
    auto setupCudaPerturbEnd = std::chrono::high_resolution_clock::now();

    auto experimentTimeStart = std::chrono::high_resolution_clock::now();

    std::stringstream fnFilenameSS;
    fnFilenameSS << "SavedFN";
    fnFilenameSS << m_experimentParams.numSegmentsPerCurve;
    fnFilenameSS << ".fnd";
    const std::filesystem::path fnFilePath = std::filesystem::current_path() / fnFilenameSS.str();
    std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> upFN = nullptr;

    const PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer
          = (m_experimentParams.weightingParameters.weightingMethod
                  != WeightingMethod::RadiativeTransfer)
          ? 1.0
          : PathWeighting::NormalizerStuff::Norm(
                m_upInitialCurve->m_numSegments, m_upInitialCurve->m_ds, boundaryConditions);
    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    auto setupTimeEnd = std::chrono::high_resolution_clock::now();
    /* --------------------- */


    /* --------------------- */

    uint64_t perturbTimeCount = 0;
    uint64_t weightCalcTimeCount = 0;

    std::cout << "numPathsInExperiment specified: " << m_experimentParams.numPathsInExperiment
              << std::endl;

    std::cout << "numPathsInExperiment generated: "
              << numCombinedWeightValuesTotal * MaxNumPathsPerCombinedWeight << std::endl;
    std::cout << "numCombinedWeightValuesTotal: " << numCombinedWeightValuesTotal << std::endl;
    std::cout << "numCombinedWeightValuesPerWarp: " << numCombinedWeightValuesPerBlock << std::endl;
    std::cout << "numPathsPerThread: " << PathsPerGPUThread << std::endl;

    auto perturbTimeStart = std::chrono::high_resolution_clock::now();

    twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions;
    csBoundaryConditions.m_startPos[0] = boundaryConditions.m_startPos.x;
    csBoundaryConditions.m_startPos[1] = boundaryConditions.m_startPos.y;
    csBoundaryConditions.m_startPos[2] = boundaryConditions.m_startPos.z;

    csBoundaryConditions.m_startDir[0] = boundaryConditions.m_startDir.x;
    csBoundaryConditions.m_startDir[1] = boundaryConditions.m_startDir.y;
    csBoundaryConditions.m_startDir[2] = boundaryConditions.m_startDir.z;

    csBoundaryConditions.m_endPos[0] = boundaryConditions.m_endPos.x;
    csBoundaryConditions.m_endPos[1] = boundaryConditions.m_endPos.y;
    csBoundaryConditions.m_endPos[2] = boundaryConditions.m_endPos.z;

    csBoundaryConditions.m_endDir[0] = boundaryConditions.m_endDir.x;
    csBoundaryConditions.m_endDir[1] = boundaryConditions.m_endDir.y;
    csBoundaryConditions.m_endDir[2] = boundaryConditions.m_endDir.z;

    csBoundaryConditions.arclength = boundaryConditions.arclength;

    {
        dim3 gridSize(PerturbGridSize, 1, 1);
        dim3 blockSize(PerturbBlockSize, 1, 1);
        size_t sharedMemorySizeBytes = 0;
        cudaStream_t stream = 0;

        std::cout << "Dispatching with: " << std::endl;
        std::cout << "\tGrid Size: " << PerturbGridSize << std::endl;
        std::cout << "\tBlock Size: " << PerturbBlockSize << std::endl;
        std::cout << "Weight Table Size: " << lookupEvaluator->AccessLookupTable().size()
                  << std::endl;
        std::cout << "DS: " << lookupEvaluator->GetDs() << std::endl;
        std::cout << "Min Curvature: " << lookupEvaluator->GetMinCurvature() << std::endl;
        std::cout << "Max Curvature: " << lookupEvaluator->GetMaxCurvature() << std::endl;
        std::cout << "Curvature Step Size: " << lookupEvaluator->GetCurvatureStepSize()
                  << std::endl;

        if (m_experimentParams.weightingParameters.weightingMethod
              == WeightingMethod::RadiativeTransfer) {
            FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_RadiativeTransfer<<<
                  gridSize, blockSize, sharedMemorySizeBytes, stream>>>(
                  numCombinedWeightValuesPerBlock, m_experimentParams.numSegmentsPerCurve,
                  m_pPerGlobalThreadRandStates, m_pPerGlobalThreadScratchSpacePositions,
                  m_pPerGlobalThreadScratchSpaceTangents, m_pPerGlobalThreadScratchSpaceCurvatures,
                  m_pFinalCombinedValues, csBoundaryConditions, m_pDeviceWeightLookupTable,
                  lookupEvaluator->AccessLookupTable().size(), lookupEvaluator->GetDs(),
                  lookupEvaluator->GetMinCurvature(), lookupEvaluator->GetMaxCurvature(),
                  lookupEvaluator->GetCurvatureStepSize());

            CudaSafeErrorCheck(cudaGetLastError(),
                  "FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_"
                  "RadiativeTransfer kernal launch");
            CudaSafeErrorCheck(cudaDeviceSynchronize(),
                  "FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_"
                  "RadiativeTransfer kernel sync");
        } else {
            std::cout << "Currently not implemented" << std::endl;
            // FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_SimplifiedModel<<<
            //       gridSize, blockSize, sharedMemorySizeBytes, stream>>>(
            //       numCombinedWeightValuesTotal, numCombinedWeightValuesPerWarp,
            // , m_experimentParams.numSegmentsPerCurve,
            //       m_pPerGlobalThreadRandStates, m_pPerGlobalThreadScratchSpacePositions,
            //       m_pPerGlobalThreadScratchSpaceTangents, m_pPerGlobalThreadScratchSpaceCurvatures,
            //       m_pFinalCombinedValues, csBoundaryConditions, m_pDeviceWeightLookupTable,
            //       lookupEvaluator->AccessLookupTable().size(), lookupEvaluator->GetDs(),
            //       lookupEvaluator->GetMinCurvature(), lookupEvaluator->GetMaxCurvature(),
            //       lookupEvaluator->GetCurvatureStepSize());

            // CudaSafeErrorCheck(cudaGetLastError(),
            //       "FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_"
            //       "SimplifiedModel kernal launch");
            // CudaSafeErrorCheck(cudaDeviceSynchronize(),
            //       "FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_"
            //       "SimplifiedModel kernel sync");
        }
    }

    auto perturbTimeEnd = std::chrono::high_resolution_clock::now();
    perturbTimeCount
          = std::chrono::duration_cast<std::chrono::milliseconds>(perturbTimeEnd - perturbTimeStart)
                  .count();

    // -------------------
    auto weightingTimeStart = std::chrono::high_resolution_clock::now();

    printf("Copying values back\n");
    // Copying back values
    std::vector<CombinedWeightValues_C> combinedWeightValues(numCombinedWeightValuesTotal);
    CudaSafeErrorCheck(cudaMemcpy(combinedWeightValues.data(), m_pFinalCombinedValues,
                             sizeof(CombinedWeightValues_C) * numCombinedWeightValuesTotal,
                             cudaMemcpyDeviceToHost),
          "Copy combined values back from GPU");

    printf("Done Copying values back\n");

    // We need to calculate the absorbtion/scattering piece
    boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;

    uint64_t numWeightsGenerated = 0;

    // No, we calculating the weighting
    std::cout << "Num combined weights: " << combinedWeightValues.size() << std::endl;
    for (auto &combinedWeightValue : combinedWeightValues) {
        const boost::multiprecision::cpp_dec_float_100 extractedDispatchWeight
              = ExtractFinalValue(combinedWeightValue);
        bigTotalExperimentWeight += extractedDispatchWeight;

        if (m_experimentParams.outputBigFloatWeights) {
            UpdateConvergenceWeight(
                  combinedWeightValue.m_numValues, extractedDispatchWeight * pathNormalizer);
        }

        numWeightsGenerated += combinedWeightValue.m_numValues;
    }
    bigTotalExperimentWeight *= pathNormalizer;
    auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
    weightCalcTimeCount = std::chrono::duration_cast<std::chrono::milliseconds>(
          weightingTimeEnd - weightingTimeStart)
                                .count();
    /* --------------------- */

    // Cleanup stuff

    {
        CleanupCudaPerturb();
        CleanupCudaRandStates();
        CleanupCudaDevice();
    }

    auto timeMs
          = std::chrono::duration_cast<std::chrono::milliseconds>(setupTimeEnd - setupTimeStart);

    ExperimentResults results;
    results.experimentWeight = bigTotalExperimentWeight;
    results.totalPathsGenerated = numWeightsGenerated;
    results.numFailedPaths = 0;

    ExperimentRunner::RunnerSpecificResults specificResult;
    specificResult.experimentResults = std::make_optional<ExperimentResults>(results);
    specificResult.setupMs = timeMs.count();
    specificResult.runExperimentMs = perturbTimeCount;
    specificResult.weightingMs = weightCalcTimeCount;

    return specificResult;
}

bool FullExperimentRunnerOptimalPerturbOptimized_GPU::SetupCudaDevice()
{
    // Force the creation of the context
    cudaFree(0);

    // Query for the number of devices avalible
    int32_t numDevices = 0;
    CudaSafeErrorCheck(cudaGetDeviceCount(&numDevices), "Get Devices");

    // We need at least one cuda device
    if (numDevices <= 0) {
        printf("No CUDA device avalible. Cannot execute experiment.\n");
        return false;
    }

    // Iterate over all devices and report the device stats.
    for (int32_t i = 0; i < numDevices; ++i) {
        cudaDeviceProp prop;
        CudaSafeErrorCheck(cudaGetDeviceProperties(&prop, i), "Get Device Prop");
        std::cout << "\nDevice Number: " << i << std::endl;
        std::cout << "\tDevice name: " << prop.name << std::endl;
        std::cout << "\tDevice Capability: " << prop.major << ", " << prop.minor << std::endl;

        std::cout << "\tSM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "\tMax Blocks Per SM: " << prop.maxBlocksPerMultiProcessor << std::endl;
        std::cout << "\tMax Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "\tMax Threads Per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "\tRegisters per SM: " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "\tRegisters per Block: " << prop.regsPerBlock << std::endl;
        std::cout << "\tShared Memory per SM: " << prop.sharedMemPerMultiprocessor << std::endl;
        std::cout << "\tShared Memory per Block: " << prop.sharedMemPerBlock << std::endl;

        std::cout << "\tSingle to Double Perf Ratio: " << prop.singleToDoublePrecisionPerfRatio
                  << std::endl;

        std::cout << "\tSM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "\tSM Shared Memory: " << prop.sharedMemPerBlock << std::endl;
        std::cout << "\tWarp Size: " << prop.warpSize << std::endl;
        std::cout << "\tThreads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "\tPeak Memory Bandwidth (GB/s): "
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
        std::cout << "\tGlobal Memory: " << prop.totalGlobalMem << std::endl;
        std::cout << "\tConst Memory: " << prop.totalConstMem << std::endl;

        std::cout << "\tFeatures Supported: " << std::endl;
        std::cout << "\t\tCompute Preemption Supported: " << prop.computePreemptionSupported
                  << std::endl;
        std::cout << "\t\tConcurrent Kernels Supported: " << prop.concurrentKernels << std::endl;
        std::cout << "\t\tGlobal L1 Cache Support: " << prop.globalL1CacheSupported << std::endl;
        std::cout << "\t\tLocal L1 Cache Support: " << prop.localL1CacheSupported << std::endl;
        std::cout << "\t\tKernel Exec Timeout Enabled: " << prop.kernelExecTimeoutEnabled
                  << std::endl;
    }

    // We select the first device only
    const uint32_t selectedDeviceIdx = 0;
    std::cout << "\nSelected device: " << selectedDeviceIdx << std::endl;
    cudaDeviceProp deviceProp;
    CudaSafeErrorCheck(cudaGetDeviceProperties(&deviceProp, 0), "Get first device prop");

    m_deviceProp = deviceProp;
    return true;
}

void FullExperimentRunnerOptimalPerturbOptimized_GPU::CleanupCudaDevice() { }

bool FullExperimentRunnerOptimalPerturbOptimized_GPU::SetupCuRandStates(
      int32_t numGlobalPerturbThreads, uint32_t seed)
{
    std::cout << "Setup Cuda Perturb: " << std::endl;
    uint64_t usedMemoryInBytes = 0;

    // Random Seed Kernel
    // Every block thread needs its own curand state
    CudaSafeErrorCheck(cudaMalloc((void **)&m_pPerGlobalThreadRandStates,
                             sizeof(curandState_t) * numGlobalPerturbThreads),
          "RandState malloc");
    usedMemoryInBytes += (sizeof(curandState_t) * numGlobalPerturbThreads);

    int blockSizeRandKernel = 0;
    int minGridSizeRandKernel = 0;
    size_t sharedMemoryUse = 0;
    size_t maxBlockSize = 0;
    CudaSafeErrorCheck(
          cudaOccupancyMaxPotentialBlockSize(&minGridSizeRandKernel, &blockSizeRandKernel,
                FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState,
                sharedMemoryUse, maxBlockSize),
          "Failed to calculating occupancy for InitializeCuRandState kernel");
    std::cout << "\nInitializeCurandState: " << std::endl;
    std::cout << "\tBlock Size: " << blockSizeRandKernel << std::endl;
    std::cout << "\tMin Grid Size: " << minGridSizeRandKernel << std::endl;

    size_t gridSizeRandKernel
          = ((numGlobalPerturbThreads) + blockSizeRandKernel - 1) / blockSizeRandKernel;

    printf("\tInitializeCurandState Grid Size: %d\n", gridSizeRandKernel);
    printf("\tInitializeCurandState Block Size: %d\n", blockSizeRandKernel);

    // Dispatch CurandState
    // We need a dispatch that initializes curand per thread
    {
        dim3 gridSize(gridSizeRandKernel, 1, 1);
        dim3 blockSize(blockSizeRandKernel, 1, 1);
        size_t sharedMemorySizeBytes = 0;
        cudaStream_t stream = 0;

        FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState<<<gridSize, blockSize,
              sharedMemorySizeBytes, stream>>>(
              static_cast<uint32_t>(seed), m_pPerGlobalThreadRandStates, numGlobalPerturbThreads);

        CudaSafeErrorCheck(cudaGetLastError(), "Rand state init kernal launch");
        CudaSafeErrorCheck(cudaDeviceSynchronize(), "Rand state kernel sync");
    }


    std::cout << "\tUsed Device Memory Before: " << m_usedDeviceMemoryInBytes << std::endl;
    std::cout << "\tNewly allocated memory: " << usedMemoryInBytes << std::endl;

    m_usedDeviceMemoryInBytes += usedMemoryInBytes;

    std::cout << "\tUsed Device Memory After: " << m_usedDeviceMemoryInBytes << std::endl;

    return true;
}

void FullExperimentRunnerOptimalPerturbOptimized_GPU::CleanupCudaRandStates()
{
    CudaSafeErrorCheck(cudaFree((void *)m_pPerGlobalThreadRandStates), "RandState free");
}


// Pass in total number of threads that can be used, as well as the number of batches of 10^6 paths which will be generated
bool FullExperimentRunnerOptimalPerturbOptimized_GPU::SetupCudaPerturb(
      uint64_t numGlobalPerturbThreads, uint64_t numCombinedWeightValues, uint32_t numSegments,
      const std::vector<float> &weightTable)
{
    std::cout << "Setup Cuda Perturb: " << std::endl;
    uint64_t usedMemoryInBytes = 0;

    // Every global thread needs its own curve scratch space
    const uint64_t positionBytes = numGlobalPerturbThreads * (numSegments + 1) * sizeof(float) * 3;
    CudaSafeErrorCheck(cudaMalloc((void **)&m_pPerGlobalThreadScratchSpacePositions, positionBytes),
          "Cuda malloc Scratch Space Positions");
    usedMemoryInBytes += positionBytes;

    // Every global thread needs its own curve scratch space left and right and working
    const uint64_t tangentBytes = numGlobalPerturbThreads * numSegments * sizeof(float) * 3;
    CudaSafeErrorCheck(cudaMalloc((void **)&m_pPerGlobalThreadScratchSpaceTangents, tangentBytes),
          "Cuda malloc Scratch Space Tangents");
    usedMemoryInBytes += tangentBytes;

    // Every global thread needs its own curve scratch space left and right and working
    const uint64_t curvatureBytes = numGlobalPerturbThreads * (numSegments - 1) * sizeof(float);
    CudaSafeErrorCheck(
          cudaMalloc((void **)&m_pPerGlobalThreadScratchSpaceCurvatures, curvatureBytes),
          "Cuda malloc Scratch Space Curvatures");
    usedMemoryInBytes += curvatureBytes;

    const uint64_t combinedValueBytes = sizeof(CombinedWeightValues_C) * numCombinedWeightValues;
    CudaSafeErrorCheck(cudaMalloc((void **)&m_pFinalCombinedValues, combinedValueBytes),
          "Cuda malloc combined weight values per thread");
    usedMemoryInBytes += combinedValueBytes;

    const uint64_t weightTableBytes = sizeof(float) * weightTable.size();
    CudaSafeErrorCheck(cudaMalloc((void **)&m_pDeviceWeightLookupTable, weightTableBytes),
          "Cuda malloc combined weight values per thread");
    usedMemoryInBytes += weightTableBytes;

    std::cout << "\tUsed Device Memory Before: " << m_usedDeviceMemoryInBytes << std::endl;
    std::cout << "\tNewly allocated memory: " << usedMemoryInBytes << std::endl;

    m_usedDeviceMemoryInBytes += usedMemoryInBytes;

    std::cout << "\tUsed Device Memory After: " << m_usedDeviceMemoryInBytes << std::endl;

    // Copy that data over to the gpu

    // Setup data structures
    twisty::PerturbUtils::BoundaryConditions_CudaSafe boundaryConditionsCudaSafe
          = m_upInitialCurve->GetBoundaryConditionsCudaSafe();


    // These are page locked since we use: cudaMallocHost
    // This is a speed boost as these MUST be present in ram and cant paged
    // off to a drive

    ManagedPinnedMemory<float> managedInitialCurvePositions((numSegments + 1) * 3);
    ManagedPinnedMemory<float> managedInitialCurveTangents(numSegments * 3);
    ManagedPinnedMemory<float> managedInitialCurveCurvatures((numSegments - 1));

    memcpy(managedInitialCurvePositions.Data(), (float *)m_upInitialCurve->m_positions.data(),
          managedInitialCurvePositions.NumBytes());

    // Update and curvature
    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::RadiativeTransfer) {
        twisty::PerturbUtils::UpdateTangentsFromPos_CudaSafe(managedInitialCurvePositions.Data(),
              managedInitialCurveTangents.Data(), m_upInitialCurve->m_numSegments,
              boundaryConditionsCudaSafe);
        twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer_CudaSafe(
              managedInitialCurveTangents.Data(), managedInitialCurveCurvatures.Data(),
              m_upInitialCurve->m_numSegments, boundaryConditionsCudaSafe);
    } else {
        twisty::PerturbUtils::UpdateTangentsFromPos_CudaSafe(managedInitialCurvePositions.Data(),
              managedInitialCurveTangents.Data(), m_upInitialCurve->m_numSegments,
              boundaryConditionsCudaSafe);
        twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel_CudaSafe(
              managedInitialCurveTangents.Data(), managedInitialCurveCurvatures.Data(),
              m_upInitialCurve->m_numSegments, boundaryConditionsCudaSafe);
    }

    // TODO: Should this be intermixed somehow for better performance?
    std::cout << "Copying over intial curves" << std::endl;
    std::cout << "\tNum Global Perturb Threads: " << numGlobalPerturbThreads << std::endl;

    {
        uint64_t idx = 0;
        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx) {
            CudaSafeErrorCheck(cudaMemcpy((void *)&(m_pPerGlobalThreadScratchSpacePositions[idx]),
                                     (void *)managedInitialCurvePositions.Data(),
                                     managedInitialCurvePositions.NumBytes(),
                                     cudaMemcpyHostToDevice),
                  "Copy inital positions to per thread scratch space");
            idx += (numSegments + 1) * 3;
        }
    }

    {
        uint64_t idx = 0;
        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx) {
            CudaSafeErrorCheck(cudaMemcpy((void *)&(m_pPerGlobalThreadScratchSpaceTangents[idx]),
                                     (void *)managedInitialCurveTangents.Data(),
                                     managedInitialCurveTangents.NumBytes(),
                                     cudaMemcpyHostToDevice),
                  "Copy inital tangents to per thread scratch space");
            idx += numSegments * 3;
        }
    }

    {
        uint64_t idx = 0;
        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx) {
            CudaSafeErrorCheck(cudaMemcpy((void *)&(m_pPerGlobalThreadScratchSpaceCurvatures[idx]),
                                     (void *)managedInitialCurveCurvatures.Data(),
                                     managedInitialCurveTangents.NumBytes(),
                                     cudaMemcpyHostToDevice),
                  "Copy inital curvatures to per thread scratch space");
            idx += (numSegments - 1);
        }
    }

    // TODO: Make these paged memory for faster potentially transfer!
    std::vector<CombinedWeightValues_C> finalCombinedWeights(numCombinedWeightValues);
    for (int i = 0; i < finalCombinedWeights.size(); i++) {
        CombinedWeightValues_C_Reset(finalCombinedWeights[i]);
    }
    cudaMemcpy((void *)m_pFinalCombinedValues, (void *)finalCombinedWeights.data(),
          finalCombinedWeights.size() * sizeof(CombinedWeightValues_C), cudaMemcpyHostToDevice);

    cudaMemcpy((void *)m_pDeviceWeightLookupTable, (void *)weightTable.data(),
          weightTable.size() * sizeof(float), cudaMemcpyHostToDevice);

    return true;
}

void FullExperimentRunnerOptimalPerturbOptimized_GPU::CleanupCudaPerturb()
{
    CudaSafeErrorCheck(cudaFree((void *)m_pFinalCombinedValues),
          "Cuda free combined weight values for final answer");

    CudaSafeErrorCheck(
          cudaFree((void *)m_pDeviceWeightLookupTable), "Cuda free device weight table");

    CudaSafeErrorCheck(cudaFree((void *)m_pPerGlobalThreadScratchSpaceCurvatures),
          "Cuda free Left Scratch Space Curvatures");

    CudaSafeErrorCheck(cudaFree((void *)m_pPerGlobalThreadScratchSpaceTangents),
          "Cuda free Left Scratch Space Tangents");

    CudaSafeErrorCheck(cudaFree((void *)m_pPerGlobalThreadScratchSpacePositions),
          "Cuda free Left Scratch Space Positions");
}

__global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState(
      uint32_t seed, curandState_t *pStates, uint32_t maxNumStates)
{
    // Unique index
    const uint32_t globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (globalThreadIdx < maxNumStates) {
        curand_init(seed + globalThreadIdx, 0, 0, &pStates[globalThreadIdx]);
    }
}


__global__ void __launch_bounds__(PerturbBlockSize, PerturbGridSize)
      FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_RadiativeTransfer(
            const uint32_t numCombinedWeightValuesPerBlock,
            const uint32_t numSegmentsPerCurve,
            curandState_t *const pCurandStates,
            float *const pPerGlobalThreadScratchSpacePositions,
            float *const pPerGlobalThreadScratchSpaceTangents,
            float *const pPerGlobalThreadScratchSpaceCurvatures,
            CombinedWeightValues_C *const pFinalCombinedValues,
            const twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions,
            const float *const pWeightLookupTable,
            const uint32_t weightLookupTableSize,
            const float ds,
            const float minCurvature,
            const float maxCurvature,
            const float curvatureStepSize)
{
    __shared__ CombinedWeightValues_C perThreadWeightValues[PerturbBlockSize];

// Should be between 0 and max num threads - 1
#define GlobalThreadIdx (blockIdx.x * blockDim.x + threadIdx.x)

#define NumPosPerCurve (numSegmentsPerCurve + 1)
#define NumTanPerCurve (numSegmentsPerCurve)
#define NumCurvaturesPerCurve (numSegmentsPerCurve - 1)

#define CurrentThreadPosStartIdx (NumPosPerCurve * 3 * GlobalThreadIdx)
#define CurrentThreadTanStartIdx (NumTanPerCurve * 3 * GlobalThreadIdx)
#define CurrentThreadCurvatureStartIdx (NumCurvaturesPerCurve * GlobalThreadIdx)

    //     printf("Number of combined weight values per block: %d\n", numCombinedWeightValuesPerBlock);


    // Ok, we want to loop over the outer batches first, the number per warp
    for (uint32_t currentBlockCombinedWeightIdx = 0;
          currentBlockCombinedWeightIdx < numCombinedWeightValuesPerBlock;
          currentBlockCombinedWeightIdx++) {
        CombinedWeightValues_C_Reset(perThreadWeightValues[threadIdx.x]);

        // We want to stop generating in this case


        // We need a loop over the batches
        // TODO: Do we even need to do this?
        // Should we seperate into seperate step
        for (uint32_t pathCount = 0; pathCount < PathsPerGPUThread; pathCount++) {
            // Do our random rolls here
            float e0 = curand_uniform(&(pCurandStates[GlobalThreadIdx]));
            float e1 = curand_uniform(&(pCurandStates[GlobalThreadIdx]));
            float e2 = curand_uniform(&(pCurandStates[GlobalThreadIdx]));

            // Ok, now we first want to reset the combined weight stuff
            {
                // This is the perturbation piece.
                // Can we do this in place, most likely
                // This will modify pCurrentThreadCurve
                // Remember, the structure of this is:
                // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M
                {
                    // Should be 2 - 180
                    uint32_t maxDiff = min((uint32_t)(numSegmentsPerCurve - 2), 25);
                    uint32_t diff = floorf(e0 * (maxDiff - 2)) + 2;

                    // -2 from the -1 to offset for the +1, and -1 as required by index
                    uint32_t leftPointIndex = floorf(e1 * (numSegmentsPerCurve - diff - 2)) + 1;

                    uint32_t rightPointIndex = leftPointIndex + diff;

                    // We need two frames for each segment to get the new curvature and torsion.
                    // we need the frame left of the segment, as well as the frame right of the segment.
                    // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                    const float leftPoint_x
                          = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                + leftPointIndex * 3 + 0];
                    const float leftPoint_y
                          = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                + leftPointIndex * 3 + 1];
                    const float leftPoint_z
                          = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                + leftPointIndex * 3 + 2];

                    const float rightPoint_x
                          = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                + rightPointIndex * 3 + 0];
                    const float rightPoint_y
                          = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                + rightPointIndex * 3 + 1];
                    const float rightPoint_z
                          = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                + rightPointIndex * 3 + 2];


                    float N[3] = { rightPoint_x - leftPoint_x, rightPoint_y - leftPoint_y,
                        rightPoint_z - leftPoint_z };
                    float N_length_inv = 1.0f / sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
                    N[0] *= N_length_inv;
                    N[1] *= N_length_inv;
                    N[2] *= N_length_inv;

                    // Overwrite angle
                    const float randRotationAngleNeedsPi = (e2 * 2.0 - 1.0);

                    // Rotation
                    {
                        // Rotation by quaternion
                        {
                            const float sinRotAngle = sinpif(randRotationAngleNeedsPi / 2.0f);
                            float quaternionRotation[4] = { cospif(randRotationAngleNeedsPi / 2.0f),
                                N[0] * sinRotAngle, N[1] * sinRotAngle, N[2] * sinRotAngle };

                            for (uint32_t pointIdx = (leftPointIndex + 1);
                                  pointIdx < rightPointIndex;
                                  ++pointIdx) {
                                float *const currentPoint
                                      = &pPerGlobalThreadScratchSpacePositions
                                              [CurrentThreadPosStartIdx + pointIdx * 3];
                                currentPoint[0] -= leftPoint_x;
                                currentPoint[1] -= leftPoint_y;
                                currentPoint[2] -= leftPoint_z;

                                // Rotate and stuff back in shifted point
                                RotateVectorByQuaternion(
                                      (float *)quaternionRotation, (float *)currentPoint);
                                // Update the point with the rotated version
                                currentPoint[0] += leftPoint_x;
                                currentPoint[1] += leftPoint_y;
                                currentPoint[2] += leftPoint_z;
                            }
                        }

                        // Rotation by Matrix
                        // {
                        //     float rotationMatrix[9]
                        //           = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                        //     RotationMatrixAroundAxis_MultiplyAngleByPi_CudaSafe(
                        //           randRotationAngleNeedsPi, (float *)N, (float *)rotationMatrix);

                        //     for (uint32_t pointIdx = (leftPointIndex + 1);
                        //           pointIdx < rightPointIndex;
                        //           ++pointIdx) {
                        //         float shiftedPoint[3];
                        //         shiftedPoint[0]
                        //               = pPerGlobalThreadScratchSpacePositions
                        //                       [CurrentThreadPosStartIdx + pointIdx * 3 + 0]
                        //               - leftPoint_x;
                        //         shiftedPoint[1]
                        //               = pPerGlobalThreadScratchSpacePositions
                        //                       [CurrentThreadPosStartIdx + pointIdx * 3 + 1]
                        //               - leftPoint_y;
                        //         shiftedPoint[2]
                        //               = pPerGlobalThreadScratchSpacePositions
                        //                       [CurrentThreadPosStartIdx + pointIdx * 3 + 2]
                        //               - leftPoint_z;

                        //         // Rotate and stuff back in shifted point
                        //         RotateVectorByMatrix(
                        //               (float *)rotationMatrix, (float *)shiftedPoint);
                        //         // Update the point with the rotated version
                        //         pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                        //               + pointIdx * 3 + 0]
                        //               = shiftedPoint[0] + leftPoint_x;
                        //         pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                        //               + pointIdx * 3 + 1]
                        //               = shiftedPoint[1] + leftPoint_y;
                        //         pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                        //               + pointIdx * 3 + 2]
                        //               = shiftedPoint[2] + leftPoint_z;
                        //     }
                        // }

                        // Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                        // We can do a different approach later.
                        // Here, we want to do a perturb update call
                        twisty::PerturbUtils::UpdateTangentsFromPos_CudaSafe(
                              &(pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx]),
                              &(pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx]),
                              numSegmentsPerCurve, csBoundaryConditions);

                        twisty::PerturbUtils::
                              UpdateCurvaturesFromTangents_RadiativeTransfer_CudaSafe(
                                    &(pPerGlobalThreadScratchSpaceTangents
                                                [CurrentThreadTanStartIdx]),
                                    &(pPerGlobalThreadScratchSpaceCurvatures
                                                [CurrentThreadCurvatureStartIdx]),
                                    numSegmentsPerCurve, csBoundaryConditions);
                    }

                    double pathWeightLog10
                          = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CudaSafe(
                                &(pPerGlobalThreadScratchSpaceCurvatures
                                            [CurrentThreadCurvatureStartIdx]),
                                (numSegmentsPerCurve - 1), pWeightLookupTable,
                                weightLookupTableSize, ds, minCurvature, maxCurvature,
                                curvatureStepSize);


                    // Else, contribute to the paths
                    CombinedWeightValues_C_AddValue(
                          perThreadWeightValues[threadIdx.x], pathWeightLog10);
                }
            }
        }

        // First thread in warp responsible for combining all the weights into one
        __syncthreads();

        if (threadIdx.x == 0) {
            for (uint32_t warpThreadIdx = 1; warpThreadIdx < PerturbBlockSize; ++warpThreadIdx) {
                perThreadWeightValues[0] = CombinedWeightValues_C_CombineValues(
                      perThreadWeightValues[0], perThreadWeightValues[warpThreadIdx]);
            }

            // Finally, we write to the combined final values
            pFinalCombinedValues[blockIdx.x * numCombinedWeightValuesPerBlock
                  + currentBlockCombinedWeightIdx]
                  = perThreadWeightValues[0];
        }
        __syncthreads();
    }

// Should be between 0 and max num threads - 1
#undef GlobalThreadIdx
#undef NumPosPerCurve
#undef NumTanPerCurve
#undef NumCurvaturesPerCurve
#undef CurrentThreadPosStartIdx
#undef CurrentThreadTanStartIdx
#undef CurrentThreadCurvatureStartIdx
}

// __global__ void __launch_bounds__(PerturbBlockSize, PerturbGridSize)
//       FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel_SimplifiedModel(
//             const int64_t numCombinedWeightValuesTotal,
//             const int64_t numCombinedWeightValuesPerWarp,
//             const uint32_t numSegmentsPerCurve,
//             curandState_t *const pCurandStates,
//             float *const pPerGlobalThreadScratchSpacePositions,
//             float *const pPerGlobalThreadScratchSpaceTangents,
//             float *const pPerGlobalThreadScratchSpaceCurvatures,
//             CombinedWeightValues_C *const pFinalCombinedValues,
//             // const int32_t weightingMethod,
//             const twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions,
//             const float *const pWeightLookupTable,
//             const uint32_t weightLookupTableSize,
//             const float ds,
//             const float minCurvature,
//             const float maxCurvature,
//             const float curvatureStepSize)
// {
//     __shared__ CombinedWeightValues_C perThreadWeightValues[PerturbBlockSize];

// // Should be between 0 and max num threads - 1
// #define GlobalThreadIdx (blockIdx.x * blockDim.x + threadIdx.x)

// #define NumPosPerCurve (numSegmentsPerCurve + 1)
// #define NumTanPerCurve (numSegmentsPerCurve)
// #define NumCurvaturesPerCurve (numSegmentsPerCurve - 1)

// #define CurrentThreadPosStartIdx (NumPosPerCurve * 3 * GlobalThreadIdx)
// #define CurrentThreadTanStartIdx (NumTanPerCurve * 3 * GlobalThreadIdx)
// #define CurrentThreadCurvatureStartIdx (NumCurvaturesPerCurve * GlobalThreadIdx)

//     // Ok, we want to loop over the outer batches first, the number per warp
//     for (int64_t combinedWeightValuesWarpIdx = 0;
//           combinedWeightValuesWarpIdx < numCombinedWeightValuesPerWarp;
//           combinedWeightValuesWarpIdx++) {
//         CombinedWeightValues_C_Reset(perThreadWeightValues[threadIdx.x]);

//         // We want to stop generating in this case
//         if (combinedWeightValuesWarpIdx + numCombinedWeightValuesPerWarp * blockIdx.x
//               >= numCombinedWeightValuesTotal) {
//             continue;
//         }

//         // We need a loop over the batches
//         for (int64_t pathCount = 0; pathCount < PathsPerGPUThread; pathCount++) {
//             int64_t currentPathIdx = PathsPerGPUThread * threadIdx.x + pathCount;

//             // Do our random rolls here
//             float e0 = curand_uniform(&(pCurandStates[GlobalThreadIdx]));
//             float e1 = curand_uniform(&(pCurandStates[GlobalThreadIdx]));
//             float e2 = curand_uniform(&(pCurandStates[GlobalThreadIdx]));

//             // We can exit once this point is reached as we have generated all the paths necessary for this thread
//             if (currentPathIdx >= MaxNumPathsPerCombinedWeight) {
//                 printf("Breaking early as paths are done for this combined weight value\n");
//                 // We dont want to continue if we have already generated the correct number of paths.
//                 break;
//             }

//             // Ok, now we first want to reset the combined weight stuff
//             {
//                 // This is the perturbation piece.
//                 // Can we do this in place, most likely
//                 // This will modify pCurrentThreadCurve
//                 // Remember, the structure of this is:
//                 // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M
//                 {
//                     // Should be 2 - 180
//                     int64_t maxDiff = min((int)(numSegmentsPerCurve - 2), 25);
//                     int64_t diff = floorf(e0 * (maxDiff - 2)) + 2;

//                     // -2 from the -1 to offset for the +1, and -1 as required by index
//                     int64_t leftPointIndex = floorf(e1 * (numSegmentsPerCurve - diff - 2)) + 1;

//                     int64_t rightPointIndex = leftPointIndex + diff;

//                     // We need two frames for each segment to get the new curvature and torsion.
//                     // we need the frame left of the segment, as well as the frame right of the segment.
//                     // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
//                     const float leftPoint_x
//                           = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                 + leftPointIndex * 3 + 0];
//                     const float leftPoint_y
//                           = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                 + leftPointIndex * 3 + 1];
//                     const float leftPoint_z
//                           = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                 + leftPointIndex * 3 + 2];

//                     const float rightPoint_x
//                           = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                 + rightPointIndex * 3 + 0];
//                     const float rightPoint_y
//                           = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                 + rightPointIndex * 3 + 1];
//                     const float rightPoint_z
//                           = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                 + rightPointIndex * 3 + 2];


//                     float N[3] = { rightPoint_x - leftPoint_x, rightPoint_y - leftPoint_y,
//                         rightPoint_z - leftPoint_z };
//                     volatile float N_length = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
//                     N[0] /= N_length;
//                     N[1] /= N_length;
//                     N[2] /= N_length;

//                     // Overwrite angle
//                     const float randRotationAngleNeedsPi = (e2 * 2.0 - 1.0);

//                     // Rotation
//                     {
//                         float rotationMatrix[9]
//                               = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
//                         RotationMatrixAroundAxis_MultiplyAngleByPi_CudaSafe(
//                               randRotationAngleNeedsPi, (float *)N, (float *)rotationMatrix);

//                         for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
//                               ++pointIdx) {
//                             float shiftedPoint[3];
//                             shiftedPoint[0]
//                                   = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                           + pointIdx * 3 + 0]
//                                   - leftPoint_x;
//                             shiftedPoint[1]
//                                   = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                           + pointIdx * 3 + 1]
//                                   - leftPoint_y;
//                             shiftedPoint[2]
//                                   = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                           + pointIdx * 3 + 2]
//                                   - leftPoint_z;

//                             // Rotate and stuff back in shifted point
//                             RotateVectorByMatrix((float *)rotationMatrix, (float *)shiftedPoint);
//                             // Update the point with the rotated version
//                             pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                   + pointIdx * 3 + 0]
//                                   = shiftedPoint[0] + leftPoint_x;
//                             pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                   + pointIdx * 3 + 1]
//                                   = shiftedPoint[1] + leftPoint_y;
//                             pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
//                                   + pointIdx * 3 + 2]
//                                   = shiftedPoint[2] + leftPoint_z;
//                         }

//                         // Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
//                         // We can do a different approach later.
//                         // Here, we want to do a perturb update call
//                         twisty::PerturbUtils::UpdateTangentsFromPos_CudaSafe(
//                               &(pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx]),
//                               &(pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx]),
//                               numSegmentsPerCurve, csBoundaryConditions);

//                         twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel_CudaSafe(
//                               &(pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx]),
//                               &(pPerGlobalThreadScratchSpaceCurvatures
//                                           [CurrentThreadCurvatureStartIdx]),
//                               numSegmentsPerCurve, csBoundaryConditions);
//                     }

//                     double pathWeightLog10
//                           = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CudaSafe(
//                                 &(pPerGlobalThreadScratchSpaceCurvatures
//                                             [CurrentThreadCurvatureStartIdx]),
//                                 (numSegmentsPerCurve - 1), pWeightLookupTable,
//                                 weightLookupTableSize, ds, minCurvature, maxCurvature,
//                                 curvatureStepSize);

//                     // Else, contribute to the paths
//                     CombinedWeightValues_C_AddValue(
//                           perThreadWeightValues[threadIdx.x], pathWeightLog10);
//                 }
//             }
//         }

//         // First thread in warp responsible for combining all the weights into one
//         __syncthreads();

//         if (threadIdx.x == 0) {
//             for (uint32_t warpThreadIdx = 1; warpThreadIdx < blockDim.x; ++warpThreadIdx) {
//                 perThreadWeightValues[0] = CombinedWeightValues_C_CombineValues(
//                       perThreadWeightValues[0], perThreadWeightValues[warpThreadIdx]);
//             }

//             // Finally, we write to the combined final values
//             pFinalCombinedValues[blockIdx.x * numCombinedWeightValuesPerWarp
//                   + combinedWeightValuesWarpIdx]
//                   = perThreadWeightValues[0];
//         }
//     }
// #undef GlobalThreadIdx
// #undef NumPosPerCurve
// #undef NumTanPerCurve
// #undef NumCurvaturesPerCurve
// #undef CurrentThreadPosStartIdx
// #undef CurrentThreadTanStartIdx
// #undef CurrentThreadCurvatureStartIdx
// }
}