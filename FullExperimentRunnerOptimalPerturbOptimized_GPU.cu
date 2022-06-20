#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"

#include <boost\multiprecision\cpp_dec_float.hpp>

#include "CombinedWeightUtils.h"
#include "CurvePerturbUtils.h"
#include "CurveUtils.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "PathWeighters.h"

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

const uint32_t PerturbGridSize = 64;
const uint32_t PerturbBlockSize = 64;

namespace twisty {
static void CudaSafeErrorCheck(cudaError_t error, std::string message)
{
    if (error != cudaSuccess) {
        std::string errorString(cudaGetErrorString(error));
        fprintf(stdout, "ERROR: %s : %s\n", message.c_str(), errorString.c_str());
        // assert(false);
    }
}
__global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel(
      const int64_t numCombinedWeightValuesTotal,
      const int64_t numCombinedWeightValuesPerWarp,
      const int64_t numPathsPerThread,
      const int64_t numPathsToSkipPerThread,
      const int64_t numSegmentsPerCurve,
      curandState_t *const pCurandStates,
      float *const pPerGlobalThreadScratchSpacePositions,
      float *const pPerGlobalThreadScratchSpaceTangents,
      float *const pPerGlobalThreadScratchSpaceCurvatures,
      CombinedWeightValues_C *const pFinalCombinedValues,
      const int32_t weightingMethod,
      const double pathNormalizerLog10,
      const twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions,
      const double *const pWeightLookupTable,
      const int32_t weightLookupTableSize,
      const double ds,
      const double minCurvature,
      const double maxCurvature,
      const double curvatureStepSize);

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

    assert(m_experimentParams.weightingParameters.scatterValues.size() > 0);

    // TODO: For now, we simply will support one scattering value
    if (m_experimentParams.weightingParameters.scatterValues.size() > 1) {
        std::cout << "WARNING: Only one scatter value supported, defaulting to first specified "
                     "scatter parameter"
                  << std::endl;
    }
    m_experimentParams.weightingParameters.scatter
          = m_experimentParams.weightingParameters.scatterValues[0];
    std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;

    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::SimplifiedModel) {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_segmentLength);
    } else {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_segmentLength);
    }

    lookupEvaluator->ExportValues(m_experimentDirPath.string());

    twisty::PerturbUtils::BoundaryConditions boundaryConditions
          = m_upInitialCurve->GetBoundaryConditions();

    bool result = SetupCudaDevice();
    if (!result) {
        printf("Failed to setup cuda device\n");
        return {};
    }

    // Calculate number of paths needed to generate

    const uint32_t warpPathCount = MaxNumPathsPerCombinedWeight;
    const uint32_t numGlobalPerturbThreads = PerturbGridSize * PerturbBlockSize;

    const uint32_t numCombinedWeightValuesTotal
          = (m_experimentParams.numPathsInExperiment + MaxNumPathsPerCombinedWeight - 1)
          / MaxNumPathsPerCombinedWeight;
    const uint32_t numCombinedWeightValuesPerWarp
          = (numCombinedWeightValuesTotal + PerturbGridSize - 1) / PerturbGridSize;
    const uint32_t numPathsPerThread
          = (MaxNumPathsPerCombinedWeight + PerturbBlockSize - 1) / PerturbBlockSize;

    std::cout << "Num Global Perturb Threads: " << numGlobalPerturbThreads << std::endl;
    std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
    std::cout << "numPathsPerBatch: " << warpPathCount << std::endl;
    std::cout << "Num Combined Weight Values Total: " << numCombinedWeightValuesTotal << std::endl;
    std::cout << "Num Combined weights per warp: " << numCombinedWeightValuesPerWarp << std::endl;
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

    // We dont need this actually, so we can just load the default one
    //     {
    //         // If we can load the fn data, load it
    //         if (std::filesystem::exists(fnFilePath)) {
    //             std::cout << "Using cached fd file at: " << fnFilePath << std::endl;
    //             std::ifstream inFile(fnFilePath);
    //             upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(inFile);
    //             inFile.close();
    //         }
    //         // We need to generate it this time and save it off to use next time
    //         else {
    //             // This is the max M value
    //             const int maxorder = m_upInitialCurve->m_numSegments;

    //             // Generate the fn data table
    //             const int numZSamples = 5000;
    //             const int numIntegrationSamples = 5000;

    //             // Arbitrarily set min and max |r_vec| values.
    //             // Why this specific max bound, I do not know
    //             const double rMin = 0.0;
    //             const double rMax = 200.0;
    //             upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(
    //                   numZSamples, numIntegrationSamples, maxorder, rMin, rMax);

    //             std::ofstream outFile(fnFilePath);
    //             dynamic_cast<PathWeighting::NormalizerStuff::FN *>(upFN.get())->WriteToFile(outFile);
    //             outFile.close();
    //         }
    //     }
    //     PathWeighting::NormalizerStuff::BaseNormalizer &fn = (*upFN);

    // Why the 1/(delta s) = (M+2)/s?
    //     Farlor::Vector3 Z = (boundaryConditions.m_endPos - boundaryConditions.m_startPos)
    //                 * (m_upInitialCurve->m_numSegments + 2) / boundaryConditions.arclength
    //           - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
    //     std::cout << "Z: " << Z << std::endl;
    //     std::cout << "|Z|: " << Z.Magnitude() << std::endl;

    PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = 1.0;
    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::RadiativeTransfer) {
        pathNormalizer = PathWeighting::NormalizerStuff::Norm(m_upInitialCurve->m_numSegments,
              m_upInitialCurve->m_segmentLength, boundaryConditions);
    }
    const boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10
          = boost::multiprecision::log10(pathNormalizer);

    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    std::cout << "PathNormalizerLog10: " << pathNormalizerLog10 << std::endl;

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
    std::cout << "numCombinedWeightValuesPerWarp: " << numCombinedWeightValuesPerWarp << std::endl;
    std::cout << "numPathsPerThread: " << numPathsPerThread << std::endl;

    auto perturbTimeStart = std::chrono::high_resolution_clock::now();

    twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions;
    csBoundaryConditions.m_startPos[0] = boundaryConditions.m_startPos[0];
    csBoundaryConditions.m_startPos[1] = boundaryConditions.m_startPos[1];
    csBoundaryConditions.m_startPos[2] = boundaryConditions.m_startPos[2];

    csBoundaryConditions.m_startDir[0] = boundaryConditions.m_startDir[0];
    csBoundaryConditions.m_startDir[1] = boundaryConditions.m_startDir[1];
    csBoundaryConditions.m_startDir[2] = boundaryConditions.m_startDir[2];

    csBoundaryConditions.m_endPos[0] = boundaryConditions.m_endPos[0];
    csBoundaryConditions.m_endPos[1] = boundaryConditions.m_endPos[1];
    csBoundaryConditions.m_endPos[2] = boundaryConditions.m_endPos[2];

    csBoundaryConditions.m_endDir[0] = boundaryConditions.m_endDir[0];
    csBoundaryConditions.m_endDir[1] = boundaryConditions.m_endDir[1];
    csBoundaryConditions.m_endDir[2] = boundaryConditions.m_endDir[2];

    csBoundaryConditions.arclength = boundaryConditions.arclength;

    {
        dim3 gridSize(PerturbGridSize, 1, 1);
        dim3 blockSize(PerturbBlockSize, 1, 1);
        size_t sharedMemorySizeBytes = 0;
        cudaStream_t stream = 0;

        std::cout << "Dispatching with: " << std::endl;
        std::cout << "\tGrid Size: " << PerturbGridSize << std::endl;
        std::cout << "\tBlock Size: " << PerturbBlockSize << std::endl;


        std::cout << "Weight Table Ptr: " << lookupEvaluator->AccessLookupTable().data()
                  << std::endl;
        std::cout << "Weight Table Size: " << lookupEvaluator->AccessLookupTable().size()
                  << std::endl;
        std::cout << "DS: " << lookupEvaluator->GetDs() << std::endl;
        std::cout << "Min Curvature: " << lookupEvaluator->GetMinCurvature() << std::endl;
        std::cout << "Max Curvature: " << lookupEvaluator->GetMaxCurvature() << std::endl;
        std::cout << "Curvature Step Size: " << lookupEvaluator->GetCurvatureStepSize()
                  << std::endl;

        FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel<<<gridSize, blockSize,
              sharedMemorySizeBytes, stream>>>(numCombinedWeightValuesTotal,
              numCombinedWeightValuesPerWarp,
              numPathsPerThread,
              m_experimentParams.numPathsToSkip,
              m_experimentParams.numSegmentsPerCurve,
              m_pPerGlobalThreadRandStates,
              m_pPerGlobalThreadScratchSpacePositions,
              m_pPerGlobalThreadScratchSpaceTangents,
              m_pPerGlobalThreadScratchSpaceCurvatures,
              m_pFinalCombinedValues,
              (int32_t)m_experimentParams.weightingParameters.weightingMethod,
              m_experimentParams.weightingParameters.weightingMethod
                          == twisty::WeightingMethod::RadiativeTransfer
                    ? pathNormalizerLog10.convert_to<double>()
                    : 0.0,
              csBoundaryConditions,
              m_pDeviceWeightLookupTable,
              lookupEvaluator->AccessLookupTable().size(),
              lookupEvaluator->GetDs(),
              lookupEvaluator->GetMinCurvature(),
              lookupEvaluator->GetMaxCurvature(),
              lookupEvaluator->GetCurvatureStepSize());

        CudaSafeErrorCheck(cudaGetLastError(), "GPU_GeometryRandomKernel kernal launch");
        CudaSafeErrorCheck(cudaDeviceSynchronize(), "GPU_GeometryRandomKernel kernel sync");
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
    for (auto &combinedWeightValue : combinedWeightValues) {
        // std::cout << "Combined Weight Value" << std::endl;
        // std::cout << "Extracted Value: " << ExtractFinalValue(combinedWeightValue);
        // std::cout << "\tNum Values: " << combinedWeightValue.m_numValues << std::endl;
        // std::cout << "\tOffset: " << combinedWeightValue.m_offset << std::endl;
        // std::cout << "\tRunning Total: " << combinedWeightValue.m_runningTotal << std::endl;

        const boost::multiprecision::cpp_dec_float_100 extractedDispatchWeight
              = ExtractFinalValue(combinedWeightValue);
        bigTotalExperimentWeight += extractedDispatchWeight;

        if (m_experimentParams.outputBigFloatWeights) {
            UpdateConvergenceWeight(combinedWeightValue.m_numValues, extractedDispatchWeight);
        }

        numWeightsGenerated += combinedWeightValue.m_numValues;
    }
    // bigTotalExperimentWeight *= pathNormalizer;

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
    results.experimentWeights.push_back(bigTotalExperimentWeight);
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
        printf("\nDevice Number: %d\n", i);
        printf("\tDevice name: %s\n", prop.name);
        printf("\tSM Count: %d\n", prop.multiProcessorCount);
        printf("\tSM Shared Memory: %d\n", prop.sharedMemPerBlock);
        printf("\tWarp Size: %d\n", prop.warpSize);
        printf("\tThreads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("\tPeak Memory Bandwidth (GB/s): %f\n",
              2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("\tGlobal Memory: %zu\n", prop.totalGlobalMem);
        printf("\tConst Memory: %zu\n", prop.totalConstMem);
    }

    // We select the first device only
    const uint32_t selectedDeviceIdx = 0;
    std::cout << "\nSelected device: " << selectedDeviceIdx << std::endl;
    cudaDeviceProp deviceProp;
    CudaSafeErrorCheck(cudaGetDeviceProperties(&deviceProp, 0), "Get first device prop");

    m_numSMs = deviceProp.multiProcessorCount;
    m_warpSize = deviceProp.warpSize;
    m_maxThreadsPerMultiprocessor = deviceProp.maxThreadsPerMultiProcessor;

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
      int32_t numGlobalPerturbThreads, int32_t numCombinedWeightValues, int32_t numSegments,
      const std::vector<double> &weightTable)
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

    const uint64_t weightTableBytes = sizeof(double) * weightTable.size();
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

    // Setup data structures
    float *pInitialCurvePositions = nullptr;
    float *pInitialCurveTangents = nullptr;
    float *pInitialCurveCurvatures = nullptr;
    cudaMallocHost(&pInitialCurvePositions, (numSegments + 1) * sizeof(float) * 3);
    cudaMallocHost(&pInitialCurveTangents, numSegments * sizeof(float) * 3);
    cudaMallocHost(&pInitialCurveCurvatures, (numSegments - 1) * sizeof(float));

    memcpy(pInitialCurvePositions, (float *)m_upInitialCurve->m_positions.data(),
          (numSegments + 1) * sizeof(float) * 3);

    // Update and curvature
    twisty::PerturbUtils::UpdateTangentsFromPosCudaSafe(pInitialCurvePositions,
          pInitialCurveTangents, m_upInitialCurve->m_numSegments, boundaryConditionsCudaSafe);
    twisty::PerturbUtils::UpdateCurvaturesFromTangentsCudaSafe(pInitialCurveTangents,
          pInitialCurveCurvatures, m_upInitialCurve->m_numSegments, boundaryConditionsCudaSafe,
          (int32_t)m_experimentParams.weightingParameters.weightingMethod);

    // TODO: Should this be intermixed somehow for better performance?
    std::cout << "Copying over intial curves" << std::endl;
    std::cout << "\tNum Global Perturb Threads: " << numGlobalPerturbThreads << std::endl;

    {
        uint64_t idx = 0;
        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx) {
            CudaSafeErrorCheck(cudaMemcpy((void *)&(m_pPerGlobalThreadScratchSpacePositions[idx]),
                                     (void *)pInitialCurvePositions,
                                     (numSegments + 1) * sizeof(float) * 3,
                                     cudaMemcpyHostToDevice),
                  "Copy inital positions to per thread scratch space");
            idx += (numSegments + 1) * 3;
        }
    }

    {
        uint64_t idx = 0;
        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx) {
            CudaSafeErrorCheck(cudaMemcpy((void *)&(m_pPerGlobalThreadScratchSpaceTangents[idx]),
                                     (void *)pInitialCurveTangents,
                                     numSegments * sizeof(float) * 3,
                                     cudaMemcpyHostToDevice),
                  "Copy inital tangents to per thread scratch space");
            idx += numSegments * 3;
        }
    }

    {
        uint64_t idx = 0;
        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx) {
            CudaSafeErrorCheck(cudaMemcpy((void *)&(m_pPerGlobalThreadScratchSpaceCurvatures[idx]),
                                     (void *)pInitialCurveCurvatures,
                                     (numSegments - 1) * sizeof(float),
                                     cudaMemcpyHostToDevice),
                  "Copy inital curvatures to per thread scratch space");
            idx += (numSegments - 1);
        }
    }

    std::vector<CombinedWeightValues_C> finalCombinedWeights(numCombinedWeightValues);
    for (int i = 0; i < finalCombinedWeights.size(); i++) {
        CombinedWeightValues_C_Reset(finalCombinedWeights[i]);
    }
    cudaMemcpy((void *)m_pFinalCombinedValues, (void *)finalCombinedWeights.data(),
          finalCombinedWeights.size() * sizeof(CombinedWeightValues_C), cudaMemcpyHostToDevice);

    cudaMemcpy((void *)m_pDeviceWeightLookupTable, (void *)weightTable.data(),
          weightTable.size() * sizeof(double), cudaMemcpyHostToDevice);

    return true;
}

void FullExperimentRunnerOptimalPerturbOptimized_GPU::CleanupCudaPerturb()
{
    CudaSafeErrorCheck(cudaFree((void *)m_pFinalCombinedValues),
          "Cuda free combined weight values for final answer");

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

__device__ double WeightCurveViaCurvatureLog10_CUDA(float *pCurvatureStart, uint32_t numCurvatures,
      double *pWeightIntegral, double ds, twisty::WeightingParameters weightingParams_cuda)
{
    return 0.0;
}

__global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel(
      const int64_t numCombinedWeightValuesTotal,
      const int64_t numCombinedWeightValuesPerWarp,
      const int64_t numPathsPerThread,
      const int64_t numPathsToSkipPerThread,
      const int64_t numSegmentsPerCurve,
      curandState_t *const pCurandStates,
      float *const pPerGlobalThreadScratchSpacePositions,
      float *const pPerGlobalThreadScratchSpaceTangents,
      float *const pPerGlobalThreadScratchSpaceCurvatures,
      CombinedWeightValues_C *const pFinalCombinedValues,
      const int32_t weightingMethod,
      const double pathNormalizerLog10,
      const twisty::PerturbUtils::BoundaryConditions_CudaSafe csBoundaryConditions,
      const double *const pWeightLookupTable,
      const int32_t weightLookupTableSize,
      const double ds,
      const double minCurvature,
      const double maxCurvature,
      const double curvatureStepSize)
{
    __shared__ CombinedWeightValues_C perThreadWeightValues[PerturbBlockSize];

    // Should be between 0 and max num threads - 1
    volatile uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

    volatile int32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
    volatile int32_t NumTanPerCurve = numSegmentsPerCurve;
    volatile int32_t NumCurvaturesPerCurve = (numSegmentsPerCurve - 1);

    volatile int32_t CurrentThreadPosStartIdx = NumPosPerCurve * 3 * globalThreadIdx;
    volatile int32_t CurrentThreadTanStartIdx = NumTanPerCurve * 3 * globalThreadIdx;
    volatile int32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * globalThreadIdx;

    // Ok, we want to loop over the outer batches first, the number per warp
    for (int64_t combinedWeightValuesWarpIdx = 0;
          combinedWeightValuesWarpIdx < numCombinedWeightValuesPerWarp;
          combinedWeightValuesWarpIdx++) {
        CombinedWeightValues_C_Reset(perThreadWeightValues[threadIdx.x]);

        // We want to stop generating in this case
        if (combinedWeightValuesWarpIdx + numCombinedWeightValuesPerWarp * blockIdx.x
              >= numCombinedWeightValuesTotal) {
            // printf("Exiting early\n");
            continue;
        }

        // We need a loop over the batches
        for (int64_t pathCount = 0; pathCount < numPathsToSkipPerThread + numPathsPerThread;
              pathCount++) {
            int64_t currentPathIdx
                  = numPathsPerThread * threadIdx.x + pathCount - numPathsToSkipPerThread;

            // Do our random rolls here
            float e0 = curand_uniform(&(pCurandStates[globalThreadIdx]));
            float e1 = curand_uniform(&(pCurandStates[globalThreadIdx]));
            float e2 = curand_uniform(&(pCurandStates[globalThreadIdx]));

            // We can exit once this point is reached as we have generated all the paths necessary for this thread
            if (currentPathIdx >= MaxNumPathsPerCombinedWeight) {
                printf("Breaking early as paths are done for this combined weight value\n");
                // We dont want to continue if we have already generated the correct number of paths.
                break;
            }

            // Ok, now we first want to reset the combined weight stuff
            {
                // This is the perturbation piece.
                // Can we do this in place, most likely
                // This will modify pCurrentThreadCurve
                // Remember, the structure of this is:
                // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M
                {
                    // Should be 2 - 180
                    int64_t maxDiff = min((int)(numSegmentsPerCurve - 2), 25);
                    int64_t diff = floorf(e0 * (maxDiff - 2)) + 2;

                    // -2 from the -1 to offset for the +1, and -1 as required by index
                    int64_t leftPointIndex = floorf(e1 * (numSegmentsPerCurve - diff - 2)) + 1;

                    int64_t rightPointIndex = leftPointIndex + diff;

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
                    volatile float N_length = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
                    N[0] /= N_length;
                    N[1] /= N_length;
                    N[2] /= N_length;

                    // Overwrite angle
                    const float randRotationAngle = (e2 * 2.0 - 1.0) * TwistyPi;

                    // Rotation
                    {
                        float rotationMatrix[9]
                              = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                        RotationMatrixAroundAxis(
                              randRotationAngle, (float *)N, (float *)rotationMatrix);

                        for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
                              ++pointIdx) {
                            float shiftedPoint[3];
                            shiftedPoint[0]
                                  = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                          + pointIdx * 3 + 0]
                                  - leftPoint_x;
                            shiftedPoint[1]
                                  = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                          + pointIdx * 3 + 1]
                                  - leftPoint_y;
                            shiftedPoint[2]
                                  = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                          + pointIdx * 3 + 2]
                                  - leftPoint_z;

                            // Rotate and stuff back in shifted point
                            RotateVectorByMatrix((float *)rotationMatrix, (float *)shiftedPoint);
                            // Update the point with the rotated version
                            pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                  + pointIdx * 3 + 0]
                                  = shiftedPoint[0] + leftPoint_x;
                            pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                  + pointIdx * 3 + 1]
                                  = shiftedPoint[1] + leftPoint_y;
                            pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx
                                  + pointIdx * 3 + 2]
                                  = shiftedPoint[2] + leftPoint_z;
                        }

                        // //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                        // //We can do a different approach later.
                        // // Here, we want to do a perturb update call
                        twisty::PerturbUtils::UpdateTangentsFromPosCudaSafe(
                              &(pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx]),
                              &(pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx]),
                              numSegmentsPerCurve, csBoundaryConditions);

                        twisty::PerturbUtils::UpdateCurvaturesFromTangentsCudaSafe(
                              &(pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx]),
                              &(pPerGlobalThreadScratchSpaceCurvatures
                                          [CurrentThreadCurvatureStartIdx]),
                              numSegmentsPerCurve, csBoundaryConditions, weightingMethod);
                    }

                    double pathWeightLog10
                          = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CudaSafe(
                                &(pPerGlobalThreadScratchSpaceCurvatures
                                            [CurrentThreadCurvatureStartIdx]),
                                (numSegmentsPerCurve - 1), pWeightLookupTable,
                                weightLookupTableSize, ds, minCurvature, maxCurvature,
                                curvatureStepSize);
                    pathWeightLog10 += pathNormalizerLog10;

                    if (pathCount < numPathsToSkipPerThread) {
                        // Skip
                    } else {
                        // Else, contribute to the paths
                        CombinedWeightValues_C_AddValue(
                              perThreadWeightValues[threadIdx.x], pathWeightLog10);
                    }
                }
            }
        }

        // First thread in warp responsible for combining all the weights into one
        __syncthreads();

        if (threadIdx.x == 0) {
            for (uint32_t warpThreadIdx = 1; warpThreadIdx < blockDim.x; ++warpThreadIdx) {
                perThreadWeightValues[0] = CombinedWeightValues_C_CombineValues(
                      perThreadWeightValues[0], perThreadWeightValues[warpThreadIdx]);
            }

            // Finally, we write to the combined final values
            pFinalCombinedValues[blockIdx.x * numCombinedWeightValuesPerWarp
                  + combinedWeightValuesWarpIdx]
                  = perThreadWeightValues[0];
        }
    }
}

}