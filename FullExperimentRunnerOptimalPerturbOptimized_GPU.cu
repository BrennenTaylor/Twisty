#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"

#include <boost\multiprecision\cpp_dec_float.hpp>

#include "CurvePerturbUtils.h"
#include "CurveUtils.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

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

const uint32_t PerturbWarpSize = 32;
const uint32_t PerturbGridSize = 10;

namespace twisty
{
    static void CudaSafeErrorCheck(cudaError_t error, std::string message)
    {
        if (error != cudaSuccess)
        {
            std::string errorString(cudaGetErrorString(error));
            fprintf(stderr, "ERROR: %s : %s\n", message.c_str(), errorString.c_str());
            assert(false);
        }
    }

            // Assumes pVector3f is an array of 3 floats
    static __host__ __device__ void NormalizeVector3f(float* pVector3f)
    {
        float normalizer = pVector3f[0] * pVector3f[0] + pVector3f[1] * pVector3f[1] + pVector3f[2] * pVector3f[2];
        normalizer = 1.0f / sqrt(normalizer);
        pVector3f[0] *= normalizer;
        pVector3f[1] *= normalizer;
        pVector3f[2] *= normalizer;
    }

    // This has an out parameter
    static __host__ __device__  void RotationMatrixAroundAxis(float angle, float* pAxisVector3f, float* pMatrix3x3)
    {
        // Ensure its normalized
        NormalizeVector3f(pAxisVector3f);

        pMatrix3x3[0] = cos(angle) + pAxisVector3f[0] * pAxisVector3f[0] * (1.0f - cos(angle));
        pMatrix3x3[1] = pAxisVector3f[0] * pAxisVector3f[1] * (1.0f - cos(angle)) - pAxisVector3f[2] * sin(angle);
        pMatrix3x3[2] = pAxisVector3f[0] * pAxisVector3f[2] * (1.0f - cos(angle)) + pAxisVector3f[1] * sin(angle);

        pMatrix3x3[3] = pAxisVector3f[1] * pAxisVector3f[0] * (1.0f - cos(angle)) + pAxisVector3f[2] * sin(angle);
        pMatrix3x3[4] = cos(angle) + pAxisVector3f[1] * pAxisVector3f[1] * (1 - cos(angle));
        pMatrix3x3[5] = pAxisVector3f[1] * pAxisVector3f[2] * (1 - cos(angle)) - pAxisVector3f[0] * sin(angle);

        pMatrix3x3[6] = pAxisVector3f[2] * pAxisVector3f[0] * (1 - cos(angle)) - pAxisVector3f[1] * sin(angle);
        pMatrix3x3[7] = pAxisVector3f[2] * pAxisVector3f[1] * (1 - cos(angle)) + pAxisVector3f[0] * sin(angle);
        pMatrix3x3[8] = cos(angle) + pAxisVector3f[2] * pAxisVector3f[2] * (1 - cos(angle));
    }

    static __host__ __device__  float DotVector3fVector3f(float* lhs, float* rhs)
    {
        return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
    }

    // TODO: Check if needed during rewrite
    // static __host__ __device__  float MagVector3f(float* pVec)
    // {
    //     return sqrt(pVec[0] * pVec[0] + pVec[1] * pVec[1] + pVec[2] * pVec[2]);
    // }

    static __host__ __device__  void RotateVectorByMatrix(float* pRotationMatrix, float* pVector)
    {
        float val[3];
        val[0] = DotVector3fVector3f(pRotationMatrix, pVector);
        val[1] = DotVector3fVector3f(pRotationMatrix + 3, pVector);
        val[2] = DotVector3fVector3f(pRotationMatrix + 6, pVector);
        
        // Write it back to pVector
        pVector[0] = val[0];
        pVector[1] = val[1];
        pVector[2] = val[2];
    }


    FullExperimentRunnerOptimalPerturbOptimized_GPU::FullExperimentRunnerOptimalPerturbOptimized_GPU(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper)
        : ExperimentRunner(experimentParams, bootstrapper)
    {
    }

    FullExperimentRunnerOptimalPerturbOptimized_GPU::~FullExperimentRunnerOptimalPerturbOptimized_GPU()
    {
    }

    ExperimentRunner::RunnerSpecificResults FullExperimentRunnerOptimalPerturbOptimized_GPU::RunnerSpecificRunExperiment()
    {
        /* --------------------- */
        auto setupTimeStart = std::chrono::high_resolution_clock::now();
        /* --------------------- */


        // TODO: For now, we simply will support one scattering value 
        if (m_experimentParams.weightingParameters.scatterValues.size() > 1)
        {
            std::cout << "WARNING: Only one scatter value supported, defaulting to first specified scatter parameter" << std::endl;
        }
        m_experimentParams.weightingParameters.scatter = m_experimentParams.weightingParameters.scatterValues[0];
        std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;

        if (m_experimentParams.weightingParameters.weightingMethod == WeightingMethod::SimplifiedModel)
        {
            lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(m_experimentParams.weightingParameters,
                                                                               m_upInitialCurve->m_segmentLength);
        }
        else
        {
            lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(m_experimentParams.weightingParameters,
                                                                                                 m_upInitialCurve->m_segmentLength);
        }
            
        lookupEvaluator->ExportValues(m_experimentParams.experimentDirPath);

        twisty::PerturbUtils::BoundaryConditions boundaryConditions = m_upInitialCurve->GetBoundaryConditions();

        bool result = SetupCudaDevice();
        if (!result)
        {
            printf("Failed to setup cuda device\n");
            return {};
        }

        // Calculate number of paths needed to generate

        const uint32_t warpPathCount = 1000000;
        const uint32_t numGlobalPerturbThreads = PerturbGridSize * PerturbWarpSize;
        
        uint32_t numThreadBatches = (m_experimentParams.numPathsInExperiment + warpPathCount - 1) / warpPathCount;
        uint32_t numCachedWeightsPerWarp = (numThreadBatches + PerturbGridSize - 1) / PerturbGridSize;
        uint32_t numPathsPerThread = warpPathCount / PerturbWarpSize;
        
        std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
        std::cout << "numPathsPerBatch: " << warpPathCount << std::endl;
        std::cout << "Num Thread Batches: " << numThreadBatches << std::endl;
        std::cout << "Num cached weights per warp: " << numCachedWeightsPerWarp << std::endl;
        std::cout << "Num paths per thread: " << numPathsPerThread << std::endl;
        std::cout << "Perturb Warp Size required: " << PerturbWarpSize << std::endl;
        std::cout << "Perturb Grid Size required: " << PerturbGridSize << std::endl;

        int32_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }

        auto setupCuRandTimeStart = std::chrono::high_resolution_clock::now();
        {
            result = SetupCuRandStates(numGlobalPerturbThreads, seed);
            if (!result)
            {
                printf("Failed to setup CuRand states\n");
                return {};
            }
        }
        auto setupCuRandTimeEnd = std::chrono::high_resolution_clock::now();
 
        auto setupCudaPerturbStart = std::chrono::high_resolution_clock::now();
        {
            result = SetupCudaPerturb(numGlobalPerturbThreads, numThreadBatches);
            if (!result)
            {
                printf("Failed to setup Cuda Perturb\n");
                return {};
            }
        }
        auto setupCudaPerturbEnd = std::chrono::high_resolution_clock::now();
        
        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        // Setup data structures
        std::vector<Farlor::Vector3> initialCurvePositions = m_upInitialCurve->m_positions;
        std::vector<Farlor::Vector3> initialCurveTangents(initialCurvePositions.size());
        std::vector<float> initialCurveCurvatures(initialCurvePositions.size() - 1);

        // Update and curvature
        twisty::PerturbUtils::UpdateTangentsFromPos(initialCurvePositions.data(), initialCurveTangents.data(),
            m_upInitialCurve->m_numSegments, boundaryConditions);
        twisty::PerturbUtils::UpdateCurvaturesFromTangents(initialCurveTangents.data(), initialCurveCurvatures.data(),
            m_upInitialCurve->m_numSegments, boundaryConditions, m_experimentParams.weightingParameters);

        const int64_t NumPosPerCurve = initialCurvePositions.size();
        const int64_t NumTanPerCurve = initialCurveTangents.size();
        const int64_t NumCurvaturePerCurve = initialCurveCurvatures.size();

        std::stringstream fnFilenameSS;
        fnFilenameSS << "SavedFN";
        fnFilenameSS << m_experimentParams.numSegmentsPerCurve;
        fnFilenameSS << ".fnd";
        const std::filesystem::path fnFilePath = std::filesystem::current_path() / fnFilenameSS.str();
        std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> upFN = nullptr;

        // We dont need this actually, so we can just load the default one
        {
            // If we can load the fn data, load it
            if (std::filesystem::exists(fnFilePath))
            {
                std::cout << "Using cached fd file at: " << fnFilePath << std::endl;
                std::ifstream inFile(fnFilePath);
                upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(inFile);
                inFile.close();
            }
            // We need to generate it this time and save it off to use next time
            else
            {
                // This is the max M value
                const int maxorder = m_upInitialCurve->m_numSegments;

                // Generate the fn data table
                const int numZSamples = 5000;
                const int numIntegrationSamples = 5000;

                // Arbitrarily set min and max |r_vec| values.
                // Why this specific max bound, I do not know
                const double rMin = 0.0;
                const double rMax = 200.0;
                upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(numZSamples, numIntegrationSamples, maxorder, rMin, rMax);

                std::ofstream outFile(fnFilePath);
                dynamic_cast<PathWeighting::NormalizerStuff::FN*>(upFN.get())->WriteToFile(outFile);
                outFile.close();
            }
        }
        PathWeighting::NormalizerStuff::BaseNormalizer& fn = (*upFN);

        // Why the 1/(delta s) = (M+2)/s?
        Farlor::Vector3 Z = (boundaryConditions.m_endPos - boundaryConditions.m_startPos) * (m_upInitialCurve->m_numSegments + 2) / boundaryConditions.arclength
            - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
        std::cout << "Z: " << Z << std::endl;
        std::cout << "|Z|: " << Z.Magnitude() << std::endl;

        PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = 1.0;
        if (m_experimentParams.weightingParameters.weightingMethod == WeightingMethod::RadiativeTransfer)
        {
            pathNormalizer = PathWeighting::NormalizerStuff::Norm(fn, m_upInitialCurve->m_numSegments,
                                                 Z.Magnitude(), boundaryConditions.arclength);
        }
        const boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10 = boost::multiprecision::log10(pathNormalizer);

        std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
        std::cout << "PathNormalizerLog10: " << pathNormalizerLog10 << std::endl;

        auto setupTimeEnd = std::chrono::high_resolution_clock::now();
        /* --------------------- */


        /* --------------------- */

        long long perturbTimeCount = 0;
        long long weightCalcTimeCount = 0;

        std::cout << "numPathsInExperiment specified: " << m_experimentParams.numPathsInExperiment << std::endl;

        const uint32_t numCombinedWeightValuesTotal = (m_experimentParams.numPathsInExperiment + MaxNumberOfPaths - 1)
            / MaxNumberOfPaths;
        const uint32_t numCombinedWeightValuesPerWarp = (numCombinedWeightValuesTotal + PerturbGridSize - 1) / PerturbGridSize;
        const uint32_t numCombinedWeightValuesPerThread = (MaxNumberOfPaths + PerturbWarpSize - 1) / PerturbWarpSize;

        std::cout << "numPathsInExperiment generated: " << numCombinedWeightValuesTotal * MaxNumberOfPaths << std::endl;

        std::vector<twisty::FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues> combinedWeightValues(numCombinedWeightValuesTotal);

        auto perturbTimeStart = std::chrono::high_resolution_clock::now();

        {
            dim3 gridSize(PerturbWarpSize, 1, 1);
            dim3 blockSize(PerturbGridSize, 1, 1);
            size_t sharedMemorySizeBytes = 0;
            cudaStream_t stream = 0;

            FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel << <gridSize, blockSize, sharedMemorySizeBytes, stream >> >
                (
                    numCombinedWeightValuesTotal,
                    numCombinedWeightValuesPerWarp,
                    numCombinedWeightValuesPerThread,
                    m_experimentParams.numPathsToSkip,
                    m_experimentParams.numSegmentsPerCurve,
                    m_pPerGlobalThreadRandStates,
                    m_pPerGlobalThreadScratchSpacePositions,
                    m_pPerGlobalThreadScratchSpaceTangents,
                    m_pPerGlobalThreadScratchSpaceCurvatures,
                    m_pPerGlobalThreadWorkingScratchSpacePositions,
                    m_pPerGlobalThreadWorkingScratchSpaceTangents,
                    m_pPerGlobalThreadWorkingScratchSpaceCurvatures,
                    m_pPerThreadCombinedWeightValues,
                    m_pFinalCombinedValues,
                    boundaryConditions,
                    nullptr
                );

            CudaSafeErrorCheck(cudaGetLastError(), "Rand state init kernal launch");
            CudaSafeErrorCheck(cudaDeviceSynchronize(), "Rand state kernel sync");
        }

        auto perturbTimeEnd = std::chrono::high_resolution_clock::now();
        perturbTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(perturbTimeEnd - perturbTimeStart).count();

        // -------------------
        auto weightingTimeStart = std::chrono::high_resolution_clock::now();

        // We need to calculate the absorbtion/scattering piece
        boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;
        // No, we calculating the weighting
        for (auto& combinedWeightValue : combinedWeightValues)
        {
            bigTotalExperimentWeight += combinedWeightValue.ExtractFinalValue();
        }
        bigTotalExperimentWeight *= pathNormalizer;

        auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
        weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightingTimeEnd - weightingTimeStart).count();
        /* --------------------- */

        // Cleanup stuff

        {
            CleanupCudaPerturb();
            CleanupCudaRandStates();
            CleanupCudaDevice();
        }

        auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupTimeEnd - setupTimeStart);

        ExperimentResults results;
        results.experimentWeights.push_back(bigTotalExperimentWeight);
        results.totalPathsGenerated = numCombinedWeightValuesTotal * MaxNumberOfPaths;
        results.numFailedPaths = 0;

        ExperimentRunner::RunnerSpecificResults specificResult;
        specificResult.experimentResults = std::make_optional<ExperimentResults>(results);
        specificResult.setupMsCount = timeMs.count();
        specificResult.runExperimentMsCount = perturbTimeCount;
        specificResult.weightingMsCount = weightCalcTimeCount;

        return specificResult;
    }

    bool FullExperimentRunnerOptimalPerturbOptimized_GPU::SetupCudaDevice()
    {
        // Query for the number of devices avalible
        int32_t numDevices = 0;
        CudaSafeErrorCheck(cudaGetDeviceCount(&numDevices), "Get Devices");

        // We need at least one cuda device
        if (numDevices <= 0)
        {
            printf("No CUDA device avalible. Cannot execute experiment.\n");
            return false;
        }

        // Iterate over all devices and report the device stats.
        for (int32_t i = 0; i < numDevices; ++i)
        {
            cudaDeviceProp prop;
            CudaSafeErrorCheck(cudaGetDeviceProperties(&prop, i), "Get Device Prop");
            printf("\nDevice Number: %d\n", i);
            printf("\tDevice name: %s\n", prop.name);
            printf("\tSM Count: %d\n", prop.multiProcessorCount);
            printf("\tSM Shared Memory: %d\n", prop.sharedMemPerBlock);
            printf("\tWarp Size: %d\n", prop.warpSize);
            printf("\tThreads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
            printf("\tPeak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
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

    void FullExperimentRunnerOptimalPerturbOptimized_GPU::CleanupCudaDevice()
    {
    }
    
    bool FullExperimentRunnerOptimalPerturbOptimized_GPU::SetupCuRandStates(uint32_t numGlobalPerturbThreads, uint32_t seed)
    {
        std::cout << "Setup Cuda Perturb: " << std::endl;
        uint64_t usedMemoryInBytes = 0;

        // Random Seed Kernel
        // Every block thread needs its own curand state
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadRandStates, sizeof(curandState_t) * numGlobalPerturbThreads),
            "RandState malloc");
        usedMemoryInBytes += (sizeof(curandState_t) * numGlobalPerturbThreads);

        int blockSizeRandKernel = 0;
        int minGridSizeRandKernel = 0;
        size_t sharedMemoryUse = 0;
        size_t maxBlockSize = 0;
        CudaSafeErrorCheck(cudaOccupancyMaxPotentialBlockSize(&minGridSizeRandKernel, &blockSizeRandKernel, FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState, sharedMemoryUse, maxBlockSize),
            "Failed to calculating occupancy for InitializeCuRandState kernel");
        std::cout << "\nInitializeCurandState: " << std::endl;
        std::cout << "\tBlock Size: " << blockSizeRandKernel << std::endl;
        std::cout << "\tMin Grid Size: " << minGridSizeRandKernel << std::endl;

        size_t gridSizeRandKernel = ((numGlobalPerturbThreads)+blockSizeRandKernel - 1) / blockSizeRandKernel;

        printf("\tInitializeCurandState Grid Size: %d\n", gridSizeRandKernel);
        printf("\tInitializeCurandState Block Size: %d\n", blockSizeRandKernel);

        // Dispatch CurandState
        // We need a dispatch that initializes curand per thread
        {
            dim3 gridSize(gridSizeRandKernel, 1, 1);
            dim3 blockSize(blockSizeRandKernel, 1, 1);
            size_t sharedMemorySizeBytes = 0;
            cudaStream_t stream = 0;

            FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState << <gridSize, blockSize, sharedMemorySizeBytes, stream >> > (
                static_cast<uint32_t>(seed),
                m_pPerGlobalThreadRandStates,
                numGlobalPerturbThreads
            );

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
        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadRandStates),
            "RandState free");
    }


    // Pass in total number of threads that can be used, as well as the number of batches of 10^6 paths which will be generated
    bool FullExperimentRunnerOptimalPerturbOptimized_GPU::SetupCudaPerturb(uint32_t numGlobalPerturbThreads, uint32_t numCombinedWeightValues)
    {
        std::cout << "Setup Cuda Perturb: " << std::endl;
        uint64_t usedMemoryInBytes = 0;

        // Every global thread needs its own curve scratch space
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadScratchSpacePositions, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Left Scratch Space Positions");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        // Every global thread needs its own curve scratch space left and right and working
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadScratchSpaceTangents, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Left Scratch Space Tangents");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        // Every global thread needs its own curve scratch space left and right and working
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadScratchSpaceCurvatures, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1)),
            "Cuda malloc Left Scratch Space Curvatures");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadWorkingScratchSpacePositions, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Working Scratch Space Positions");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadWorkingScratchSpaceTangents, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Working Scratch Space Tangents");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadWorkingScratchSpaceCurvatures, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1)),
            "Cuda malloc Working Scratch Space Curvatures");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerThreadCombinedWeightValues, sizeof(CombinedWeightValues_C) * numGlobalPerturbThreads),
            "Cuda malloc combined weight values per thread");
        usedMemoryInBytes += (sizeof(CombinedWeightValues_C) * numGlobalPerturbThreads);

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pFinalCombinedValues, sizeof(CombinedWeightValues_C) * numCombinedWeightValues),
            "Cuda malloc combined weight values per thread");
        usedMemoryInBytes += (sizeof(CombinedWeightValues_C) * numCombinedWeightValues);

        std::cout << "\tUsed Device Memory Before: " << m_usedDeviceMemoryInBytes << std::endl;
        std::cout << "\tNewly allocated memory: " << usedMemoryInBytes << std::endl;

        m_usedDeviceMemoryInBytes += usedMemoryInBytes;

        std::cout << "\tUsed Device Memory After: " << m_usedDeviceMemoryInBytes << std::endl;

        // Copy that data over to the gpu
        
        // Setup data structures
        twisty::PerturbUtils::BoundaryConditions boundaryConditions;
        boundaryConditions.arclength = m_upInitialCurve->m_arclength;
        boundaryConditions.m_startPos = m_upInitialCurve->m_basePos;
        boundaryConditions.m_startDir = m_upInitialCurve->m_baseTangent;
        boundaryConditions.m_endPos = m_upInitialCurve->m_targetPos;
        boundaryConditions.m_endDir = m_upInitialCurve->m_targetTangent;

        std::vector<Farlor::Vector3> initialCurvePositions(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> initialCurveTangents(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<float> initialCurveCurvatures(m_experimentParams.numSegmentsPerCurve);

        // Positions
        // Hard code the first two positions
        initialCurvePositions[0] = m_upInitialCurve->m_basePos;
        initialCurvePositions[1] = m_upInitialCurve->m_basePos + m_upInitialCurve->m_baseTangent.Normalized() * m_upInitialCurve->m_segmentLength;
        for (int64_t segmentIdx = 2; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
        {
            initialCurvePositions[segmentIdx] = m_upInitialCurve->m_positions[segmentIdx];
        }
        // Hard code the final position
        initialCurvePositions[m_experimentParams.numSegmentsPerCurve] = m_upInitialCurve->m_targetPos;

        twisty::PerturbUtils::UpdateTangentsFromPos(initialCurvePositions.data(), initialCurveTangents.data(),
            m_upInitialCurve->m_numSegments, boundaryConditions);
        twisty::PerturbUtils::UpdateCurvaturesFromTangents(initialCurveTangents.data(), initialCurveCurvatures.data(),
            m_upInitialCurve->m_numSegments, boundaryConditions, m_experimentParams.weightingParameters);

        const int64_t NumPosPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumTanPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumCurvaturePerCurve = (m_upInitialCurve->m_numSegments);
        const int64_t NumPosFloatsPerCurve = NumPosPerCurve * 3;
        const int64_t NumTanFloatsPerCurve = NumPosPerCurve * 3;
        const int64_t NumCurvatureFloatsPerCurve = NumPosPerCurve * 1;

        // TODO: Should this be intermixed somehow for better performance?
        uint64_t idx = 0;
        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx)
        {
            for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
            {
                cudaMemcpy((void*)&m_pPerGlobalThreadScratchSpacePositions[idx], (void*)initialCurvePositions.data(), initialCurvePositions.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadWorkingScratchSpacePositions[idx], (void*)initialCurvePositions.data(), initialCurvePositions.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
            }
            idx += initialCurvePositions.size() * 3;
        }

        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx)
        {
            // Copy Tan
            for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                cudaMemcpy((void*)&m_pPerGlobalThreadScratchSpaceTangents[idx], (void*)initialCurveTangents.data(), initialCurveTangents.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadWorkingScratchSpaceTangents[idx], (void*)initialCurveTangents.data(), initialCurveTangents.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
            }
            idx += initialCurveTangents.size() * 3;
        }

        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx)
        {
            // Copy Curvatures
            for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
            {
                cudaMemcpy((void*)&m_pPerGlobalThreadScratchSpaceCurvatures[idx], (void*)initialCurveCurvatures.data(), initialCurveCurvatures.size() * sizeof(float) * 1, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadWorkingScratchSpaceCurvatures[idx], (void*)initialCurveCurvatures.data(), initialCurveCurvatures.size() * sizeof(float) * 1, cudaMemcpyHostToDevice);
            }

            idx += initialCurveCurvatures.size();
        }


        // TODO: Is this cache even used?
        std::vector<CombinedWeightValues_C> perThreadCombinedWeightValues(numGlobalPerturbThreads);
        cudaMemcpy((void*)m_pPerThreadCombinedWeightValues, (void*)perThreadCombinedWeightValues.data(), perThreadCombinedWeightValues.size() * sizeof(CombinedWeightValues_C), cudaMemcpyHostToDevice);

        std::vector<CombinedWeightValues_C> finalCombinedWeights(numCombinedWeightValues);
        cudaMemcpy((void*)m_pFinalCombinedValues, (void*)finalCombinedWeights.data(), finalCombinedWeights.size() * sizeof(CombinedWeightValues_C), cudaMemcpyHostToDevice);

        return true;
    }

    void FullExperimentRunnerOptimalPerturbOptimized_GPU::CleanupCudaPerturb()
    {
        CudaSafeErrorCheck(cudaFree((void*)m_pFinalCombinedValues),
            "Cuda free combined weight values for final answer");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerThreadCombinedWeightValues),
            "Cuda free combined weight values per thread");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadWorkingScratchSpaceCurvatures),
            "Cuda free Working Scratch Space Curvatures");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadWorkingScratchSpaceTangents),
            "Cuda free Working Scratch Space Tangents");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadWorkingScratchSpacePositions),
            "Cuda free Working Scratch Space Positions");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadScratchSpaceCurvatures),
            "Cuda free Left Scratch Space Curvatures");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadScratchSpaceTangents),
            "Cuda free Left Scratch Space Tangents");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadScratchSpacePositions),
            "Cuda free Left Scratch Space Positions");
    }



    __global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_InitializeCurandState(uint32_t seed, curandState_t* pStates, uint32_t maxNumStates)
    {
        // Unique index
        const uint32_t globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (globalThreadIdx < maxNumStates)
        {
            curand_init(seed + globalThreadIdx, 0, 0, &pStates[globalThreadIdx]);
        }
    }

    __device__ void MergeCombinedWeightValues(FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C* pLeft,
        FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C* pRight)
    {
        //// In the case we haven't added a value yet, we can early out
        //if (pRight->m_numValues == 0)
        //{
        //    // Nothing needs done
        //    return;
        //}


        //// If we already have a value and its not larger than the current max, then throw it in.
        //if (pLeft->m_maxWeightLog10 > pRight->m_maxWeightLog10)
        //{
        //    pLeft->m_runningTotal += pow(10.0, (valueLog10 + pData->m_offset));
        //    pLeft->m_numValues++;
        //    return;
        //}

        //// If it is larger, we need to rescale everything around that new value

        //// New difference
        //double newMaxPossibleFinalWeightLog10 = valueLog10 + MaxNumberOfPathsLog10;
        //double newOffset = MaxDoubleLog10 - newMaxPossibleFinalWeightLog10;

        //double offsetDelta = newOffset - pData->m_offset;
        //double log10RunningTotal = log10(pData->m_runningTotal);
        //double adjustedLog10RunningTotal = log10RunningTotal + offsetDelta;
        //pData->m_runningTotal = pow(10.0, adjustedLog10RunningTotal);

        //// Update
        //pData->m_maxWeightLog10 = valueLog10;
        //pData->m_maxPossibleFinalWeightLog10 = newMaxPossibleFinalWeightLog10;
        //pData->m_offset = newOffset;

        //pData->m_runningTotal += pow(10.0, (valueLog10 + pData->m_offset));
        //pData->m_numValues++;
    }


    // __device__ void RecalculateTangentsCurvaturesFromPos_CUDA(float* pPositions, float* pTangents, float* pCurvatures, uint32_t numSegmentsPerCurve,
    //     FullExperimentRunnerOptimalPerturbOptimized_GPU::BoundaryConditions_CUDA boundaryConditions)
    // {
    //     const float ds = boundaryConditions.arclength / numSegmentsPerCurve;

    //     // Set initial and final positions
    //     pPositions[0 * 3 + 0] = boundaryConditions.m_startPos[0];
    //     pPositions[0 * 3 + 1] = boundaryConditions.m_startPos[1];
    //     pPositions[0 * 3 + 2] = boundaryConditions.m_startPos[2];

    //     pPositions[1 * 3 + 0] = pPositions[0 * 3 + 0] + ds * boundaryConditions.m_startDir[0];
    //     pPositions[1 * 3 + 1] = pPositions[0 * 3 + 1] + ds * boundaryConditions.m_startDir[1];
    //     pPositions[1 * 3 + 2] = pPositions[0 * 3 + 2] + ds * boundaryConditions.m_startDir[2];


    //     pPositions[numSegmentsPerCurve * 3 + 0] = boundaryConditions.m_endPos[0];
    //     pPositions[numSegmentsPerCurve * 3 + 1] = boundaryConditions.m_endPos[1];
    //     pPositions[numSegmentsPerCurve * 3 + 2] = boundaryConditions.m_endPos[2];

    //     // Update tangents
    //     // Set first tangent directly, defined by boundary conditions
    //     pTangents[0 * 3 + 0] = boundaryConditions.m_startDir[0];
    //     pTangents[0 * 3 + 1] = boundaryConditions.m_startDir[1];
    //     pTangents[0 * 3 + 2] = boundaryConditions.m_startDir[2];

    //     // Set others via finite difference
    //     for (uint32_t i = 1; i < numSegmentsPerCurve; ++i)
    //     {
    //         float diff_x = pPositions[(i + 1) * 3 + 0] - pPositions[(i - 1) * 3 + 0];
    //         float diff_y = pPositions[(i + 1) * 3 + 1] - pPositions[(i - 1) * 3 + 1];
    //         float diff_z = pPositions[(i + 1) * 3 + 2] - pPositions[(i - 1) * 3 + 2];
    //         float diff_length = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
    //         pTangents[i * 3 + 0] = diff_x / diff_length;
    //         pTangents[i * 3 + 1] = diff_y / diff_length;
    //         pTangents[i * 3 + 2] = diff_z / diff_length;
    //     }
    //     pTangents[numSegmentsPerCurve * 3 + 0] = boundaryConditions.m_endDir[0];
    //     pTangents[numSegmentsPerCurve * 3 + 1] = boundaryConditions.m_endDir[1];
    //     pTangents[numSegmentsPerCurve * 3 + 2] = boundaryConditions.m_endDir[2];

    //     // Calculate curvature
    //     // First, we calcualte only the first using the old method
    //     {
    //         float denom = 1.0f / ds;
    //         float diff_x = (pTangents[1 * 3 + 0] - pTangents[0 * 3 + 0]) * denom;
    //         float diff_y = (pTangents[1 * 3 + 1] - pTangents[0 * 3 + 1]) * denom;
    //         float diff_z = (pTangents[1 * 3 + 2] - pTangents[0 * 3 + 2]) * denom;
    //         float diff_length = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
    //         {
    //             const float curvature = diff_length;
    //             pCurvatures[0] = curvature;
    //         }
    //     }

    //     // All the rest we calculate using the new method
    //     for (uint32_t i = 1; i < numSegmentsPerCurve; ++i)
    //     {
    //         // First, grab tangent
    //         float tan_x = pTangents[i * 3 + 0];
    //         float tan_y = pTangents[i * 3 + 1];
    //         float tan_z = pTangents[i * 3 + 2];

    //         // Second, calculate dp2ds2
    //         float dp2ds2_x = pPositions[(i + 1) * 3 + 0] + pPositions[(i - 1) * 3 + 0] - 2.0f * pPositions[i * 3 + 0];
    //         float dp2ds2_y = pPositions[(i + 1) * 3 + 1] + pPositions[(i - 1) * 3 + 1] - 2.0f * pPositions[i * 3 + 1];
    //         float dp2ds2_z = pPositions[(i + 1) * 3 + 2] + pPositions[(i - 1) * 3 + 2] - 2.0f * pPositions[i * 3 + 2];
    //         dp2ds2_x *= (1.0f / (ds * ds));
    //         dp2ds2_y *= (1.0f / (ds * ds));
    //         dp2ds2_z *= (1.0f / (ds * ds));

    //         // Third, calculate dTds
    //         float tanDotdp2ds2 = tan_x * dp2ds2_x + tan_y * dp2ds2_y + tan_z * dp2ds2_z;

    //         float dTds_x = dp2ds2_x - tan_x * tanDotdp2ds2;
    //         float dTds_y = dp2ds2_y - tan_y * tanDotdp2ds2;
    //         float dTds_z = dp2ds2_z - tan_z * tanDotdp2ds2;
    //         pCurvatures[i] = sqrt(dTds_x * dTds_x + dTds_y * dTds_y + dTds_z * dTds_z);
    //     }
    // }

    __device__ double WeightCurveViaCurvatureLog10_CUDA(float* pCurvatureStart, uint32_t numCurvatures, double* pWeightIntegral, double ds,
        twisty::WeightingParameters weightingParams_cuda)
    {
        return 0.0;
    }

    __global__ void FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryRandomKernel(
        int64_t numCombinedWeightValuesTotal,
        int64_t numCombinedWeightValuesPerWarp,
        int64_t numCombinedWeightValuesPerThread,
        int64_t numPathsToSkipPerThread,
        int64_t numSegmentsPerCurve,
        curandState_t* pCurandStates,
        float* pPerGlobalThreadScratchSpacePositions,
        float* pPerGlobalThreadScratchSpaceTangents,
        float* pPerGlobalThreadScratchSpaceCurvatures,
        float* pPerGlobalThreadWorkingScratchSpacePositions,
        float* pPerGlobalThreadWorkingScratchSpaceTangents,
        float* pPerGlobalThreadWorkingScratchSpaceCurvatures,
        FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C* pPerThreadCombinedWeightValues,
        FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C* pFinalCombinedValues,
        const twisty::WeightingParameters& weightingParams,
        const twisty::PerturbUtils::BoundaryConditions& boundaryConditions,
        double* pLookupTable
        
        //const PathWeighting::NormalizerStuff::FN& fn
    )
    {
        const uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

        uint32_t numPathsAccepted = 0;
        uint32_t numPathsUnaccepted = 0;
        uint32_t numPathsUnacceptedTangentPdf = 0;
        uint32_t numPathsUnacceptedCurvaturePdf = 0;

        const int32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
        const int32_t NumTanPerCurve = (numSegmentsPerCurve + 1);
        const int32_t NumCurvaturesPerCurve = numSegmentsPerCurve;

        const int32_t CurrentThreadPosStartIdx = NumPosPerCurve * globalThreadIdx;
        const int32_t CurrentThreadTanStartIdx = NumTanPerCurve * globalThreadIdx;
        const int32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * globalThreadIdx;

        int64_t numToSkip = numPathsToSkipPerThread;

        // Ok, we want to loop over the outer batches first, the number per warp
        for (int64_t combinedWeightValuesWarpIdx = 0; combinedWeightValuesWarpIdx < numCombinedWeightValuesPerWarp; combinedWeightValuesWarpIdx++)
        {
            // We want to stop generating in this case
            if (combinedWeightValuesWarpIdx + numCombinedWeightValuesPerWarp * blockIdx.x >= numCombinedWeightValuesTotal)
            {
                return;
            }

            // We need a loop over the batches
            for (int64_t combinedWeightValuesThreadIdx = 0; combinedWeightValuesThreadIdx < numCombinedWeightValuesPerThread; combinedWeightValuesThreadIdx++)
            {
                // Ok, now we first want to reset the combined weight stuff
                FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C_Reset(&pPerThreadCombinedWeightValues[blockIdx.x * blockDim.x + threadIdx.x]);

                FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues combinedWeightValue;
                {

                    // This is the perturbation piece.
                    // Can we do this in place, most likely
                    // This will modify pCurrentThreadCurve
                    // Remember, the structure of this is:
                    // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M

                    // Start at the thread's first path idx

                    int64_t numCurvesInBatch = 0;
                    int64_t outputIdx = 0;

                    bool useOptimal = false;
                    const uint32_t numRandom = 1000;
                    const uint32_t numOptimal = 5000;
                    uint32_t countCurrentMethod = 0;

                    for (int64_t pathCount = 0; pathCount < (numToSkip + MaxNumberOfPaths); ++pathCount)
                    {
                        // Do the perturb now
                        // Each time, we first copy the "old path" to the "scratch space"
                        for (uint32_t segIdx = 0; segIdx <= numSegmentsPerCurve; ++segIdx)
                        {
                            pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 0] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 0];
                            pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 1] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 1];
                            pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 2] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 2];
                        }

                        FullExperimentRunnerOptimalPerturbOptimized_GPU::BoundaryConditions_CUDA boundaryConditions_cuda;
                        boundaryConditions_cuda.arclength = boundaryConditions.arclength;

                        boundaryConditions_cuda.m_startPos[0] = boundaryConditions.m_startPos.x;
                        boundaryConditions_cuda.m_startPos[1] = boundaryConditions.m_startPos.y;
                        boundaryConditions_cuda.m_startPos[2] = boundaryConditions.m_startPos.z;

                        boundaryConditions_cuda.m_endPos[0] = boundaryConditions.m_endPos.x;
                        boundaryConditions_cuda.m_endPos[1] = boundaryConditions.m_endPos.y;
                        boundaryConditions_cuda.m_endPos[2] = boundaryConditions.m_endPos.z;

                        boundaryConditions_cuda.m_startDir[0] = boundaryConditions.m_startDir.x;
                        boundaryConditions_cuda.m_startDir[1] = boundaryConditions.m_startDir.y;
                        boundaryConditions_cuda.m_startDir[2] = boundaryConditions.m_startDir.z;

                        boundaryConditions_cuda.m_endDir[0] = boundaryConditions.m_endDir.x;
                        boundaryConditions_cuda.m_endDir[1] = boundaryConditions.m_endDir.y;
                        boundaryConditions_cuda.m_endDir[2] = boundaryConditions.m_endDir.z;

                        twisty::PerturbUtils::UpdateTangentsFromPosCudaSafe(&pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx],
                            &pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx],
                            numSegmentsPerCurve, boundaryConditions);

                        twisty::PerturbUtils::UpdateCurvaturesFromTangentsCudaSafe(&pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx],
                            &pPerGlobalThreadScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions, weightingParams);

                        // Should be 2 - 180
                        int64_t diff = floorf(curand_uniform(&pCurandStates[globalThreadIdx]) * 178.0) + 2;

                        int64_t leftPointIndex = floorf(curand_uniform(&pCurandStates[globalThreadIdx]) * (numSegmentsPerCurve - 1 - diff - 1)) + 1;

                        int64_t rightPointIndex = leftPointIndex + diff;

                        assert((rightPointIndex - leftPointIndex) >= diff);
                        assert(leftPointIndex < rightPointIndex);

                        // We need two frames for each segment to get the new curvature and torsion.
                        // we need the frame left of the segment, as well as the frame right of the segment.
                        // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                        const float leftPoint_x = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 0];
                        const float leftPoint_y = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 1];
                        const float leftPoint_z = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 2];

                        const float rightPoint_x = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 0];
                        const float rightPoint_y = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 1];
                        const float rightPoint_z = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 2];

                        float N_x = (rightPoint_x - leftPoint_x);
                        float N_y = (rightPoint_y - leftPoint_y);
                        float N_z = (rightPoint_z - leftPoint_z);
                        float N_length = sqrt(N_x * N_x + N_y * N_y + N_z * N_z);
                        N_x /= N_length;
                        N_y /= N_length;
                        N_z /= N_length;

                        // Overwrite angle
                        double randRotationAngle = (curand_uniform(&pCurandStates[globalThreadIdx]) * 2.0 - 1.0) * TwistyPi;
                        float N[3] = { N_x, N_y, N_z };

                        // Rotation
                        {
                            float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                            RotationMatrixAroundAxis(randRotationAngle, (float*)(N), rotationMatrix);

                            for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                            {
                                float shiftedPoint[3];
                                shiftedPoint[0] = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 0] - leftPoint_x;
                                shiftedPoint[1] = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 1] - leftPoint_y;
                                shiftedPoint[2] = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 2] - leftPoint_z;

                                // Rotate and stuff back in shifted point
                                RotateVectorByMatrix(rotationMatrix, (float*)(shiftedPoint));
                                // Update the point with the rotated version
                                pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 0] = shiftedPoint[0] + leftPoint_x;
                                pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 1] = shiftedPoint[1] + leftPoint_y;
                                pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 2] = shiftedPoint[2] + leftPoint_z;
                            }

                            //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                            //We can do a different approach later.
                            // Here, we want to do a perturb update call
                            twisty::PerturbUtils::UpdateTangentsFromPosCudaSafe(&pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx],
                                &pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx],
                                numSegmentsPerCurve, boundaryConditions);

                            twisty::PerturbUtils::UpdateCurvaturesFromTangentsCudaSafe(&pPerGlobalThreadScratchSpaceTangents[CurrentThreadTanStartIdx],
                                &pPerGlobalThreadScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                                numSegmentsPerCurve, boundaryConditions, weightingParams);
                        }

                        double pathWeightLog10 = 0.0;// = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CUDA(&(pPerGlobalThreadLeftScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx]),
                            //numSegmentsPerCurve, pLookupTable);

                        // Now we have a candidate path
                        // We perform metropolis and see if we want to accept the path, i.e. copy the scratch space values to the actual curve values, or reroll a new curve
                        {
                            for (uint32_t i = 0; i <= numSegmentsPerCurve; i++)
                            {
                                pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 0] = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 0];
                                pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 1] = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 1];
                                pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 2] = pPerGlobalThreadScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 2];
                            }

                            twisty::PerturbUtils::UpdateTangentsFromPosCudaSafe(&pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx],
                                &pPerGlobalThreadWorkingScratchSpaceTangents[CurrentThreadTanStartIdx],
                                numSegmentsPerCurve, boundaryConditions);

                            twisty::PerturbUtils::UpdateCurvaturesFromTangentsCudaSafe(&pPerGlobalThreadWorkingScratchSpaceTangents[CurrentThreadTanStartIdx],
                                &pPerGlobalThreadWorkingScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                                numSegmentsPerCurve, boundaryConditions, weightingParams);
                        }

                        numPathsAccepted++;
                        if (pathCount < numPathsToSkipPerThread)
                        {
                            // Skip
                        }
                        else
                        {
                            // Else, contribute to the paths
                            FullExperimentRunnerOptimalPerturbOptimized_GPU::CombinedWeightValues_C_AddValue(&pPerThreadCombinedWeightValues[blockIdx.x * blockDim.x + threadIdx.x], pathWeightLog10);
                        }
                    }
                }
            }

            __syncthreads();

            if (threadIdx.x == 0)
            {
                const uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

                for (uint32_t warpThreadIdx = 1; warpThreadIdx < blockDim.x; ++warpThreadIdx)
                {
                    MergeCombinedWeightValues(&pPerThreadCombinedWeightValues[blockIdx.x * blockDim.x + 0], &pPerThreadCombinedWeightValues[blockIdx.x * blockDim.x + warpThreadIdx]);
                }

                // Finally, we write to the combined final values
                (pFinalCombinedValues[blockIdx.x * numCombinedWeightValuesPerWarp + combinedWeightValuesWarpIdx]) = (pPerThreadCombinedWeightValues[blockIdx.x * blockDim.x + 0]);
            }

            __syncthreads();
        }


        //std::cout << "Num path accepted: " << numPathsAccepted << std::endl;
        //std::cout << "Num path unaccepted: " << numPathsUnaccepted << std::endl;
        //std::cout << "Num path unaccepted tangents: " << numPathsUnacceptedTangentPdf << std::endl;
        //std::cout << "Num path unaccepted curvature: " << numPathsUnacceptedCurvaturePdf << std::endl;
    }

}