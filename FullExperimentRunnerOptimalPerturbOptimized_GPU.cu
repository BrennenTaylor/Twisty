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
const uint32_t PerturbGridSize = 20;

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

    bool FullExperimentRunnerOptimalPerturbOptimized_GPU::Setup()
    {
        std::cout << "Random Seeds: " << std::endl;
        std::cout << "\tBootstrap seed: " << m_experimentParams.bootstrapSeed << std::endl;
        std::cout << "\tPerturb seed: " << m_experimentParams.curvePurturbSeed << std::endl;

        // Ask the bootstrapper to generate a discrete curve.
        // If we fail, we want to exit the experiment.
        bool successfulGen = false;
        while (!successfulGen)
        {
            m_upInitialCurve = m_bootstrapper.CreateCurveGeometricSafe(m_experimentParams.numSegmentsPerCurve, m_experimentParams.arclength);
            if (!m_upInitialCurve)
            {
                printf("Both bootstrap versions failed, now we have to error out.\n");
                return false;
            }

            // Lets also get the error of the initial curve, just to know
            float curveError = CurveUtils::CalculateCurveError(*m_upInitialCurve);
            std::cout << "Seed curve error: " << curveError << std::endl;

            if (curveError < m_experimentParams.maximumBootstrapCurveError)
            {
                successfulGen = true;
            }
        }

        const std::filesystem::path experimentDirPath = m_experimentParams.experimentDirPath;
        if (!std::filesystem::exists(experimentDirPath))
        {
            std::filesystem::create_directories(experimentDirPath);
        }

        bool result = SetupCudaDevice();
        if (!result)
        {
            printf("Failed to setup cuda device\n");
            return false;
        }

        return true;
    }

    ExperimentRunner::ExperimentResults FullExperimentRunnerOptimalPerturbOptimized_GPU::RunExperiment()
    {
        int64_t numFailures = 0;
        int64_t totalFailures = 0;
        int64_t totalSuccess = 0;

        /* --------------------- */
        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();
        /* --------------------- */
        auto setupTimeStart = std::chrono::high_resolution_clock::now();
        /* --------------------- */


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

        bool result = true;
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

        std::cout << "Got here - Cuda Setup" << std::endl;

        const double ds = m_upInitialCurve->m_arclength / m_experimentParams.numSegmentsPerCurve;
        twisty::PathWeighting::WeightLookupTableIntegral lookupEvaluator(m_experimentParams.weightingParameters, ds);
        
        twisty::PerturbUtils::BoundaryConditions boundaryConditions;
        boundaryConditions.arclength = m_upInitialCurve->m_arclength;
        boundaryConditions.m_startPos = m_upInitialCurve->m_basePos;
        boundaryConditions.m_startDir = m_upInitialCurve->m_baseTangent;
        boundaryConditions.m_endPos = m_upInitialCurve->m_targetPos;
        boundaryConditions.m_endDir = m_upInitialCurve->m_targetTangent;
        
        // Constants
        double minCurvature = 0.0;
        double maxCurvature = 0.0;
        // TODO: Change this to float rather than double
        twisty::PathWeighting::CalcMinMaxCurvature(minCurvature, maxCurvature, ds);
        const float curvatureStepSize = (maxCurvature - minCurvature) / m_experimentParams.weightingParameters.numCurvatureSteps;

        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        // Setup data structures
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

        twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(initialCurvePositions.data(), initialCurveTangents.data(),
            initialCurveCurvatures.data(), m_upInitialCurve->m_numSegments, boundaryConditions);

        const int64_t NumPosPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumTanPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumCurvaturePerCurve = (m_upInitialCurve->m_numSegments);


        std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> upFN = PathWeighting::NormalizerStuff::GetNormalizer(m_experimentParams.numSegmentsPerCurve);
        PathWeighting::NormalizerStuff::BaseNormalizer& fn = *upFN;

        Farlor::Vector3 Z = (boundaryConditions.m_endPos - boundaryConditions.m_startPos) * (m_upInitialCurve->m_numSegments + 2) / boundaryConditions.arclength
            - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
        std::cout << "Z: " << Z << std::endl;
        std::cout << "|Z|: " << Z.Magnitude() << std::endl;

        PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = PathWeighting::NormalizerStuff::Norm(fn, m_upInitialCurve->m_numSegments,
            Z.Magnitude(), boundaryConditions.arclength);
        std::cout << "PathNormalizer: " << pathNormalizer << std::endl;

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

            FullExperimentRunnerOptimalPerturbOptimized_GPU_GeometryPerturbKernel << <gridSize, blockSize, sharedMemorySizeBytes, stream >> >
                (
                    numCombinedWeightValuesTotal,
                    numCombinedWeightValuesPerWarp,
                    numCombinedWeightValuesPerThread,
                    m_experimentParams.numPathsToSkip,
                    m_experimentParams.numSegmentsPerCurve,
                    m_pPerGlobalThreadRandStates,
                    m_pPerGlobalThreadLeftScratchSpacePositions,
                    m_pPerGlobalThreadRightScratchSpacePositions,
                    m_pPerGlobalThreadWorkingScratchSpacePositions,
                    m_pPerGlobalThreadLeftScratchSpaceTangents,
                    m_pPerGlobalThreadRightScratchSpaceTangents,
                    m_pPerGlobalThreadWorkingScratchSpaceTangents,
                    m_pPerGlobalThreadLeftScratchSpaceCurvatures,
                    m_pPerGlobalThreadRightScratchSpaceCurvatures,
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


        auto runExperimentTimeEnd = std::chrono::high_resolution_clock::now();


        // Cleanup stuff

        {
            CleanupCudaPerturb();
            CleanupCudaRandStates();
            CleanupCudaDevice();
        }




        std::cout << "Experiment Time Reporting: " << std::endl;
        auto runExperimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(runExperimentTimeEnd - runExperimentTimeStart);
        std::cout << "\tTotal Experiment Time: " << runExperimentTimeMs.count() << "ms" << std::endl;

        {
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupTimeEnd - setupTimeStart);
            std::cout << "\tsetup Time: " << timeMs.count() << "ms - " << ((float)timeMs.count() / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tperturb Time: " << perturbTimeCount << "ms - " << ((float)perturbTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tweighting Time: " << weightCalcTimeCount << "ms - " << ((float)weightCalcTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        ExperimentResults results;
        results.experimentWeights.push_back(bigTotalExperimentWeight);
        results.totalPathsGenerated = numCombinedWeightValuesTotal * MaxNumberOfPaths;
        results.numFailedPaths = 0;
        return results;
    }

    void FullExperimentRunnerOptimalPerturbOptimized_GPU::Shutdown()
    {
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

        // Every global thread needs its own curve scratch space left and right and working
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadLeftScratchSpacePositions, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Left Scratch Space Positions");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadRightScratchSpacePositions, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Right Scratch Space Positions");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadWorkingScratchSpacePositions, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Working Scratch Space Positions");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        // Every global thread needs its own curve scratch space left and right and working
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadLeftScratchSpaceTangents, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Left Scratch Space Tangents");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadRightScratchSpaceTangents, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Right Scratch Space Tangents");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadWorkingScratchSpaceTangents, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3)),
            "Cuda malloc Working Scratch Space Tangents");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (3));

        // Every global thread needs its own curve scratch space left and right and working
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadLeftScratchSpaceCurvatures, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1)),
            "Cuda malloc Left Scratch Space Curvatures");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1));

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadRightScratchSpaceCurvatures, sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1)),
            "Cuda malloc Right Scratch Space Curvatures");
        usedMemoryInBytes += (sizeof(float) * numGlobalPerturbThreads * m_upInitialCurve->m_numSegments * (1));

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

        twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(initialCurvePositions.data(), initialCurveTangents.data(),
            initialCurveCurvatures.data(), m_upInitialCurve->m_numSegments, boundaryConditions);

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
                cudaMemcpy((void*)&m_pPerGlobalThreadLeftScratchSpacePositions[idx], (void*)initialCurvePositions.data(), initialCurvePositions.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadRightScratchSpacePositions[idx], (void*)initialCurvePositions.data(), initialCurvePositions.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadWorkingScratchSpacePositions[idx], (void*)initialCurvePositions.data(), initialCurvePositions.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
            }
            idx += initialCurvePositions.size() * 3;
        }

        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx)
        {
            // Copy Tan
            for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                cudaMemcpy((void*)&m_pPerGlobalThreadLeftScratchSpaceTangents[idx], (void*)initialCurveTangents.data(), initialCurveTangents.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadRightScratchSpaceTangents[idx], (void*)initialCurveTangents.data(), initialCurveTangents.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadWorkingScratchSpaceTangents[idx], (void*)initialCurveTangents.data(), initialCurveTangents.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);
            }
            idx += initialCurveTangents.size() * 3;
        }

        for (int64_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx)
        {
            // Copy Curvatures
            for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
            {
                cudaMemcpy((void*)&m_pPerGlobalThreadLeftScratchSpaceCurvatures[idx], (void*)initialCurveCurvatures.data(), initialCurveCurvatures.size() * sizeof(float) * 1, cudaMemcpyHostToDevice);
                cudaMemcpy((void*)&m_pPerGlobalThreadRightScratchSpaceCurvatures[idx], (void*)initialCurveCurvatures.data(), initialCurveCurvatures.size() * sizeof(float) * 1, cudaMemcpyHostToDevice);
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

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadRightScratchSpaceCurvatures),
            "Cuda free Right Scratch Space Curvatures");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadLeftScratchSpaceCurvatures),
            "Cuda free Left Scratch Space Curvatures");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadWorkingScratchSpaceTangents),
            "Cuda free Working Scratch Space Tangents");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadRightScratchSpaceTangents),
            "Cuda free Right Scratch Space Tangents");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadLeftScratchSpaceTangents),
            "Cuda free Left Scratch Space Tangents");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadWorkingScratchSpacePositions),
            "Cuda free Working Scratch Space Positions");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadRightScratchSpacePositions),
            "Cuda free Right Scratch Space Positions");

        CudaSafeErrorCheck(cudaFree((void*)m_pPerGlobalThreadLeftScratchSpacePositions),
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


    __device__ void RecalculateTangentsCurvaturesFromPos_CUDA(float* pPositions, float* pTangents, float* pCurvatures, uint32_t numSegmentsPerCurve,
        FullExperimentRunnerOptimalPerturbOptimized_GPU::BoundaryConditions_CUDA boundaryConditions)
    {
        const float ds = boundaryConditions.arclength / numSegmentsPerCurve;

        // Set initial and final positions
        pPositions[0 * 3 + 0] = boundaryConditions.m_startPos[0];
        pPositions[0 * 3 + 1] = boundaryConditions.m_startPos[1];
        pPositions[0 * 3 + 2] = boundaryConditions.m_startPos[2];

        pPositions[1 * 3 + 0] = pPositions[0 * 3 + 0] + ds * boundaryConditions.m_startDir[0];
        pPositions[1 * 3 + 1] = pPositions[0 * 3 + 1] + ds * boundaryConditions.m_startDir[1];
        pPositions[1 * 3 + 2] = pPositions[0 * 3 + 2] + ds * boundaryConditions.m_startDir[2];


        pPositions[numSegmentsPerCurve * 3 + 0] = boundaryConditions.m_endPos[0];
        pPositions[numSegmentsPerCurve * 3 + 1] = boundaryConditions.m_endPos[1];
        pPositions[numSegmentsPerCurve * 3 + 2] = boundaryConditions.m_endPos[2];

        // Update tangents
        // Set first tangent directly, defined by boundary conditions
        pTangents[0 * 3 + 0] = boundaryConditions.m_startDir[0];
        pTangents[0 * 3 + 1] = boundaryConditions.m_startDir[1];
        pTangents[0 * 3 + 2] = boundaryConditions.m_startDir[2];

        // Set others via finite difference
        for (uint32_t i = 1; i < numSegmentsPerCurve; ++i)
        {
            float diff_x = pPositions[(i + 1) * 3 + 0] - pPositions[(i - 1) * 3 + 0];
            float diff_y = pPositions[(i + 1) * 3 + 1] - pPositions[(i - 1) * 3 + 1];
            float diff_z = pPositions[(i + 1) * 3 + 2] - pPositions[(i - 1) * 3 + 2];
            float diff_length = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
            pTangents[i * 3 + 0] = diff_x / diff_length;
            pTangents[i * 3 + 1] = diff_y / diff_length;
            pTangents[i * 3 + 2] = diff_z / diff_length;
        }
        pTangents[numSegmentsPerCurve * 3 + 0] = boundaryConditions.m_endDir[0];
        pTangents[numSegmentsPerCurve * 3 + 1] = boundaryConditions.m_endDir[1];
        pTangents[numSegmentsPerCurve * 3 + 2] = boundaryConditions.m_endDir[2];

        // Calculate curvature
        // First, we calcualte only the first using the old method
        {
            float denom = 1.0f / ds;
            float diff_x = (pTangents[1 * 3 + 0] - pTangents[0 * 3 + 0]) * denom;
            float diff_y = (pTangents[1 * 3 + 1] - pTangents[0 * 3 + 1]) * denom;
            float diff_z = (pTangents[1 * 3 + 2] - pTangents[0 * 3 + 2]) * denom;
            float diff_length = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);
            {
                const float curvature = diff_length;
                pCurvatures[0] = curvature;
            }
        }

        // All the rest we calculate using the new method
        for (uint32_t i = 1; i < numSegmentsPerCurve; ++i)
        {
            // First, grab tangent
            float tan_x = pTangents[i * 3 + 0];
            float tan_y = pTangents[i * 3 + 1];
            float tan_z = pTangents[i * 3 + 2];

            // Second, calculate dp2ds2
            float dp2ds2_x = pPositions[(i + 1) * 3 + 0] + pPositions[(i - 1) * 3 + 0] - 2.0f * pPositions[i * 3 + 0];
            float dp2ds2_y = pPositions[(i + 1) * 3 + 1] + pPositions[(i - 1) * 3 + 1] - 2.0f * pPositions[i * 3 + 1];
            float dp2ds2_z = pPositions[(i + 1) * 3 + 2] + pPositions[(i - 1) * 3 + 2] - 2.0f * pPositions[i * 3 + 2];
            dp2ds2_x *= (1.0f / (ds * ds));
            dp2ds2_y *= (1.0f / (ds * ds));
            dp2ds2_z *= (1.0f / (ds * ds));

            // Third, calculate dTds
            float tanDotdp2ds2 = tan_x * dp2ds2_x + tan_y * dp2ds2_y + tan_z * dp2ds2_z;

            float dTds_x = dp2ds2_x - tan_x * tanDotdp2ds2;
            float dTds_y = dp2ds2_y - tan_y * tanDotdp2ds2;
            float dTds_z = dp2ds2_z - tan_z * tanDotdp2ds2;
            pCurvatures[i] = sqrt(dTds_x * dTds_x + dTds_y * dTds_y + dTds_z * dTds_z);
        }
    }

    __device__ double WeightCurveViaCurvatureLog10_CUDA(float* pCurvatureStart, uint32_t numCurvatures, double* pWeightIntegral, double ds,
        twisty::WeightingParameters weightingParams_cuda)
    {
        return 0.0;
        //if (!pCurvatureStart || (numCurvatures == 0))
        //{
        //    return 0.0;
        //}

        ////double ds = weightIntegral.GetDs();
        //const auto& weightingParams = weightIntegral.GetWeightingParams();
        //double minCurvature = 0.0;
        //double maxCurvature = 0.0;
        //twisty::PathWeighting::CalcMinMaxCurvature(minCurvature, maxCurvature, ds);
        //const double curvatureStepSize = (maxCurvature - minCurvature) / weightingParams.numCurvatureSteps;
        //auto& lookupTable = weightIntegral.AccessLookupTable();

        //const float c = weightingParams.scatter + weightingParams.absorbtion;
        //const float absorbtionConst = std::exp(-c * ds) / (2.0 * TwistyPi * TwistyPi);
        //const float absorbtionConstLog10 = std::log10(absorbtionConst);

        //// Calculate value
        //double runningPathWeightLog10 = 0.0;
        //for (int64_t segIdx = 0; segIdx < numCurvatures; ++segIdx)
        //{
        //    // Extract curvature
        //    double curvature = pCurvatureStart[segIdx];
        //    double distance = curvature - minCurvature;
        //    double realIdx = distance / curvatureStepSize;
        //    int64_t leftIdx = floor(realIdx);
        //    int64_t rightIdx = leftIdx + 1;

        //    double leftLookup = lookupTable[leftIdx];
        //    double rightLookup = lookupTable[rightIdx];

        //    double leftDist = distance - (leftIdx * curvatureStepSize);

        //    double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
        //    // Take the natural log of the interpolated results
        //    double interpolatedResultLog10 = std::log10(interpolatedResult);
        //    // Lets do weights as doubles for now
        //    double segmentWeightLog10 = interpolatedResultLog10;

        //    // Take natural log of this constant
        //    segmentWeightLog10 += absorbtionConstLog10;

        //    // Update the running path weight. We also want to cache the segment weights
        //    runningPathWeightLog10 += segmentWeightLog10;
        //}
        //return runningPathWeightLog10;
    }

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
                            pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 0] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 0];
                            pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 1] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 1];
                            pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 2] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 2];

                            pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 0] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 0];
                            pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 1] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 1];
                            pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 2] = pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + segIdx * 3 + 2];
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

                        RecalculateTangentsCurvaturesFromPos_CUDA(&pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx],
                            &pPerGlobalThreadLeftScratchSpaceTangents[CurrentThreadTanStartIdx], &pPerGlobalThreadLeftScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions_cuda);

                        RecalculateTangentsCurvaturesFromPos_CUDA(&pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx],
                            &pPerGlobalThreadRightScratchSpaceTangents[CurrentThreadTanStartIdx], &pPerGlobalThreadRightScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions_cuda);

#ifdef HardcodedSegments
                        int64_t leftPointIndex = 25;
                        int64_t rightPointIndex = 75;
#else

#if defined(HardcodedDifference)
                        int64_t diff = 20;
#else
                        // Should be 2 - 180
                        int64_t diff = floorf(curand_uniform(&pCurandStates[globalThreadIdx]) * 178.0) + 2;
#endif

                        int64_t leftPointIndex = floorf(curand_uniform(&pCurandStates[globalThreadIdx]) * (numSegmentsPerCurve - 1 - diff - 1)) + 1;

                        int64_t rightPointIndex = leftPointIndex + diff;

                        assert((rightPointIndex - leftPointIndex) >= diff);
#endif
                        assert(leftPointIndex < rightPointIndex);

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                        {
                            printf("Left point idx: %d\n", leftPointIndex);
                            printf("Right point idx: %d\n", rightPointIndex);
                        }
#endif
                        // We need two frames for each segment to get the new curvature and torsion.
                        // we need the frame left of the segment, as well as the frame right of the segment.
                        // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                        const float leftPoint_x = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 0];
                        const float leftPoint_y = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 1];
                        const float leftPoint_z = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 2];

                        const float rightPoint_x = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 0];
                        const float rightPoint_y = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 1];
                        const float rightPoint_z = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 2];

                        float N_x = (rightPoint_x - leftPoint_x);
                        float N_y = (rightPoint_y - leftPoint_y);
                        float N_z = (rightPoint_z - leftPoint_z);
                        float N_length = sqrt(N_x * N_x + N_y * N_y + N_z * N_z);
                        N_x /= N_length;
                        N_y /= N_length;
                        N_z /= N_length;

                        double leftRotationAngle = 0.0;
                        {
#if defined(DetailedPurturb) && defined(SingleThreadMode)
                            printf("Axis before (%.6f, %.6f, %.6f)\n",
                                N[0], N[1], N[2]
                            );
#endif

                            const float Xss1_x = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + (leftPointIndex - 1) * 3 + 0];
                            const float Xss1_y = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + (leftPointIndex - 1) * 3 + 2];
                            const float Xss1_z = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + (leftPointIndex - 1) * 3 + 3];

                            const float Xs_x = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 0];
                            const float Xs_y = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 1];
                            const float Xs_z = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + leftPointIndex * 3 + 2];

                            const float Xsp1_x = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + (leftPointIndex + 1) * 3 + 0];
                            const float Xsp1_y = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + (leftPointIndex + 1) * 3 + 1];
                            const float Xsp1_z = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + (leftPointIndex + 1) * 3 + 2];

                            const float Xsp1mXsDotN = (Xsp1_x - Xs_x) * N_x + (Xsp1_y - Xs_y) * N_y + (Xsp1_z - Xs_z) * N_z;

                            const float PL_x = Xs_x + Xsp1mXsDotN * N_x;
                            const float PL_y = Xs_y + Xsp1mXsDotN * N_y;
                            const float PL_z = Xs_z + Xsp1mXsDotN * N_z;

                            const float Xss1mPLDotN = (Xss1_x - PL_x) * N_x + (Xss1_y - PL_y) * N_y + (Xss1_z - PL_z) * N_z;
                            const float ZL_x = Xss1_x - Xss1mPLDotN * N_x;
                            const float ZL_y = Xss1_y - Xss1mPLDotN * N_y;
                            const float ZL_z = Xss1_z - Xss1mPLDotN * N_z;

                            // Get side of plane Z is on

                            float Xsp1mPL_x = Xsp1_x - PL_x;
                            float Xsp1mPL_y = Xsp1_y - PL_y;
                            float Xsp1mPL_z = Xsp1_z - PL_z;
                            float Xsp1mPL_length = sqrt(Xsp1mPL_x * Xsp1mPL_x + Xsp1mPL_y * Xsp1mPL_y + Xsp1mPL_z * Xsp1mPL_z);
                            Xsp1mPL_x /= Xsp1mPL_length;
                            Xsp1mPL_y /= Xsp1mPL_length;
                            Xsp1mPL_z /= Xsp1mPL_length;

                            float NL_x = (N_y * Xsp1mPL_z - N_z * Xsp1mPL_y);
                            float NL_y = (N_x * Xsp1mPL_z - N_z * Xsp1mPL_x);
                            float NL_z = (N_x * Xsp1mPL_y - N_y * Xsp1mPL_x);
                            float NL_length = sqrt(NL_x * NL_x + NL_y * NL_y + NL_z * NL_z);
                            NL_x /= NL_length;
                            NL_y /= NL_length;
                            NL_z /= NL_length;

                            const float sideDistL = ((ZL_x - PL_x) * (NL_x)) + ((ZL_y - PL_y) * (NL_y)) + ((ZL_z - PL_z) * (NL_z));

                            float ZPnorm_x = ZL_x - PL_x;
                            float ZPnorm_y = ZL_y - PL_y;
                            float ZPnorm_z = ZL_z - PL_z;
                            float ZPnorm_length = sqrt(ZPnorm_x * ZPnorm_x + ZPnorm_y * ZPnorm_y + ZPnorm_z * ZPnorm_z);
                            ZPnorm_x /= ZPnorm_length;
                            ZPnorm_y /= ZPnorm_length;
                            ZPnorm_z /= ZPnorm_length;

                            double cosAngle = (ZPnorm_x * Xsp1mPL_x) + (ZPnorm_y * Xsp1mPL_y) + (ZPnorm_z * Xsp1mPL_z);
                            cosAngle = max(-1.0, cosAngle);
                            cosAngle = min(1.0, cosAngle);
                            const double angle = acos(cosAngle);

                            const double threshold = 10e-10;

                            // In the case, we are aligned, we want a Pi rotation
                            if (abs(angle) <= threshold)
                            {
                                leftRotationAngle = TwistyPi;
                            }
                            // If we are Pi away, we want no rotation
                            else if (abs(abs(angle) - TwistyPi) < threshold)
                            {
                                leftRotationAngle = 0.0;
                            }
                            // On back side
                            else if (sideDistL < 0.0)
                            {
                                leftRotationAngle = TwistyPi - angle;
                            }
                            // On front side
                            else
                            {
                                leftRotationAngle = -1.0 * (TwistyPi - angle);
                            }
                        }


                        double rightRotationAngle = 0.0;
                        {
                            const float Xes1_x = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + (rightPointIndex - 1) * 3 + 0];
                            const float Xes1_y = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + (rightPointIndex - 1) * 3 + 2];
                            const float Xes1_z = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + (rightPointIndex - 1) * 3 + 3];

                            const float Xe_x = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 0];
                            const float Xe_y = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 1];
                            const float Xe_z = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + rightPointIndex * 3 + 2];

                            const float Xep1_x = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + (rightPointIndex + 1) * 3 + 0];
                            const float Xep1_y = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + (rightPointIndex + 1) * 3 + 1];
                            const float Xep1_z = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + (rightPointIndex + 1) * 3 + 2];


                            const float Xes1mXeDotN = (Xep1_x - Xe_x) * N_x + (Xep1_y - Xe_y) * N_y + (Xep1_z - Xe_z) * N_z;

                            const float PR_x = Xe_x + Xes1mXeDotN * N_x;
                            const float PR_y = Xe_y + Xes1mXeDotN * N_y;
                            const float PR_z = Xe_z + Xes1mXeDotN * N_z;

                            const float Xep1mPRDotN = (Xep1_x - PR_x) * N_x + (Xep1_y - PR_y) * N_y + (Xep1_z - PR_z) * N_z;
                            const float ZL_x = Xep1_x - Xep1mPRDotN * N_x;
                            const float ZL_y = Xep1_y - Xep1mPRDotN * N_y;
                            const float ZL_z = Xep1_z - Xep1mPRDotN * N_z;

                            // Get side of plane Z is on

                            float Xes1mPR_x = Xes1_x - PR_x;
                            float Xes1mPR_y = Xes1_y - PR_y;
                            float Xes1mPR_z = Xes1_z - PR_z;
                            float Xes1mPR_length = sqrt(Xes1mPR_x * Xes1mPR_x + Xes1mPR_y * Xes1mPR_y + Xes1mPR_z * Xes1mPR_z);
                            Xes1mPR_x /= Xes1mPR_length;
                            Xes1mPR_y /= Xes1mPR_length;
                            Xes1mPR_z /= Xes1mPR_length;

                            float NR_x = (N_y * Xes1mPR_z - N_z * Xes1mPR_y);
                            float NR_y = (N_x * Xes1mPR_z - N_z * Xes1mPR_x);
                            float NR_z = (N_x * Xes1mPR_y - N_y * Xes1mPR_x);
                            float NR_length = sqrt(NR_x * NR_x + NR_y * NR_y + NR_z * NR_z);
                            NR_x /= NR_length;
                            NR_y /= NR_length;
                            NR_z /= NR_length;

                            const float sideDistR = ((ZL_x - PR_x) * (NR_x)) + ((ZL_y - PR_y) * (NR_y)) + ((ZL_z - PR_z) * (NR_z));

                            float ZPnorm_x = ZL_x - PR_x;
                            float ZPnorm_y = ZL_y - PR_y;
                            float ZPnorm_z = ZL_z - PR_z;
                            float ZPnorm_length = sqrt(ZPnorm_x * ZPnorm_x + ZPnorm_y * ZPnorm_y + ZPnorm_z * ZPnorm_z);
                            ZPnorm_x /= ZPnorm_length;
                            ZPnorm_y /= ZPnorm_length;
                            ZPnorm_z /= ZPnorm_length;

                            double cosAngle = (ZPnorm_x * Xes1mPR_x) + (ZPnorm_y * Xes1mPR_y) + (ZPnorm_z * Xes1mPR_z);
                            cosAngle = max(-1.0, cosAngle);
                            cosAngle = min(1.0, cosAngle);
                            const double angle = acos(cosAngle);

                            const double threshold = 10e-10;

                            // In the case, we are aligned, we want a Pi rotation
                            if (abs(angle) <= threshold)
                            {
                                rightRotationAngle = TwistyPi;
                            }
                            // If we are Pi away, we want no rotation
                            else if (abs(abs(angle) - TwistyPi) < threshold)
                            {
                                rightRotationAngle = 0.0;
                            }
                            // On back side
                            else if (sideDistR < 0.0)
                            {
                                rightRotationAngle = TwistyPi - angle;
                            }
                            // On front side
                            else
                            {
                                rightRotationAngle = -1.0 * (TwistyPi - angle);
                            }
                        }

                        // Overwrite angle
                        if (!useOptimal)
                        {
                            countCurrentMethod++;
                            if (countCurrentMethod >= numRandom)
                            {
                                useOptimal = !useOptimal;
                            }
                            double randRotationAngle = (curand_uniform(&pCurandStates[globalThreadIdx]) * 2.0 - 1.0) * TwistyPi;
                            leftRotationAngle = randRotationAngle;
                            rightRotationAngle = randRotationAngle;
                        }
                        else
                        {
                            countCurrentMethod++;
                            if (countCurrentMethod >= numOptimal)
                            {
                                useOptimal = !useOptimal;
                            }
                        }

                        float N[3] = { N_x, N_y, N_z };

                        // Left Rotation
                        {
                            float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                            RotationMatrixAroundAxis(leftRotationAngle, (float*)(N), rotationMatrix);

                            for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                            {
                                float shiftedPoint[3];
                                shiftedPoint[0] = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 0] - leftPoint_x;
                                shiftedPoint[1] = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 1] - leftPoint_y;
                                shiftedPoint[2] = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 2] - leftPoint_z;

                                // Rotate and stuff back in shifted point
                                RotateVectorByMatrix(rotationMatrix, (float*)(shiftedPoint));
                                // Update the point with the rotated version
                                pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 0] = shiftedPoint[0] + leftPoint_x;
                                pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 1] = shiftedPoint[1] + leftPoint_y;
                                pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 2] = shiftedPoint[2] + leftPoint_z;
                            }

                            //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                            //We can do a different approach later.
                            // Here, we want to do a perturb update call
                            RecalculateTangentsCurvaturesFromPos_CUDA(&pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx],
                                &pPerGlobalThreadLeftScratchSpaceTangents[CurrentThreadTanStartIdx], &pPerGlobalThreadLeftScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                                numSegmentsPerCurve, boundaryConditions_cuda);
                        }

                        // Right Rotation
                        {
                            float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                            RotationMatrixAroundAxis(rightRotationAngle, (float*)(&N), rotationMatrix);

                            for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                            {
                                float shiftedPoint[3];
                                shiftedPoint[0] = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 0] - leftPoint_x;
                                shiftedPoint[1] = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 1] - leftPoint_y;
                                shiftedPoint[2] = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 2] - leftPoint_z;

                                // Rotate and stuff back in shifted point
                                RotateVectorByMatrix(rotationMatrix, (float*)(shiftedPoint));
                                // Update the point with the rotated version
                                pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 0] = shiftedPoint[0] + leftPoint_x;
                                pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 1] = shiftedPoint[1] + leftPoint_y;
                                pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + pointIdx * 3 + 2] = shiftedPoint[2] + leftPoint_z;
                            }

                            //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                            //We can do a different approach later.
                            // Here, we want to do a perturb update call
                            RecalculateTangentsCurvaturesFromPos_CUDA(&pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx],
                                &pPerGlobalThreadRightScratchSpaceTangents[CurrentThreadTanStartIdx], &pPerGlobalThreadRightScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                                numSegmentsPerCurve, boundaryConditions_cuda);
                        }

                        double leftPathWeightLog10 = 0.0;// = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CUDA(&(pPerGlobalThreadLeftScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx]),
                            //numSegmentsPerCurve, pLookupTable);

                        double rightPathWeightLog10 = 0.0;// = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CUDA(&(pPerGlobalThreadRightScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx]),
                            //numSegmentsPerCurve, pLookupTable);

                        bool useLeftRotation = (leftPathWeightLog10 >= rightPathWeightLog10);
                        double pathWeightLog10 = useLeftRotation ? leftPathWeightLog10 : rightPathWeightLog10;

                        // Now we have a candidate path
                        // We perform metropolis and see if we want to accept the path, i.e. copy the scratch space values to the actual curve values, or reroll a new curve
                        {
                            for (uint32_t i = 0; i <= numSegmentsPerCurve; i++)
                            {
                                if (useLeftRotation)
                                {
                                    pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 0] = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 0];
                                    pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 1] = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 1];
                                    pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 2] = pPerGlobalThreadLeftScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 2];
                                }
                                else
                                {
                                    pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 0] = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 0];
                                    pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 1] = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 1];
                                    pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 2] = pPerGlobalThreadRightScratchSpacePositions[CurrentThreadPosStartIdx + i * 3 + 2];
                                }
                            }

                            RecalculateTangentsCurvaturesFromPos_CUDA(&pPerGlobalThreadWorkingScratchSpacePositions[CurrentThreadPosStartIdx],
                                &pPerGlobalThreadWorkingScratchSpaceTangents[CurrentThreadTanStartIdx], &pPerGlobalThreadWorkingScratchSpaceCurvatures[CurrentThreadCurvatureStartIdx],
                                numSegmentsPerCurve, boundaryConditions_cuda);
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