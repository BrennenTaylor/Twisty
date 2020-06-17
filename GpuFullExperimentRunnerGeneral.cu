#include "GpuFullExperimentRunnerGeneral.h"

#include "CurvePerturbUtils.h"

#include "CurveUtils.h"
#include "MathConsts.h"

#include "Twisty_Cuda_Helpers.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#include <omp.h>

#include <assert.h>
#include <ctime>
#include <fstream>
#include <limits>

#include <chrono>
#include <thread>

struct GpuVec3
{
    float x;
    float y;
    float z;
};

//#define DetailedPurturb
//#define SingleThreadDebugMode

/*

    In this, we allocate as follows:

    1. Each thread gets its own copy of the initial seeed curve. This is the same for all threads.
    2. Each thread then purturbs into a different curve each time. These all diverge, but run the same algorithm, so performance should be ok.
    3. After each perturb, we weight and generate a single double value for each path. These are then returned to the CPU.

*/

// Cuda Functions
namespace twisty
{
    // Dispatch kernel for initializing the curand states
    // This really should be generalized or better thought out for experimentation sake
    __global__ void InitializeCurandState(uint32_t seed, curandState_t* pStates, uint32_t maxNumStates)
    {
        // Unique index
        const uint32_t globalThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
        if (globalThreadIdx < maxNumStates)
        {
            curand_init(seed + globalThreadIdx, 0, 0, &pStates[globalThreadIdx]);
        }
    }

    // Dispatch which atually runs the purtibation algorithm on the GPU
    __global__ void GeneratePathBatchPutrubations(
        uint32_t numExperimentPaths,
        uint32_t numPathsPerThread,
        uint32_t numPathsToSkipPerThread,
        uint32_t numSegmentsPerCurve,
        curandState_t* pRandStates, 
        float* pPerThreadPositions,
        float* pPerThreadTangents,
        float* pPerThreadCurvatures,
        double* pCachedSegmentWeights,
        double* pCompressedPathWeights,
        float segmentLength,
        float scattering,
        float absorbtion,
        double* pLookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize
    )
    {

        // First, we calculate the block thread index of the entire gpu dispatch.
        // For the number of threads dispatched, each is responsible for a number of 
        // paths to generate, i.e. numExperimentThreads / numGlobalThreads
        uint32_t globalThreadIdx = threadIdx.x + blockDim.x * blockIdx.x;

        uint32_t currentPathIdx = numPathsPerThread * globalThreadIdx;
        if (currentPathIdx >= numExperimentPaths)
        {
            //printf("Not executing thread idx: %d\n", globalThreadIdx);

            // We dont want to continue if we have already generated the correct number of paths.
            return;
        }

        if (globalThreadIdx == 0)
        {
            printf("Thread Idx 0 called\n");
        }

        // The current thread is stored at the beginning
        // We want to index into shared memory via threadIdx.x as this is assigned per block
        const uint32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
        const uint32_t NumTanPerCurve = (numSegmentsPerCurve + 1);
        const uint32_t NumCurvaturesPerCurve = numSegmentsPerCurve;

        // 3 floats per pos and tan
        const uint32_t CurrentThreadPosStartIdx = 3 *  NumPosPerCurve * globalThreadIdx;
        const uint32_t CurrentThreadTanStartIdx = 3 * NumTanPerCurve * globalThreadIdx;
        const uint32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * globalThreadIdx;

        float c = scattering + absorbtion;
        float absorbtionConst = std::exp(-c * segmentLength) / (2.0 * CUDART_PI_F * CUDART_PI_F);
        float lnAbsorbtionConst = log(absorbtionConst);


        double runningPathWeight = 0.0;
        // Lets precache all the segment weights
        {
            for (uint32_t segIdx = 0; segIdx < numSegmentsPerCurve; ++segIdx)
            {
                float curvature = pPerThreadCurvatures[CurrentThreadCurvatureStartIdx + segIdx];

                float distance = curvature - minCurvature;
                float realIdx = distance / curvatureStepSize;
                uint32_t leftIdx = floor(realIdx);
                uint32_t rightIdx = leftIdx + 1;

                float leftLookup = pLookupTable[leftIdx];
                float rightLookup = pLookupTable[rightIdx];
                float leftDist = distance - (leftIdx * curvatureStepSize);
                double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                double interpolatedResultLog = log(interpolatedResult);
                double segmentWeight = interpolatedResultLog;

                segmentWeight += lnAbsorbtionConst;

                // Update the running path weight. We also want to cache the segment weights
                runningPathWeight += segmentWeight;
                pCachedSegmentWeights[segIdx + (numSegmentsPerCurve * globalThreadIdx)] = segmentWeight;
                //if (globalThreadIdx == 0)
                //{
                //    printf("Segment %d cached weight: %0.6f\n", segIdx, segmentWeight);
                //}
            }

#if defined(SingleThreadDebugMode)
            {
                printf("Cached Weights:\n");

                for (uint32_t segIdx = 0; segIdx < numSegmentsPerCurve; segIdx++)
                {
                    printf("\tCached Weight: <%0.6f>\n", pCachedSegmentWeights[segIdx + numSegmentsPerCurve * globalThreadIdx]);
                }
            }
#endif
        }

        if (globalThreadIdx == 0)
        {
            printf("Running path weight after cache: %0.6f\n", runningPathWeight);
        }

        // Now, we can begin the actual algorithm
        {
            // This is the perturbation piece.
            // Can we do this in place

            uint32_t numCurvesInBatch = 0;
            uint32_t outputIdx = 0;

            int32_t cacheStartPathIdx = numPathsPerThread * globalThreadIdx;

            for (int32_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
            {
                if (globalThreadIdx == 0)
                {
                    printf("Path count: %d\n", pathCount);
                }

                // Start at the thread's first path idx
                int32_t currentPathIdx = numPathsPerThread * globalThreadIdx + pathCount - numPathsToSkipPerThread;
                if (currentPathIdx >= numExperimentPaths)
                {
#ifdef BlockingOutputThread
                    {
                        std::scoped_lock<std::mutex> lock(outputThreadMutex);
                        std::cout << "Returning, all paths complete" << std::endl;
                    }
#endif

#if defined(ExportPathBatches)
                    if (numCurvesInBatch > 0)
                    {
                        ExportPathBatchesMutex.lock();

                        if (threadIdx == 11)
                        {
                            std::cout << "Should be exporting thread 12" << std::endl;
                        }


                        curvesMetadataFile << threadIdx << " ";
                        curvesMetadataFile << outputIdx << " ";
                        curvesMetadataFile << numCurvesInBatch << std::endl;

                        curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                        numCurvesInBatch = 0;
                        outputIdx++;

                        ExportPathBatchesMutex.unlock();
                    }
#endif


                    // We dont want to continue if we have already generated the correct number of paths.
                    break;
                }

                if (globalThreadIdx == 0)
                {
                    printf("Beginning perturb of path %d\n", pathCount);
                }

                // Do the perturb now
                {
                    // We bound on left by one as we dont want to rotate the first segment at all
                    // Left bound by m-2 as we at least want there to be one point between the left and right points selected so an actual perturbation occurs
                    float leftPtRand = curand_uniform(&pRandStates[globalThreadIdx]);
                    float rightPtRand = curand_uniform(&pRandStates[globalThreadIdx]);
                    unsigned int leftPointIndex = floorf(leftPtRand * ((numSegmentsPerCurve - 3) - 1) + 1);
                    unsigned int rightPointIndex = floorf(leftPtRand * ((numSegmentsPerCurve - 1) - (leftPointIndex + 2)) + (leftPointIndex + 2));

                    //unsigned int leftPointIndex = 17;
                    //unsigned int rightPointIndex = 39;

#if defined(SingleThreadDebugMode)
                    {
                        printf("Left point idx: %d\n", leftPointIndex);
                        printf("Right point idx: %d\n", rightPointIndex);
                    }
#endif

                    assert(leftPointIndex < rightPointIndex);
                    assert((rightPointIndex - leftPointIndex) >= 2);

                    // We need two frames for each segment to get the new curvature and torsion.
                    // we need the frame left of the segment, as well as the frame right of the segment.

                    // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                    float* pLeftPoint = pPerThreadPositions + CurrentThreadPosStartIdx  + PositionFloatCount * leftPointIndex;
                    float* pRightPoint = pPerThreadPositions + CurrentThreadPosStartIdx + PositionFloatCount * rightPointIndex;
                    
#if defined(SingleThreadDebugMode)
                    {
                        printf("Left Point: (%.6f, %.6f, %.6f)\n", pLeftPoint[0], pLeftPoint[1], pLeftPoint[2]);
                        printf("Right Point: (%.6f, %.6f, %.6f)\n", pRightPoint[0], pRightPoint[1], pRightPoint[2]);
                    }
#endif

                    float axisOfRotation[3];
                    axisOfRotation[0] = pRightPoint[0] - pLeftPoint[0];
                    axisOfRotation[1] = pRightPoint[1] - pLeftPoint[1];
                    axisOfRotation[2] = pRightPoint[2] - pLeftPoint[2];

                    NormalizeVector3f(axisOfRotation);


#if defined(SingleThreadDebugMode)
                    {
                        printf("Axis of rotation: (%.6f, %.6f, %.6f)\n", axisOfRotation[0], axisOfRotation[1], axisOfRotation[2]);
                    }
#endif

                    float randomAngle = (curand_uniform(&pRandStates[globalThreadIdx]) * 360.0f) - 180.0f;

#if defined(SingleThreadDebugMode)
                    {
                        printf("randomAngle: %.6f\n", randomAngle);
                    }
#endif


                    float rotationMatrix[9];
                    RotationMatrixAroundAxis(randomAngle, axisOfRotation, rotationMatrix);
#if defined(SingleThreadDebugMode)
                    {
                        printf("Rotation Matrix\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n",
                            rotationMatrix[0], rotationMatrix[1], rotationMatrix[2],
                            rotationMatrix[3], rotationMatrix[4], rotationMatrix[5],
                            rotationMatrix[6], rotationMatrix[7], rotationMatrix[8]
                        );
                    }
#endif

                    uint32_t numChanged = 0;
                    for (uint32_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                    {
                        numChanged++;
                        float* pCurrentPoint = pPerThreadPositions + CurrentThreadPosStartIdx + PositionFloatCount * pointIdx;

                        float shiftedPoint[3];
                        shiftedPoint[0] = pCurrentPoint[0] - pLeftPoint[0];
                        shiftedPoint[1] = pCurrentPoint[1] - pLeftPoint[1];
                        shiftedPoint[2] = pCurrentPoint[2] - pLeftPoint[2];
                        // Rotate and stuff back in shifted point
                        RotateVectorByMatrix(rotationMatrix, shiftedPoint);

                        // Update the point with the rotated version
                        pCurrentPoint[0] = shiftedPoint[0] + pLeftPoint[0];
                        pCurrentPoint[1] = shiftedPoint[1] + pLeftPoint[1];
                        pCurrentPoint[2] = shiftedPoint[2] + pLeftPoint[2];
                    }

                    //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                    //We can do a different approach later.

                    // Left side
                    {
                        float* pLeftPointTanCalc = pPerThreadPositions + CurrentThreadPosStartIdx + PositionFloatCount * leftPointIndex;
                        float* pRightPointTanCalc = pPerThreadPositions + CurrentThreadPosStartIdx + PositionFloatCount * (leftPointIndex + 1);

                        float* pCurrentTan = pPerThreadTangents + CurrentThreadTanStartIdx + TangentFloatCount * leftPointIndex;

                        pCurrentTan[0] = pRightPointTanCalc[0] - pLeftPointTanCalc[0];
                        pCurrentTan[1] = pRightPointTanCalc[1] - pLeftPointTanCalc[1];
                        pCurrentTan[2] = pRightPointTanCalc[2] - pLeftPointTanCalc[2];

                        NormalizeVector3f(pCurrentTan);
                    }

                    // Right side
                    {
                        float* pLeftPointTanCalc = pPerThreadPositions + CurrentThreadPosStartIdx + PositionFloatCount * (rightPointIndex - 1);
                        float* pRightPointTanCalc = pPerThreadPositions + CurrentThreadPosStartIdx + PositionFloatCount * (rightPointIndex);

                        float* pCurrentTan = pPerThreadTangents + CurrentThreadTanStartIdx + TangentFloatCount * (rightPointIndex - 1);

                        pCurrentTan[0] = pRightPointTanCalc[0] - pLeftPointTanCalc[0];
                        pCurrentTan[1] = pRightPointTanCalc[1] - pLeftPointTanCalc[1];
                        pCurrentTan[2] = pRightPointTanCalc[2] - pLeftPointTanCalc[2];

                        NormalizeVector3f(pCurrentTan);
                    }

                    // Update left curvature
                    {
                        float* pLeftTanCurvatureCalc = pPerThreadTangents + CurrentThreadTanStartIdx + TangentFloatCount * (leftPointIndex - 1);
                        float* pRightTanCurvatureCalc = pPerThreadTangents + CurrentThreadTanStartIdx + TangentFloatCount * leftPointIndex;
                        float* pCurvature = pPerThreadCurvatures + CurrentThreadCurvatureStartIdx + (leftPointIndex - 1);


                        float temp[3];
                        temp[0] = (pRightTanCurvatureCalc[0] - pLeftTanCurvatureCalc[0]) * (1.0f / segmentLength);
                        temp[1] = (pRightTanCurvatureCalc[1] - pLeftTanCurvatureCalc[1]) * (1.0f / segmentLength);
                        temp[2] = (pRightTanCurvatureCalc[2] - pLeftTanCurvatureCalc[2]) * (1.0f / segmentLength);

                        float curvature = MagVector3f(temp);
                        pCurvature[0] = curvature;

                        // Also, cache the weight of that changed segment
                        float distance = curvature - minCurvature;
                        float realIdx = distance / curvatureStepSize;
                        uint32_t leftIdx = floor(realIdx);
                        uint32_t rightIdx = leftIdx + 1;

                        double leftLookup = pLookupTable[leftIdx];
                        double rightLookup = pLookupTable[rightIdx];

                        float leftDist = distance - (leftIdx * curvatureStepSize);

                        double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                        double interpolatedResultLog = log(interpolatedResult);
                        double segmentWeight = interpolatedResultLog;
                        segmentWeight += lnAbsorbtionConst;

                        // Remove old segmentWeight
                        runningPathWeight -= pCachedSegmentWeights[(leftPointIndex - 1) + (numSegmentsPerCurve * globalThreadIdx)];
                        pCachedSegmentWeights[(leftPointIndex - 1) + (numSegmentsPerCurve * globalThreadIdx)] = segmentWeight;
                        runningPathWeight += segmentWeight;
                    }

                    // Update right curvature
                    {
                        float* pLeftTanCurvatureCalc = pPerThreadTangents + CurrentThreadTanStartIdx + TangentFloatCount * (rightPointIndex - 1);
                        float* pRightTanCurvatureCalc = pPerThreadTangents + CurrentThreadTanStartIdx + TangentFloatCount * rightPointIndex;
                        float* pCurvature = pPerThreadCurvatures + CurrentThreadCurvatureStartIdx + (rightPointIndex - 1);

                        float temp[3];
                        temp[0] = (pRightTanCurvatureCalc[0] - pLeftTanCurvatureCalc[0]) * (1.0f / segmentLength);
                        temp[1] = (pRightTanCurvatureCalc[1] - pLeftTanCurvatureCalc[1]) * (1.0f / segmentLength);
                        temp[2] = (pRightTanCurvatureCalc[2] - pLeftTanCurvatureCalc[2]) * (1.0f / segmentLength);

                        float curvature = MagVector3f(temp);
                        pCurvature[0] = curvature;

                        // Also, cache the weight of that changed segment
                        float distance = curvature - minCurvature;
                        float realIdx = distance / curvatureStepSize;
                        uint32_t leftIdx = floor(realIdx);
                        uint32_t rightIdx = leftIdx + 1;

                        double leftLookup = pLookupTable[leftIdx];
                        double rightLookup = pLookupTable[rightIdx];

                        float leftDist = distance - (leftIdx * curvatureStepSize);

                        double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                        double interpolatedResultLog = log(interpolatedResult);
                        double segmentWeight = interpolatedResultLog;
                        segmentWeight += lnAbsorbtionConst;

                        // Remove old segmentWeight
                        runningPathWeight -= pCachedSegmentWeights[(rightPointIndex - 1) + (numSegmentsPerCurve * globalThreadIdx)];
                        pCachedSegmentWeights[(rightPointIndex - 1) + (numSegmentsPerCurve * globalThreadIdx)] = segmentWeight;
                        runningPathWeight += segmentWeight;
                    }

                    if (globalThreadIdx == 0)
                    {
                        printf("Made it to the end of path %d\n", pathCount);
                    }

                    if (pathCount < numPathsToSkipPerThread)
                    {
                        // Skip
                    }
                    else
                    {
                        printf("Here\n");
                        // Else, contribute to the paths
                        int32_t currentPathIdx = numPathsPerThread * globalThreadIdx + pathCount - numPathsToSkipPerThread;
                        assert(currentPathIdx >= numPathsPerThread * globalThreadIdx);
                        pCompressedPathWeights[currentPathIdx] = runningPathWeight;
                        printf("Updating path weight at %d to %0.6f\n", currentPathIdx, runningPathWeight);
                    }
                }
            }
        }
    }
}

namespace twisty
{

    GpuFullExperimentRunnerGeneral::GpuFullExperimentRunnerGeneral(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper, Range kdsRange, Range tdsRange)
        : ExperimentRunner(experimentParams, bootstrapper)
        , m_rng()
        , m_numSMs(0)
        , m_warpSize(0)
        , m_maxThreadsPerMultiprocessor(0)
        , m_pPerGlobalThreadRandStates(nullptr)
        , m_pPerPathCompressedWeightGlobal(nullptr)
    {
        uint32_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }
        std::cout << "\nPurturb seed used: " << seed << std::endl;
        m_rng = std::mt19937(seed);
    }


    GpuFullExperimentRunnerGeneral::~GpuFullExperimentRunnerGeneral()
    {
    }

    bool GpuFullExperimentRunnerGeneral::Setup()
    {
        bool result = SetupCudaDevice();
        if (!result)
        {
            printf("Failed to setup cuda device\n");
            return false;
        }

        // Ask the bootstrapper to generate a discrete curve.
        // If we fail, we want to exit the experiment.

        bool successfulGen = false;
        while (!successfulGen)
        {
            m_upInitialCurve = m_bootstrapper.CreateCurve(m_experimentParams.numSegmentsPerCurve);
            if (!m_upInitialCurve)
            {
                printf("Failed to create bootstrap curve.\n");
                return false;
            }

            // Lets also get the error of the initial curve, just to know
            float curveError = CurveUtils::CalculateCurveError(*m_upInitialCurve);
            std::cout << "\tSeed curve error: " << curveError << std::endl;

            if (curveError < m_experimentParams.maximumBootstrapCurveError)
            {
                successfulGen = true;
            }
        }

        return true;
    }



    // This sets up the cuda device for use. This could be pulled out into a more general class.
    bool GpuFullExperimentRunnerGeneral::SetupCudaDevice()
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

    // We calculate the dispatch parameters based off kernel/problem complexity
    bool GpuFullExperimentRunnerGeneral::SetupKernelDispatchParameters(uint32_t numPathWeightsInShared, uint32_t& numGlobalPerturbThreads, uint32_t& perturbBlockSize, uint32_t& perturbGridSize)
    {
        // Calculate minimum grid size and block size required to achieves maximum potential occupancy for GeneratePathBatchPutrubations
        {
            int blockSizePurturbKernel = 0;
            int minGridSizePurturbKernel = 0;
            size_t maxBlockSize = 0;

            size_t segmentsPerCurve = m_experimentParams.numSegmentsPerCurve;

            size_t sharedMemoryUse = 0;
            
            //// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g77b3bfb154b86e215a5bc01509ce8ea6
            CudaSafeErrorCheck(cudaOccupancyMaxPotentialBlockSize(&minGridSizePurturbKernel, &blockSizePurturbKernel,
                GeneratePathBatchPutrubations, sharedMemoryUse, maxBlockSize),
                "cudaOccupancyMaxPotentialBlockSize for perturb kernel");

            std::cout << "\nGeneratePutrubations: " << std::endl;
            std::cout << "\tMax Block Size: " << blockSizePurturbKernel << std::endl;
            std::cout << "\tMin Grid Size: " << minGridSizePurturbKernel << std::endl;

            // Assume we generate one path per thread
             perturbGridSize = minGridSizePurturbKernel;
             perturbBlockSize = blockSizePurturbKernel;

            perturbGridSize = 1024 * 2 * 2;
            perturbBlockSize = 32;

            numGlobalPerturbThreads = perturbGridSize * perturbBlockSize;

            printf("\tGeneratePathBatchPutrubations Grid Size: %d\n", perturbGridSize);
            printf("\tGeneratePutrubations Block Size: %d\n", perturbBlockSize);
            printf("\tGeneratePutrubations Num Block Threads: %d\n", numGlobalPerturbThreads);
        }

        return true;
    }

    bool GpuFullExperimentRunnerGeneral::SetupCuRandStates(uint32_t numGlobalPerturbThreads)
    {
        // Random Seed Kernel
        // Every block thread needs its own curand state
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerGlobalThreadRandStates, sizeof(curandState_t) * numGlobalPerturbThreads),
            "RandState malloc");
        
        int blockSizeRandKernel = 0;
        int minGridSizeRandKernel = 0;
        size_t sharedMemoryUse = 0;
        size_t maxBlockSize = 0;
        cudaOccupancyMaxPotentialBlockSize(&minGridSizeRandKernel, &blockSizeRandKernel, InitializeCurandState, sharedMemoryUse, maxBlockSize);
        std::cout << "\nInitializeCurandState: " << std::endl;
        std::cout << "\tBlock Size: " << blockSizeRandKernel << std::endl;
        std::cout << "\tMin Grid Size: " << minGridSizeRandKernel << std::endl;
        
        size_t gridSizeRandKernel = ((numGlobalPerturbThreads) + blockSizeRandKernel - 1) / blockSizeRandKernel;

        printf("\tInitializeCurandState Grid Size: %d\n", gridSizeRandKernel);
        printf("\tInitializeCurandState Block Size: %d\n", blockSizeRandKernel);
        
        // Dispatch CurandState
        // We need a dispatch that initializes curand per thread
        {
            dim3 gridSize(gridSizeRandKernel, 1, 1);
            dim3 blockSize(blockSizeRandKernel, 1, 1);
            size_t sharedMemorySizeBytes = 0;
            cudaStream_t stream = 0;

            InitializeCurandState <<<gridSize, blockSize, sharedMemorySizeBytes, stream >>>(
                static_cast<uint32_t>(m_experimentParams.curvePurturbSeed),
                m_pPerGlobalThreadRandStates,
                numGlobalPerturbThreads
            );

            CudaSafeErrorCheck(cudaGetLastError(), "Rand state init kernal launch");
            CudaSafeErrorCheck(cudaDeviceSynchronize(), "Rand state kernel sync");
        }

        return true;
    }


    bool GpuFullExperimentRunnerGeneral::SetupCrossDispatchCurveData(uint32_t numGlobalPerturbThreads)
    {
        std::vector<Farlor::Vector3> initialCurvePositions(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> initialCurveTangents(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<float> initialCurveCurvatures(m_experimentParams.numSegmentsPerCurve);

        // Positions
        // Hard code the first two positions
        initialCurvePositions[0] = m_upInitialCurve->m_basePos;
        initialCurvePositions[1] = m_upInitialCurve->m_basePos + m_upInitialCurve->m_baseTangent.Normalized() * m_upInitialCurve->m_segmentLength;
        for (uint32_t segmentIdx = 2; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
        {
            initialCurvePositions[segmentIdx] = m_upInitialCurve->m_positions[segmentIdx];
        }
        // Hard code the final position
        initialCurvePositions[m_experimentParams.numSegmentsPerCurve] = m_upInitialCurve->m_targetPos;

#if defined(DetailedPurturb)
        {
            std::cout << "Positions" << std::endl;
            for (uint32_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurvePositions[segmentIdx] << std::endl;
            }
        }
#endif

        // Tangents
        // Hardcode intial tangent
        initialCurveTangents[0] = m_upInitialCurve->m_baseTangent;
        for (uint32_t tanIdx = 1; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
        {
            Farlor::Vector3 leftPos = initialCurvePositions[tanIdx];
            Farlor::Vector3 rightPos = initialCurvePositions[tanIdx + 1];

            initialCurveTangents[tanIdx] = (rightPos - leftPos).Normalized();
        }
        // Final Tangents
        initialCurveTangents[m_experimentParams.numSegmentsPerCurve] = m_upInitialCurve->m_targetTangent;

#if defined(DetailedPurturb)
        {
            std::cout << "Tangents" << std::endl;
            for (uint32_t tanIdx = 0; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
            {
                std::cout << "\t" << initialCurveTangents[tanIdx] << std::endl;
            }
        }
#endif

        // Curvatures
        float segLength = m_upInitialCurve->m_arclength / m_upInitialCurve->m_numSegments;
        for (uint32_t curvatureIdx = 0; curvatureIdx < m_experimentParams.numSegmentsPerCurve; ++curvatureIdx)
        {
            Farlor::Vector3 tanLeft = initialCurveTangents[curvatureIdx];
            Farlor::Vector3 tanRight = initialCurveTangents[curvatureIdx + 1];

            Farlor::Vector3 curvatureVec = (tanRight - tanLeft) * (1.0f / segLength);
            float curvature = curvatureVec.Magnitude();
            initialCurveCurvatures[curvatureIdx] = curvature;
        }

#if defined(DetailedPurturb)
        {
            std::cout << "Curvatures" << std::endl;
            for (uint32_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurveCurvatures[segmentIdx] << std::endl;
            }
        }
#endif

        const uint32_t NumPosPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const uint32_t NumTanPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const uint32_t NumCurvaturePerCurve = (m_upInitialCurve->m_numSegments);

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerThreadPositions, sizeof(Farlor::Vector3) * (m_experimentParams.numSegmentsPerCurve + 1) * numGlobalPerturbThreads), "Failed to allocate per thread position memory");
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerThreadTangents, sizeof(Farlor::Vector3) * (m_experimentParams.numSegmentsPerCurve + 1) * numGlobalPerturbThreads), "Failed to allocate per thread tangent memory");
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerThreadCurvatures, sizeof(float) * (m_experimentParams.numSegmentsPerCurve) * numGlobalPerturbThreads), "Failed to allocate per thread curvature memory");

        for (uint32_t threadIdx = 0; threadIdx < numGlobalPerturbThreads; ++threadIdx)
        {
            {
                float* pDest = m_pPerThreadPositions + 3 * (m_experimentParams.numSegmentsPerCurve + 1) * threadIdx;
                CudaSafeErrorCheck(cudaMemcpy(pDest, initialCurvePositions.data(), sizeof(Farlor::Vector3) * (m_experimentParams.numSegmentsPerCurve + 1), cudaMemcpyHostToDevice),
                    "Copy initial curve to device");
            }

            {
                float* pDest = m_pPerThreadTangents + 3 * (m_experimentParams.numSegmentsPerCurve + 1) * threadIdx;
                CudaSafeErrorCheck(cudaMemcpy(pDest, initialCurveTangents.data(), sizeof(Farlor::Vector3) * (m_experimentParams.numSegmentsPerCurve + 1), cudaMemcpyHostToDevice),
                    "Copy initial curve to device");
            }

            {
                float* pDest = m_pPerThreadCurvatures + 1 * m_experimentParams.numSegmentsPerCurve * threadIdx;
                CudaSafeErrorCheck(cudaMemcpy(pDest, initialCurveCurvatures.data(), sizeof(float) * m_experimentParams.numSegmentsPerCurve, cudaMemcpyHostToDevice),
                    "Copy initial curve to device");
            }
        }

        // TODO: Move this over to shared memory, this should fit
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerThreadSegmentWeightCache, sizeof(double) * m_experimentParams.numSegmentsPerCurve * numGlobalPerturbThreads), "Failed to allocate cached segment weight buffer");
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerPathCompressedWeightGlobal, sizeof(double) * m_experimentParams.numPathsPerBatch), "Failed to allocate global path weight buffer");

        return true;
    }

    void GpuFullExperimentRunnerGeneral::CleanupCudaMemory()
    {
        cudaFree(m_pPerPathCompressedWeightGlobal);
        cudaFree(m_pPerThreadSegmentWeightCache);
        cudaFree(m_pPerThreadCurvatures);
        cudaFree(m_pPerThreadTangents);
        cudaFree(m_pPerThreadPositions);
        cudaFree(m_pWeightLookupTable);
        cudaFree(m_pPerGlobalThreadRandStates);
    }

    bool GpuFullExperimentRunnerGeneral::SetupWeightLookupTexture(const twisty::PathSpaceUtils::WeightLookupTableIntegral& lookupEvaluator)
    {
        auto& weightValues = lookupEvaluator.AccessLookupTable();

        CudaSafeErrorCheck(
            cudaMalloc((void**)&m_pWeightLookupTable, sizeof(double) * weightValues.size()),
            "Failed to allocate weight lookup table"
        );

        CudaSafeErrorCheck(
            cudaMemcpy(m_pWeightLookupTable, weightValues.data(), sizeof(double) * weightValues.size(), cudaMemcpyHostToDevice),
            "Copy weight lookup table to device"
        );

        return true;
    }

    void GpuFullExperimentRunnerGeneral::WeightCombineThreadKernel(const uint32_t threadIdx, uint32_t numWeights, uint32_t numWeightsPerThread, const std::vector<double>& compressedWeights,
        twisty::BigFloat& threadWeight)
    {
        for (uint32_t i = 0; i < numWeightsPerThread; i++)
        {
            uint32_t idx = threadIdx * numWeightsPerThread + i;
            if (idx >= numWeights)
            {
                break;
            }

            twisty::BigFloat bigfloatCompressed = compressedWeights[idx];
#ifdef BigFloatMultiprecision
            threadWeight += boost::multiprecision::exp(bigfloatCompressed);
#else
            //threadWeight += std::exp(bigfloatCompressed);
#endif
        }
    }

    ExperimentRunner::ExperimentResults GpuFullExperimentRunnerGeneral::RunExperiment()
    {
        // A value of 1000 failed to work, not enough shared mem.
        // A value of 10 seems to work nicely
        // TODO: Play with this more to find a better value
        // WTF does this represent?
        const uint32_t numCachedPathWeightsShardMem = 1;

        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();

        // Calculate grid and block sizes based on the kernels we will call
        uint32_t numGlobalPerturbThreads = 0;
        uint32_t perturbBlockSize = 0;
        uint32_t perturbGridSize = 0;

        bool result = true;
        auto setupKernelParamsTimeStart = std::chrono::high_resolution_clock::now();
        {
            result = SetupKernelDispatchParameters(numCachedPathWeightsShardMem, numGlobalPerturbThreads, perturbBlockSize, perturbGridSize);
            if (!result)
            {
                printf("Failed to setup kernel dispatch parameters\n");
                return {};
            }
        }
        auto setupKernelParamsTimeEnd = std::chrono::high_resolution_clock::now();

        auto setupCurandTimeStart = std::chrono::high_resolution_clock::now();
        {
            result = SetupCuRandStates(numGlobalPerturbThreads);
            if (!result)
            {
                printf("Failed to setup curand states\n");
                return {};
            }
        }
        auto setupCurandTimeEnd = std::chrono::high_resolution_clock::now();


        /* ---------------------------- */
        auto setupCurveDataStructuresTimeStart = std::chrono::high_resolution_clock::now();
        uint32_t numFailures = 0;
        uint32_t totalFailures = 0;
        uint32_t totalSuccess = 0;
        float ds = m_upInitialCurve->m_arclength / m_experimentParams.numSegmentsPerCurve;
        double minCurvature = 0.0;
        double maxCurvature = (3.47 / ds) * 2.0;

        twisty::PathSpaceUtils::WeightLookupTableIntegral lookupEvaluator(ds, m_experimentParams.weightingParameters.mu, m_experimentParams.weightingParameters.numStepsInt,
            m_experimentParams.weightingParameters.minBound, m_experimentParams.weightingParameters.maxBound, m_experimentParams.weightingParameters.eps,
            minCurvature, maxCurvature, m_experimentParams.weightingParameters.numCurvatureSteps, m_experimentParams.weightingParameters.scatter);
        const float curvatureStepSize = (maxCurvature - minCurvature) / m_experimentParams.weightingParameters.numCurvatureSteps;



        // Do non dispatch specific setup
        {
            // Allocate and copy lookup table over to GPU
            result = SetupWeightLookupTexture(lookupEvaluator);
            if (!result)
            {
                printf("Failed to setup weight lookup texture\n");
                return {};
            }

            result = SetupCrossDispatchCurveData(numGlobalPerturbThreads);
            if (!result)
            {
                printf("Failed to setup curve device data structures\n");
                return {};
            }
        }
        auto setupCurveDataStructuresTimeEnd = std::chrono::high_resolution_clock::now();
        /* ---------------------------- */



        const uint32_t numDispatches = (m_experimentParams.numPathsInExperiment + m_experimentParams.numPathsPerBatch - 1) / m_experimentParams.numPathsPerBatch;
        std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
        std::cout << "numPathsPerBatch: " << m_experimentParams.numPathsPerBatch << std::endl;
        std::cout << "Num dispatches required: " << numDispatches << std::endl;
        uint32_t numPathsLeft = m_experimentParams.numPathsInExperiment;
        uint32_t numPathsGenerated = 0;

        // We need to calculate the absorbtion/scattering piece
        boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;

        // We need a number of dispatches
        long long perturbTimeCount = 0;
        long long weightCopyTimeCount = 0;
        long long weightCalcTimeCount = 0;
        for (uint32_t dispatchIdx = 0; dispatchIdx < numDispatches; ++dispatchIdx)
        {
            uint32_t pathsInDispatch = std::min(m_experimentParams.numPathsPerBatch, numPathsLeft);
            std::cout << "Paths in dispatch " << dispatchIdx << ": " << pathsInDispatch << std::endl;

            auto perturbPhaseTimeStart = std::chrono::high_resolution_clock::now();
            {

                // At this point, we know how many curves we'll want to generate. So, we setup our parameters to handle this.
                std::cout << "Experiment Information: " << std::endl;

                uint32_t numPathsPerThread = (pathsInDispatch + numGlobalPerturbThreads - 1) / numGlobalPerturbThreads;
                std::cout << "\tNum paths generated per global thread: " << numPathsPerThread << std::endl;


                // Dispatch CurandState
                // We need a dispatch that initializes curand per thread
                {
                    dim3 gridSize(perturbGridSize, 1, 1);
                    dim3 blockSize(perturbBlockSize, 1, 1);
                    size_t sharedMemorySizeBytes = 0;
                    cudaStream_t stream = 0;

                    /*
                __global__ void GeneratePathBatchPutrubations(
                uint32_t numExperimentPaths,
                uint32_t numPathsPerThread,
                uint32_t numPathsToSkipPerThread,
                uint32_t numSegmentsPerCurve,
                curandState_t* pRandStates,
                float* pPerThreadPositions,
                float* pPerThreadTangents,
                float* pPerThreadCurvatures,
                double* pCachedSegmentWeights,
                double* pCompressedPathWeights,
                float segmentLength,
                float scattering,
                float absorbtion,
                double* pLookupTable,
                float minCurvature,
                float maxCurvature,
                float curvatureStepSize
                )
                */

                    GeneratePathBatchPutrubations << <gridSize, blockSize, sharedMemorySizeBytes, stream >> > (
                        pathsInDispatch,
                        numPathsPerThread,
                        m_experimentParams.numPathsToSkip,
                        m_experimentParams.numSegmentsPerCurve,
                        m_pPerGlobalThreadRandStates,
                        m_pPerThreadPositions,
                        m_pPerThreadTangents,
                        m_pPerThreadCurvatures,
                        m_pPerThreadSegmentWeightCache,
                        m_pPerPathCompressedWeightGlobal,
                        m_upInitialCurve->m_segmentLength,
                        m_experimentParams.weightingParameters.scatter,
                        m_experimentParams.weightingParameters.absorbtion,
                        m_pWeightLookupTable,
                        minCurvature,
                        maxCurvature,
                        curvatureStepSize
                        );

                    CudaSafeErrorCheck(cudaGetLastError(), "Perturb kernal launch error");
                    CudaSafeErrorCheck(cudaDeviceSynchronize(), "Perturb kernel sync error");

                    std::cout << "Done with the perturb phase" << std::endl;
                }
            }
            auto perturbPhaseTimeEnd = std::chrono::high_resolution_clock::now();
            perturbTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(perturbPhaseTimeEnd - perturbPhaseTimeStart).count();
            /* ---------------------------- */

            //TODO: Read back weights
            /* ---------------------------- */
            auto weightCopyTimeStart = std::chrono::high_resolution_clock::now();
            std::vector<double> compressedWeightBuffer(m_experimentParams.numPathsPerBatch);
            CudaSafeErrorCheck(cudaMemcpy(compressedWeightBuffer.data(), m_pPerPathCompressedWeightGlobal, sizeof(double) * pathsInDispatch, cudaMemcpyDeviceToHost),
                "Copy back compressed weights from device");
            auto weightCopyTimeEnd = std::chrono::high_resolution_clock::now();
            weightCopyTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightCopyTimeEnd - weightCopyTimeStart).count();
            /* ---------------------------- */

            /* ---------------------------- */
            auto weightingTimeStart = std::chrono::high_resolution_clock::now();
            twisty::BigFloat totalExperimentWeight = 0.0f;
            // Create threads and dispatch them
            uint32_t numHardwareThreads = std::thread::hardware_concurrency();
            uint32_t numWeightsPerThread = (pathsInDispatch + numHardwareThreads - 1) / numHardwareThreads;

            std::vector<std::thread> threads(numHardwareThreads);
            std::vector<twisty::BigFloat> threadWeights(numHardwareThreads);
            for (uint32_t threadIdx = 0; threadIdx < numHardwareThreads; ++threadIdx)
            {
                threadWeights[threadIdx] = 0.0;
                std::thread newThread(&GpuFullExperimentRunnerGeneral::WeightCombineThreadKernel, this, threadIdx,
                    pathsInDispatch, numWeightsPerThread, std::ref(compressedWeightBuffer), std::ref(threadWeights[threadIdx]));
                threads[threadIdx] = std::move(newThread);
            }

            for (uint32_t threadIdx = 0; threadIdx < numHardwareThreads; ++threadIdx)
            {
                if (threads[threadIdx].joinable())
                {
                    threads[threadIdx].join();
                }
            }

            for (uint32_t threadIdx = 0; threadIdx < numHardwareThreads; ++threadIdx)
            {
                totalExperimentWeight += threadWeights[threadIdx];
            }

            bigTotalExperimentWeight += totalExperimentWeight;

            auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
            weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightingTimeEnd - weightingTimeStart).count();
            /* ---------------------------- */

            numPathsLeft -= pathsInDispatch;
            numPathsGenerated += pathsInDispatch;
        }

        // Now that they're read back, delete the allocated memory
        CleanupCudaMemory();

        auto runExperimentTimeEnd = std::chrono::high_resolution_clock::now();

        std::cout << "Experiment Time Reporting: " << std::endl;
        auto runExperimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(runExperimentTimeEnd - runExperimentTimeStart);
        std::cout << "\tTotal Experiment Time: " << runExperimentTimeMs.count() << "ms" << std::endl;

        {
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupKernelParamsTimeEnd - setupKernelParamsTimeStart);
            std::cout << "\tsetupKernelParams Time: " << timeMs.count() << "ms - " << ((float)timeMs.count() / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupCurandTimeEnd - setupCurandTimeStart);
            std::cout << "\tsetupCurand Time: " << timeMs.count() << "ms - " << ((float)timeMs.count() / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupCurveDataStructuresTimeEnd - setupCurveDataStructuresTimeStart);
            std::cout << "\tsetupCurveDataStructures Time: " << timeMs.count() << "ms - " << ((float)timeMs.count() / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tperturbPhase Time: " << perturbTimeCount << "ms - " << ((float)perturbTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tweightCopy Time: " << weightCopyTimeCount << "ms - " << ((float)weightCopyTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tweighting Time: " << weightCalcTimeCount << "ms - " << ((float)weightCalcTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }


        ExperimentResults results;
        results.experimentWeight = bigTotalExperimentWeight;
        results.totalPathsGenerated = m_experimentParams.numPathsInExperiment;
        results.numFailedPaths = 0;
        return results;
    }

    void GpuFullExperimentRunnerGeneral::Shutdown()
    {
    }

    static std::pair<float, float> CurvatureAndTorsionBetweenTwoFrames(const Farlor::Matrix3x3& startFrame, const Farlor::Matrix3x3& endFrame, float segmentLength)
    {
        std::pair<float, float> curvatureAndTorsion = { 0.0f, 0.0f };
        {
            float curvature = ((endFrame.m_rows[0] - startFrame.m_rows[0]) * (1.0f / segmentLength)).Magnitude();
            curvatureAndTorsion.first = curvature;
        }

        {
            auto torsionLeft = -1.0f * startFrame.m_rows[1];
            auto torsionRight = (endFrame.m_rows[2] - startFrame.m_rows[2]) * (1.0f / segmentLength);
            float torsion = torsionLeft.Dot(torsionRight);
            curvatureAndTorsion.second = torsion;
        }
        return curvatureAndTorsion;
    }
}