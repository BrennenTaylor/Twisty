#include "GpuFullExperimentRunnerOptimized.h"

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

constexpr bool DetailedPurturb = false;

// TODO: This is currently arbitrary... we need to have a better method of allowing more than 
// 50 segments, most likely. Not priority atm
const uint32_t MaxNumberOfSegments = 1000;
// We need one per segment, then one additional one for the target pos
const uint32_t MaxNumberOfPositions = MaxNumberOfSegments + 1;
// We need one per segment, then one additional one for the target tan
const uint32_t MaxNumberOfTangents = MaxNumberOfSegments + 1;
const uint32_t PositionFloatCount = sizeof(twisty::GpuFullExperimentRunnerOptimized::GpuDeviceVector3Aligned) / sizeof(float);
const uint32_t TangentFloatCount = sizeof(twisty::GpuFullExperimentRunnerOptimized::GpuDeviceVector3Aligned) / sizeof(float);
__constant__ float device_constant_InitialCurve[MaxNumberOfPositions * PositionFloatCount + MaxNumberOfTangents * TangentFloatCount];

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

    // Assumes pVector3f is an array of 3 floats
    __device__ void NormalizeVector3f(float* pVector3f)
    {
        float normalizer = pVector3f[0] * pVector3f[0] + pVector3f[1] * pVector3f[1] + pVector3f[2] * pVector3f[2];
        normalizer = 1.0 / sqrt(normalizer);
        pVector3f[0] *= normalizer;
        pVector3f[1] *= normalizer;
        pVector3f[2] *= normalizer;
    }

    // This has an outparameter
    __device__ void RotationMatrixAroundAxis(float angle, float* pAxisVector3f, float* pMatrix3x3)
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

    __device__ float DotVector3fVector3f(float* lhs, float* rhs)
    {
        return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
    }

    __device__ float MagVector3f(float* pVec)
    {
        return sqrt(pVec[0] * pVec[0] + pVec[1] * pVec[1] + pVec[2] * pVec[2]);
    }

    __device__ void RotateVectorByMatrix(float* pRotationMatrix, float* pVector)
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
        float maxCurvature)
    {
        // First, we calculate the block thread index of the entire gpu dispatch.
        // For the number of threads dispatched, each is responsible for a number of 
        // paths to generate, i.e. numExperimentThreads / numGlobalThreads
        uint32_t globalThreadIdx = threadIdx.x + blockDim.x * blockIdx.x;
        uint32_t blockThreadIdx = threadIdx.x;

        // Load the inital curve in from global memory
        extern __shared__ float SharedPerThreadData[];

        // The current thread is stored at the beginning
        // We want to index into shared memory via threadIdx.x as this is assigned per block
        const uint32_t PositionOffset = 0;
        const uint32_t TangentOffset = PositionOffset + (numSegmentsPerCurve + 1) * PositionFloatCount;
        const uint32_t CacheOffset = TangentOffset + (numSegmentsPerCurve + 1) * TangentFloatCount;
        const uint32_t PerThreadFloatCount = CacheOffset + numCachedPathWeightsPerThread;

        float* pCurrentThreadCurve = (float*)SharedPerThreadData + PerThreadFloatCount * blockThreadIdx;
        float* pCurrentThreadCurvePositionStart = pCurrentThreadCurve + PositionOffset;
        float* pCurrentThreadCurveTangentStart = pCurrentThreadCurve + TangentOffset;
        float* pCurrentThreadPathWeightCacheStart = pCurrentThreadCurve + CacheOffset;

        uint32_t currentPathIdx = numPathsPerThread * globalThreadIdx;
        if (currentPathIdx >= numExperimentPaths)
        {
            // We dont want to continue if we have already generated the correct number of paths.
            return;
        }

        // Data transfer piece
        // We intialize everything we need into shared memory first
        {
            // Copy from constant to device memory
            // CacheOffset is the number of floats in position and tangents
            const uint32_t numInitialCurveFloats = (numSegmentsPerCurve + 1) * 4 * 2;
            for (uint32_t i = 0; i < numInitialCurveFloats; ++i)
            {
                pCurrentThreadCurve[i] = device_constant_InitialCurve[i];
                //printf("Value: %.6f\n", device_constant_InitialCurve[i]);
            }
        }

        // Synchronize until we have loaded all of the initial curve into each of our little shared
        // memory slots
        //__syncthreads();

        // Now, we can begin the actual algorithm
        {
            // This is the perturbation piece.
            // Can we do this in place, most likely
            // This will modify pCurrentThreadCurve
            // Remember, the structure of this is:
            // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M

            uint32_t cacheCount = 0;
            // Start at the thread's first path idx
            uint32_t cacheStartPathIdx = numPathsPerThread * globalThreadIdx;
            for (uint32_t perThreadPathCount = 0; perThreadPathCount < numPathsPerThread; ++perThreadPathCount)
            {
                uint32_t currentPathIdx = numPathsPerThread * globalThreadIdx + perThreadPathCount;
                if (currentPathIdx >= numExperimentPaths)
                {
                    // We dont want to continue if we have already generated the correct number of paths.
                    break;
                }

                // Do the perturb now
                {
                    // We bound on left by one as we dont want to rotate the first segment at all
                    // Left bound by m-2 as we at least want there to be one point between the left and right points selected so an actual perturbation occurs
                    //std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - 3); // uniform, unbiased
                    unsigned int leftPointIndex = ceilf(curand_uniform(&pRandStates[globalThreadIdx]) * (numSegmentsPerCurve - 3));
                    //std::uniform_int_distribution<int> rightPointIndexUniformDist(leftPointIndex + 2, numSegmentsPerCurve - 1); // uniform, unbiased
                    unsigned int rightPointIndex = ceilf(curand_uniform(&pRandStates[globalThreadIdx]) * ((numSegmentsPerCurve - 1) - (leftPointIndex + 2)) + (leftPointIndex + 2));
                    
                    assert(leftPointIndex < rightPointIndex);
                    assert((rightPointIndex - leftPointIndex) >= 2);

                    // We need two frames for each segment to get the new curvature and torsion.
                    // we need the frame left of the segment, as well as the frame right of the segment.

                    // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                    float* pLeftPoint = pCurrentThreadCurve + PositionFloatCount * leftPointIndex;
                    float* pRightPoint = pCurrentThreadCurve + PositionFloatCount * rightPointIndex;

                    float axisOfRotation[3];
                    axisOfRotation[0] = pRightPoint[0] - pLeftPoint[0];
                    axisOfRotation[1] = pRightPoint[1] - pLeftPoint[1];
                    axisOfRotation[2] = pRightPoint[2] - pLeftPoint[2];

                    NormalizeVector3f(axisOfRotation);

                    float randomAngle = (curand_uniform(&pRandStates[globalThreadIdx]) * 360.0f) - 180.0f;

                    float rotationMatrix[9];
                    RotationMatrixAroundAxis(randomAngle, axisOfRotation, rotationMatrix);

                    for (uint32_t pointIdx = 0; pointIdx < (numSegmentsPerCurve + 1); ++pointIdx)
                    {
                        if ((pointIdx > leftPointIndex) && (pointIdx < rightPointIndex))
                        {
                            float* pCurrentPoint = pCurrentThreadCurve + PositionFloatCount * pointIdx;


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
                    }

                    // For now, simply compute the difference in positions.
                    // We can do a different approach later.
                    for (uint32_t segIdx = 0; segIdx < numSegmentsPerCurve; ++segIdx)
                    {
                        float* pLeftPointTanCalc = pCurrentThreadCurve + PositionFloatCount * segIdx;
                        float* pRightPointTanCalc = pCurrentThreadCurve + PositionFloatCount * (segIdx + 1);

                        float* pCurrentTan = pCurrentThreadCurveTangentStart + TangentFloatCount * segIdx;

                        pCurrentTan[0] = pRightPointTanCalc[0] - pLeftPointTanCalc[0];
                        pCurrentTan[1] = pRightPointTanCalc[1] - pLeftPointTanCalc[1];
                        pCurrentTan[2] = pRightPointTanCalc[2] - pLeftPointTanCalc[2];

                        NormalizeVector3f(pCurrentTan);
                    }

                }

                // Now, we do the weighting
                {
                    float pathWeight = 0.0f;

                    for (uint32_t segIdx = 0; segIdx < numSegmentsPerCurve; ++segIdx)
                    {
                        float* pLeftTanCurvatureCalc = pCurrentThreadCurveTangentStart + TangentFloatCount * segIdx;
                        float* pRightTanCurvatureCalc = pCurrentThreadCurveTangentStart + TangentFloatCount * (segIdx + 1);

                        // Print Segs
                        //printf("Left x: %.6f\n", pLeftTanCurvatureCalc[0]);
                        //printf("Left y: %.6f\n", pLeftTanCurvatureCalc[1]);
                        //printf("Left z: %.6f\n", pLeftTanCurvatureCalc[2]);

                        //printf("Left x: %.6f\n", pRightTanCurvatureCalc[0]);
                        //printf("Left y: %.6f\n", pRightTanCurvatureCalc[1]);
                        //printf("Left z: %.6f\n", pRightTanCurvatureCalc[2]);

                        float temp[3];
                        temp[0] = (pRightTanCurvatureCalc[0] - pLeftTanCurvatureCalc[0]) * (1.0f / segmentLength);
                        temp[1] = (pRightTanCurvatureCalc[1] - pLeftTanCurvatureCalc[1]) * (1.0f / segmentLength);
                        temp[2] = (pRightTanCurvatureCalc[2] - pLeftTanCurvatureCalc[2]) * (1.0f / segmentLength);

                        float curvature = MagVector3f(temp);

                        //printf("Curvature: %.6f\n", curvature);

                        // What if we calculate the curvature as we calculate the weight?
                        // Next, we weight it

                        // TODO: Do weighting
                        float curvatureU = curvature / (maxCurvature - minCurvature);
                        float textureLookup = tex1D<float>(texObj, curvatureU);

                        float c = scattering + absorbtion;
                        float constant = std::exp(-c * segmentLength) / (2.0 * CUDART_PI_F * CUDART_PI_F);
                        pathWeight += log(constant) + log(textureLookup);

                        // Finally, we place the value into the global memory write location it belongs

                        // TODO: Write value
                    }

                    // If the cache is full, flush it over to global memory
                    if (cacheCount == numCachedPathWeightsPerThread)
                    {
                        // Copy over values then sync, this should be the same for all threads
                        for (uint32_t cacheIdx = 0; cacheIdx < cacheCount; ++cacheIdx)
                        {
                            uint32_t globalPathWeightIdx = cacheStartPathIdx + cacheIdx;
                            pGlobalPathWeights[globalPathWeightIdx] = pCurrentThreadPathWeightCacheStart[cacheIdx];
                        }
                        cacheStartPathIdx += cacheCount;
                        cacheCount = 0;

                        // TODO: Does this make it better, or worse?
                        //__syncthreads();
                    }


                    // Place the value in the cache
                    const uint32_t currentThreadCacheIdx = perThreadPathCount % numCachedPathWeightsPerThread;
                    pCurrentThreadPathWeightCacheStart[currentThreadCacheIdx] = pathWeight;
                    cacheCount++;
                }
            }

            // If we have any cached values, flush the mover to global memory
            if (cacheCount > 0)
            {
                // Copy over values then sync, this should be the same for all threads
                for (uint32_t cacheIdx = 0; cacheIdx < cacheCount; ++cacheIdx)
                {
                    uint32_t globalPathWeightIdx = cacheStartPathIdx + cacheIdx;
                    pGlobalPathWeights[globalPathWeightIdx] = pCurrentThreadPathWeightCacheStart[cacheIdx];
                }
                cacheStartPathIdx += cacheCount;

                // TODO: Does this make it better, or worse?
                //__syncthreads();
            }
        }
    }
}

namespace twisty
{

    GpuFullExperimentRunnerOptimized::GpuFullExperimentRunnerOptimized(ExperimentRunner::ExperimentParameters& experimentParams,
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


    GpuFullExperimentRunnerOptimized::~GpuFullExperimentRunnerOptimized()
    {
    }

    bool GpuFullExperimentRunnerOptimized::Setup()
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

            // Once we have a curve, we know arclength.
            // Thus, we can setup the min and max curvatures
            float ds = m_upInitialCurve->m_arclength / m_upInitialCurve->m_numSegments;

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
    bool GpuFullExperimentRunnerOptimized::SetupCudaDevice()
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
        // TODO: Allow for selection
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
    bool GpuFullExperimentRunnerOptimized::SetupKernelDispatchParameters(uint32_t numPathWeightsInShared, uint32_t& numGlobalPerturbThreads, uint32_t& perturbBlockSize, uint32_t& perturbGridSize)
    {
        // Calculate minimum grid size and block size required to achieves maximum potential occupancy for GeneratePathBatchPutrubations
        {
            int blockSizePurturbKernel = 0;
            int minGridSizePurturbKernel = 0;
            size_t maxBlockSize = 0;

            size_t segmentsPerCurve = m_experimentParams.numSegmentsPerCurve;

            size_t sharedMemoryUse = 0;
            
            //// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g77b3bfb154b86e215a5bc01509ce8ea6
            //cudaOccupancyMaxPotentialBlockSize(&minGridSizePurturbKernel, &blockSizePurturbKernel,
            //    GeneratePathBatchPutrubations, sharedMemoryUse, maxBlockSize);

            // TODO: Switch to this version
            CudaSafeErrorCheck(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSizePurturbKernel, &blockSizePurturbKernel,
                GeneratePathBatchPutrubations,
                [segmentsPerCurve, numPathWeightsInShared](int candidateBlockSize) -> size_t {
                    size_t curveWorkspaceSize = 2 * (segmentsPerCurve + 1) * 4 * 4;
                    size_t cachedPathWeights = 4 * numPathWeightsInShared;
                    return candidateBlockSize * (curveWorkspaceSize + cachedPathWeights);
                },
                maxBlockSize), 
                "cudaOccupancyMaxPotentialBlockSizeVariableSMem for perturb kernel");


            std::cout << "\nGeneratePutrubations: " << std::endl;
            std::cout << "\tMax Block Size: " << blockSizePurturbKernel << std::endl;
            std::cout << "\tMin Grid Size: " << minGridSizePurturbKernel << std::endl;

            // Assume we generate one path per thread
            perturbGridSize = minGridSizePurturbKernel;
            perturbBlockSize = blockSizePurturbKernel;

            numGlobalPerturbThreads = perturbGridSize * perturbBlockSize;

            printf("\tGeneratePathBatchPutrubations Grid Size: %d\n", perturbGridSize);
            printf("\tGeneratePutrubations Block Size: %d\n", perturbBlockSize);
            printf("\tGeneratePutrubations Num Block Threads: %d\n", numGlobalPerturbThreads);
        }

        return true;
    }

    bool GpuFullExperimentRunnerOptimized::SetupCuRandStates(uint32_t numGlobalPerturbThreads)
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

    bool GpuFullExperimentRunnerOptimized::SetupCurveDataStructures()
    {
        std::vector<GpuDeviceVector3Aligned> initialCurvePositionsAndTangents;
        initialCurvePositionsAndTangents.resize((m_experimentParams.numSegmentsPerCurve + 1) * 2);

        for (uint32_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
        {
            auto& segPos = m_upInitialCurve->m_positions[segmentIdx];
            initialCurvePositionsAndTangents[segmentIdx].x = segPos.x;
            initialCurvePositionsAndTangents[segmentIdx].y = segPos.y;
            initialCurvePositionsAndTangents[segmentIdx].z = segPos.z;
            initialCurvePositionsAndTangents[segmentIdx].w = 0.0f;
            //std::cout << "Inital Curve Seg " << segmentIdx << "x: " << initialCurvePositionsAndTangents[segmentIdx].x << std::endl;
            //std::cout << "Inital Curve Seg " << segmentIdx << "y: " << initialCurvePositionsAndTangents[segmentIdx].y << std::endl;
            //std::cout << "Inital Curve Seg " << segmentIdx << "z: " << initialCurvePositionsAndTangents[segmentIdx].z << std::endl;

            auto& segTan = m_upInitialCurve->m_tangents[segmentIdx];
            initialCurvePositionsAndTangents[segmentIdx + m_experimentParams.numSegmentsPerCurve + 1].x = segTan.x;
            initialCurvePositionsAndTangents[segmentIdx + m_experimentParams.numSegmentsPerCurve + 1].y = segTan.y;
            initialCurvePositionsAndTangents[segmentIdx + m_experimentParams.numSegmentsPerCurve + 1].z = segTan.z;
            initialCurvePositionsAndTangents[segmentIdx + m_experimentParams.numSegmentsPerCurve + 1].w = 0.0f;

            //std::cout << "Inital Curve Seg " << segmentIdx << "x: " << initialCurvePositionsAndTangents[segmentIdx + m_experimentParams.numSegmentsPerCurve + 1].x << std::endl;
            //std::cout << "Inital Curve Seg " << segmentIdx << "y: " << initialCurvePositionsAndTangents[segmentIdx + m_experimentParams.numSegmentsPerCurve + 1].y << std::endl;
            //std::cout << "Inital Curve Seg " << segmentIdx << "z: " << initialCurvePositionsAndTangents[segmentIdx + m_experimentParams.numSegmentsPerCurve + 1].z << std::endl;
        }

        // Final Position
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve].x = m_upInitialCurve->m_targetPos.x;
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve].y = m_upInitialCurve->m_targetPos.y;
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve].z = m_upInitialCurve->m_targetPos.z;
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve].w = 0.0f;
        //std::cout << "Inital Curve Seg " << m_experimentParams.numSegmentsPerCurve << "x: " << initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve].x << std::endl;
        //std::cout << "Inital Curve Seg " << m_experimentParams.numSegmentsPerCurve << "y: " << initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve].y << std::endl;
        //std::cout << "Inital Curve Seg " << m_experimentParams.numSegmentsPerCurve << "z: " << initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve].z << std::endl;

        // Final Tangents
        // TODO: Verify this math
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve * 2 + 1].x = m_upInitialCurve->m_targetTangent.x;
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve * 2 + 1].y = m_upInitialCurve->m_targetTangent.y;
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve * 2 + 1].z = m_upInitialCurve->m_targetTangent.z;
        initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve * 2 + 1].w = 0.0f;

        //std::cout << "Inital Curve Seg " << m_experimentParams.numSegmentsPerCurve << "x: " << initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve * 2 + 1].x << std::endl;
        //std::cout << "Inital Curve Seg " << m_experimentParams.numSegmentsPerCurve << "y: " << initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve * 2 + 1].y << std::endl;
        //std::cout << "Inital Curve Seg " << m_experimentParams.numSegmentsPerCurve << "z: " << initialCurvePositionsAndTangents[m_experimentParams.numSegmentsPerCurve * 2 + 1].z << std::endl;


        // Now, we want to copy all the data over to the gpu.
        // Copy over initial curve
        CudaSafeErrorCheck(cudaMemcpyToSymbol(device_constant_InitialCurve, (float*)initialCurvePositionsAndTangents.data(), sizeof(float) * 4 * (m_experimentParams.numSegmentsPerCurve + 1) * 2, 0, cudaMemcpyHostToDevice), "Copy initial curve to constant");
        // TODO: Ensure we need this
        // Make sure the values are copied into the device constant memory before we move on.
        CudaSafeErrorCheck(cudaDeviceSynchronize(), "Synchonize copy of initial curve to const memory");

        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerPathCompressedWeightGlobal, sizeof(float) * m_experimentParams.numPathsInExperiment), "Failed to allocate global path weight buffer");

        return true;
    }

    bool GpuFullExperimentRunnerOptimized::SetupWeightLookupTexture(const twisty::PathSpaceUtils::LogWeightLookupTableIntegral& lookupEvaluator)
    {
        auto& weightValues = lookupEvaluator.AccessLookupTable();

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

        m_pWeightLookupArray = nullptr;
        CudaSafeErrorCheck(
            cudaMallocArray(&m_pWeightLookupArray, &channelDesc, weightValues.size(), 1), 
            "Malloc weight texture array");

        CudaSafeErrorCheck(
            cudaMemcpyToArray(m_pWeightLookupArray, 0, 0, weightValues.data(), sizeof(float) * weightValues.size(), cudaMemcpyHostToDevice),
            "Copy weight lookup values to array");


        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = m_pWeightLookupArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
    
        // Create the actual texture object
        CudaSafeErrorCheck(
            cudaCreateTextureObject(&m_weightTextureObj, &resDesc, &texDesc, NULL),
            "Create texture object error");

        return true;
    }

    void GpuFullExperimentRunnerOptimized::CleanupWeightLookupTexture()
    {
        CudaSafeErrorCheck(
            cudaDestroyTextureObject(m_weightTextureObj),
            "Destroy texture object error");

        CudaSafeErrorCheck(
            cudaFreeArray(m_pWeightLookupArray),
            "Delete weight array error");
    }

    ExperimentRunner::ExperimentResults GpuFullExperimentRunnerOptimized::RunExperiment()
    {
        // A value of 1000 failed to work, not enough shared mem.
        // A value of 10 seems to work nicely
        // TODO: Play with this more to find a better value
        const uint32_t numCachedPathWeightsShardMem = 1;

        // Calculate grid and block sizes based on the kernels we will call
        uint32_t numGlobalPerturbThreads = 0;
        uint32_t perturbBlockSize = 0;
        uint32_t perturbGridSize = 0;
        bool result = SetupKernelDispatchParameters(numCachedPathWeightsShardMem, numGlobalPerturbThreads, perturbBlockSize, perturbGridSize);
        if (!result)
        {
            printf("Failed to setup kernel dispatch parameters\n");
            return {};
        }

        result = SetupCuRandStates(numGlobalPerturbThreads);
        if (!result)
        {
            printf("Failed to setup curand states\n");
            return {};
        }

        result = SetupCurveDataStructures();
        if (!result)
        {
            printf("Failed to setup curve device data structures\n");
            return {};
        }

        uint32_t numFailures = 0;
        uint32_t totalFailures = 0;
        uint32_t totalSuccess = 0;

        // Say that we will start outputing the path batch output
        const double mu = 0.1;
        const uint32_t numStepsInt = 2000;
        const double minBound = 0.0;
        const double maxBound = 100.0;
        const double eps = 0.5f;

        float ds = m_upInitialCurve->m_arclength / 200.0f;
        float scatter = 0.08f / ds;
        float absorbtion = 0.0f;

        double minCurvature = 0.0f;
        double maxCurvature = (2.0f / (m_upInitialCurve->m_arclength / m_upInitialCurve->m_numSegments)) * 1.1;
        uint32_t numCurvatureSteps = 10000;

        twisty::PathSpaceUtils::LogWeightLookupTableIntegral lookupEvaluator(ds, mu, numStepsInt, minBound, maxBound, eps,
            minCurvature, maxCurvature, numCurvatureSteps, scatter);

        result = SetupWeightLookupTexture(lookupEvaluator);
        if (!result)
        {
            printf("Failed to setup weight lookup texture\n");
            return {};
        }

        // At this point, we know how many curves we'll want to generate. So, we setup our parameters to handle this.
        std::cout << "Experiment Information: " << std::endl;

        uint32_t numPathsPerThread = (m_experimentParams.numPathsInExperiment + numGlobalPerturbThreads - 1) / numGlobalPerturbThreads;
        std::cout << "\tNum paths generated per global thread: " << numPathsPerThread << std::endl;


        // Dispatch CurandState
        // We need a dispatch that initializes curand per thread
        {
            dim3 gridSize(perturbGridSize, 1, 1);
            dim3 blockSize(perturbBlockSize, 1, 1);
            size_t sharedMemorySizeBytes = 0;
            {
                size_t curveWorkspaceSize = 2 * (m_experimentParams.numSegmentsPerCurve + 1) * 4 * 4;
                size_t cachedPathWeights = 4 * numCachedPathWeightsShardMem;
                sharedMemorySizeBytes = perturbBlockSize * (curveWorkspaceSize + cachedPathWeights);
            }
            cudaStream_t stream = 0;

            /*
            __global__ void GeneratePathBatchPutrubations(
            uint32_t numExperimentPaths,
            uint32_t numPathsPerThread,
            uint32_t numCachedPathWeightsPerThread,
            uint32_t numSegmentsPerCurve,
            curandState_t* pRandStates, 
            float* pGlobalPathWeights,
            float segmentLength
            )
            */

            GeneratePathBatchPutrubations << <gridSize, blockSize, sharedMemorySizeBytes, stream >> > (
                m_experimentParams.numPathsInExperiment,
                numPathsPerThread,
                numCachedPathWeightsShardMem,
                m_experimentParams.numSegmentsPerCurve,
                m_pPerGlobalThreadRandStates,
                m_pPerPathCompressedWeightGlobal,
                m_upInitialCurve->m_segmentLength,
                scatter,
                absorbtion,
                m_weightTextureObj,
                minCurvature,
                maxCurvature
            );

            CudaSafeErrorCheck(cudaGetLastError(), "Perturb kernal launch error");
            CudaSafeErrorCheck(cudaDeviceSynchronize(), "Perturb kernel sync error");

            std::cout << "Done with the perturb phase" << std::endl;
        }

        //TODO: Read back weights
        std::vector<float> compressedWeightBuffer(m_experimentParams.numPathsInExperiment);
        CudaSafeErrorCheck(cudaMemcpy(compressedWeightBuffer.data(), m_pPerPathCompressedWeightGlobal, sizeof(float) * m_experimentParams.numPathsInExperiment, cudaMemcpyDeviceToHost), "Copy back compressed weights from device");

        twisty::BigFloat totalExperimentWeight = 0.0f;
        for (auto& compressedValue : compressedWeightBuffer)
        {
            twisty::BigFloat bigfloatCompressed = compressedValue;
            totalExperimentWeight += boost::multiprecision::exp(bigfloatCompressed);
        }

        std::cout << "Total experiment weight: " << totalExperimentWeight << std::endl;

        ExperimentResults results;
        BigFloat denom = m_experimentParams.numPathsInExperiment;
        results.experimentWeight = totalExperimentWeight / denom;
        return results;
    }

    void GpuFullExperimentRunnerOptimized::Shutdown()
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