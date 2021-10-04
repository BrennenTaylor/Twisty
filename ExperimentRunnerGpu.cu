#include "ExperimentRunnerGpu.h"
#include "CurveUtils.h"

#include "DeviceCurve.h"
#include "Twisty_Cuda_Helpers.h"

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <cminpack.h>

#include <assert.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace twisty
{
    // Dispatch kernel for initializing the curand states
    // This really should be generalized or better thought out for experimentation sake
    __global__ void InitializeCurandState(uint32_t seed, curandState_t* pStates, uint32_t maxNumStates)
    {
        uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
        if (tidx < maxNumStates)
        {
            curand_init(seed, blockIdx.x, 0, &pStates[tidx]);
        }
    }

    // Dispatch which atually runs the purtibation algorithm on the GPU
    __global__ void GeneratePathBatchPutrubations(
        uint32_t numPathBatchPaths,
        uint32_t numPathsPerBlockThread,
        ExperimentRunner::ExperimentSegmentTorsion* pWriteSegments,
        ExperimentRunner::ExperimentSegmentTorsion* pInitalSegmentValuesArray,
        DeviceCurve* pSeedCurveInfo,
        curandState_t* pStates)
    {
        // First, we calculate the block thread index of the entire gpu dispatch.
        // For the number of threads dispatched, each is responsible for a number of 
        // paths to generate, i.e. pathBatchSize / numberOfTotalThreads
        uint32_t blockThreadIdx = threadIdx.x + blockIdx.x * blockDim.x;
        
        
        const uint32_t allowedFailures = 100;


        uint32_t blockThreadPathIdx = 0;
        uint32_t numFailures = 0;
        while(blockThreadPathIdx < numPathsPerBlockThread && numFailures < allowedFailures)
        {
            // This is read only memory
            // TODO: Assuming only 1 for now, but make this generic
            const ExperimentRunner::ExperimentSegmentTorsion* const pThreadInitialSegmentValues = pInitalSegmentValuesArray;

            // Treat memory as grouped by block thread paths at the lowest level
            ExperimentRunner::ExperimentSegmentTorsion* pCurrentPathWritingLocation = pWriteSegments
                + blockThreadIdx * numPathsPerBlockThread * pSeedCurveInfo->m_numSegments
                + blockThreadPathIdx * pSeedCurveInfo->m_numSegments;

            // For now, just write all values to be 1.0 for curvature and torsion
            for (uint32_t segIdx = 0; segIdx < pSeedCurveInfo->m_numSegments; ++segIdx)
            {
                pCurrentPathWritingLocation[segIdx].m_curvature = pThreadInitialSegmentValues[segIdx].m_curvature;
                //pCurrentPathWritingLocation[segIdx].m_torsion = pThreadInitialSegmentValues[segIdx].m_torsion;
            }


            //// Here, we want to select the node indices first
            uint32_t centerMin = 1;
            uint32_t centerMax = pSeedCurveInfo->m_numSegments - 2;
            uint32_t centerNodeIdx = centerMin + curand_uniform(&pStates[blockThreadIdx]) * (centerMax - centerMin);

            uint32_t leftMin = 0;
            uint32_t leftMax = centerNodeIdx - 1;
            uint32_t leftNodeIdx = leftMin + curand_uniform(&pStates[blockThreadIdx]) * (leftMax - leftMin);

            uint32_t rightMin = centerNodeIdx + 1;
            uint32_t rightMax = pSeedCurveInfo->m_numSegments - 1;
            uint32_t rightNodeIdx = rightMin + curand_uniform(&pStates[blockThreadIdx]) * (rightMax - rightMin);

            float randCurvature = curand_uniform(&pStates[blockThreadIdx]) * kmax;


            //// Implement the purturb algorithm
            //// Verify the path is within a random range
            //// NOTE: Temporarily randomly accept for now
            float validity = curand_uniform(&pStates[blockThreadIdx]);
            float acceptanceRate = 0.99f;
            if (validity > (acceptanceRate))
            {
                numFailures++;
                continue;
            }

            // Move to next path for block thread and reset failure count
            blockThreadPathIdx++;
            numFailures = 0;
        }
    }

    ExperimentRunnerGpu::ExperimentRunnerGpu(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper, uint32_t pathBatchSize)
        : ExperimentRunner(experimentParams, bootstrapper )
        , m_numSMs(0)
        , m_warpSize(0)
        , m_maxThreadsPerMultiprocessor(0)
        , m_gridSizePurturbKernel(0)
        , m_blockSizePurturbKernel(0)
        , m_numBlockThreadsInPurturbDispatch(0)
        , m_gridSizeRandKernel(0)
        , m_blockSizeRandKernel(0)
        , m_pPathBatchWrite_Device{ nullptr }
        , m_pInitialPathSegmentValues{ nullptr }
        //, m_pathBatchHostServer()
        , m_pathBatchReciever_Host()
        , m_initialSegmentDataServer()
        // Cude memory stuff
        , m_pSharedCurveInfo( nullptr )
        , m_pPerBlockThreadRandStates(nullptr)
        , m_pPerBlockThreadRotationScratchpad(nullptr)
        , m_pPerPathCurvatureAndTorsionFront(nullptr)
        , m_pPerPathCurvatureAndTorsionBack(nullptr)
    {
    }

    ExperimentRunnerGpu::~ExperimentRunnerGpu()
    {
    }

    // Based on the number of segments, we set up the cuda devices and memory which will be used.
    bool ExperimentRunnerGpu::Setup()
    {
        bool result = SetupCudaDevice();
        if (!result)
        {
            printf("Failed to setup cuda device\n");
            return false;
        }

        // Loop until we generate a good candidate for curve generation
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
            std::cout << "Seed curve error: " << curveError << std::endl;

            if (curveError <= m_experimentParams.maximumBootstrapCurveError)
            {
                successfulGen = true;
            }
        }

        // Calculate grid and block sizes based on the kernels we will call
        result = SetupKernelDispatchParameters();
        if (!result)
        {
            printf("Failed to setup kernel dispatch parameters\n");
            return false;
        }

        result = SetupCuRandStates();
        if (!result)
        {
            printf("Failed to setup curand states\n");
            return false;
        }

        result = SetupCurveDataStructures();
        if (!result)
        {
            printf("Failed to setup curve device data structures\n");
            return false;
        }

        return true;
    }

    // This sets up the cuda device for use. This could be pulled out into a more general class.
    bool ExperimentRunnerGpu::SetupCudaDevice()
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
            printf("Device Number: %d\n", i);
            printf("\tDevice name: %s\n", prop.name);
            printf("\tSM Count: %d\n", prop.multiProcessorCount);
            printf("\tWarp Size: %d\n", prop.warpSize);
            printf("\tThreads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
            printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
            printf("\tPeak Memory Bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        }

        // We select the first device only
        const uint32_t selectedDeviceIdx = 0;
        std::cout << "Selected device: " << selectedDeviceIdx << std::endl;
        cudaDeviceProp deviceProp;
        CudaSafeErrorCheck(cudaGetDeviceProperties(&deviceProp, 0), "Get first device prop");

        m_numSMs = deviceProp.multiProcessorCount;
        m_warpSize = deviceProp.warpSize;
        m_maxThreadsPerMultiprocessor = deviceProp.maxThreadsPerMultiProcessor;

        return true;
    }

    // We calculate the dispatch parameters based off kernel complexity
    bool ExperimentRunnerGpu::SetupKernelDispatchParameters()
    {
        // Calculate minimum grid size and block size required to achieves maximum potential occupancy for GeneratePathBatchPutrubations
        {
            int blockSizePurturbKernel = 0;
            int minGridSizePurturbKernel = 0;
            size_t sharedMemoryUse = 0;
            size_t maxBlockSize = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSizePurturbKernel, &blockSizePurturbKernel,
                GeneratePathBatchPutrubations, sharedMemoryUse, maxBlockSize);
            std::cout << "GeneratePutrubations: " << std::endl;
            std::cout << "\tMax Block Size: " << blockSizePurturbKernel << std::endl;
            std::cout << "\tMin Grid Size: " << minGridSizePurturbKernel << std::endl;

            // Assume we generate one path per thread
            m_gridSizePurturbKernel = minGridSizePurturbKernel;
            m_blockSizePurturbKernel = blockSizePurturbKernel;

            m_numBlockThreadsInPurturbDispatch = m_gridSizePurturbKernel * m_blockSizePurturbKernel;

            printf("\tGeneratePathBatchPutrubations Grid Size: %d\n", m_gridSizePurturbKernel);
            printf("\tGeneratePutrubations Block Size: %d\n", m_blockSizePurturbKernel);
            printf("\tGeneratePutrubations Num Block Threads: %d\n", m_numBlockThreadsInPurturbDispatch);
        }
        // Random number setup
        {
            int blockSizeRandKernel = 0;
            int minGridSizeRandKernel = 0;
            size_t sharedMemoryUse = 0;
            size_t maxBlockSize = 0;
            cudaOccupancyMaxPotentialBlockSize(&minGridSizeRandKernel, &blockSizeRandKernel, InitializeCurandState, sharedMemoryUse, maxBlockSize);
            std::cout << "InitializeCurandState: " << std::endl;
            std::cout << "\tBlock Size: " << blockSizeRandKernel << std::endl;
            std::cout << "\tMin Grid Size: " << minGridSizeRandKernel << std::endl;
            m_gridSizeRandKernel = ((m_gridSizePurturbKernel * m_blockSizePurturbKernel) + blockSizeRandKernel - 1) / blockSizeRandKernel;
            m_blockSizeRandKernel = blockSizeRandKernel;

            printf("\tInitializeCurandState Grid Size: %d\n", m_gridSizeRandKernel);
            printf("\tInitializeCurandState Block Size: %d\n", m_blockSizeRandKernel);
        }
        return true;
    }

    bool ExperimentRunnerGpu::SetupCuRandStates()
    {
        // Random Seed Kernel
        // Every block thread needs its own curand state
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPerBlockThreadRandStates,
            sizeof(curandState_t) * m_numBlockThreadsInPurturbDispatch),
            "Random state malloc");

        printf("\tDispatch InitializeCurandState as <<<%d, %d>>>\n", m_gridSizeRandKernel, m_blockSizeRandKernel);
        InitializeCurandState << <m_gridSizeRandKernel, m_blockSizeRandKernel >> > (static_cast<uint32_t>(time(0)),
            m_pPerBlockThreadRandStates, m_numBlockThreadsInPurturbDispatch);
        CudaSafeErrorCheck(cudaGetLastError(), "Rand state init kernal launch");
        CudaSafeErrorCheck(cudaDeviceSynchronize(), "Rand state kernel sync");

        int maxActiveBlocksRandKernel = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksRandKernel, InitializeCurandState, m_blockSizeRandKernel, 0);
        float randKernelOccupancy = (maxActiveBlocksRandKernel * m_blockSizeRandKernel / m_warpSize) / static_cast<float>(m_maxThreadsPerMultiprocessor / m_warpSize);
        printf("\tLaunched blocks of size %d\n", m_blockSizeRandKernel);
        printf("\tRand Kernel occupancy: %f\n", randKernelOccupancy);
        return true;
    }

    bool ExperimentRunnerGpu::SetupCurveDataStructures()
    {
        // All threads share this as read only memeory
        // Only store this information once as a result.
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pSharedCurveInfo, sizeof(DeviceCurve)),
            "Curve Head cumalloc");

        // Now, we want to copy all the data over to the gpu.
        // Copy over curve head
        // This will never change throughout the experiment
        {
            DeviceCurve curveHead;
            memset(&curveHead, 0, sizeof(curveHead));

            curveHead.m_arclength = m_upInitialCurve->m_arclength;
            curveHead.m_numSegments = m_upInitialCurve->m_numSegments;
            curveHead.m_basePos.x = m_upInitialCurve->m_basePos.x;
            curveHead.m_basePos.y = m_upInitialCurve->m_basePos.y;
            curveHead.m_basePos.z = m_upInitialCurve->m_basePos.z;
            curveHead.m_baseTangent.x = m_upInitialCurve->m_baseTangent.x;
            curveHead.m_baseTangent.y = m_upInitialCurve->m_baseTangent.y;
            curveHead.m_baseTangent.z = m_upInitialCurve->m_baseTangent.z;
            curveHead.m_baseNormal.x = m_upInitialCurve->m_baseNormal.x;
            curveHead.m_baseNormal.y = m_upInitialCurve->m_baseNormal.y;
            curveHead.m_baseNormal.z = m_upInitialCurve->m_baseNormal.z;
            curveHead.m_baseBinormal.x = m_upInitialCurve->m_baseBinormal.x;
            curveHead.m_baseBinormal.y = m_upInitialCurve->m_baseBinormal.y;
            curveHead.m_baseBinormal.z = m_upInitialCurve->m_baseBinormal.z;
            curveHead.m_targetPos.x = m_upInitialCurve->m_targetPos.x;
            curveHead.m_targetPos.y = m_upInitialCurve->m_targetPos.y;
            curveHead.m_targetPos.z = m_upInitialCurve->m_targetPos.z;
            curveHead.m_targetTangent.x = m_upInitialCurve->m_targetTangent.x;
            curveHead.m_targetTangent.y = m_upInitialCurve->m_targetTangent.y;
            curveHead.m_targetTangent.z = m_upInitialCurve->m_targetTangent.z;
            // TODO: Set these correctly
            curveHead.m_minCurvature = 0.0f;
            curveHead.m_maxCurvature = 0.0f;
            curveHead.m_minTorsion = 0.0f;
            curveHead.m_maxTorsion = 0.0f;

            // Copy shared curve head memory
            CudaSafeErrorCheck(cudaMemcpy(m_pSharedCurveInfo, &curveHead,
                sizeof(DeviceCurve), cudaMemcpyHostToDevice), "Curve head to device copy");
        }
        return true;
    }

    // This function is responsible for preparing all memory for the next path batch to be generated on the GPU
    void ExperimentRunnerGpu::ResetForPathBatch(const PathBatch& pathBatch)
    {
        // We want to free device's path batch memory for active writing so we can reallocate it to match the size of the current path batch we are generating
        if (m_pPathBatchWrite_Device)
        {
            cudaFree(m_pPathBatchWrite_Device);
        }
        if (m_pInitialPathSegmentValues)
        {
            cudaFree(m_pInitialPathSegmentValues);
        }

        // Calculate the number of paths per thread which will be generated
        const uint32_t pathsPerBlockThread = (pathBatch.numberOfPathsInBatch + m_numBlockThreadsInPurturbDispatch - 1)
            / m_numBlockThreadsInPurturbDispatch;
        const uint32_t numTotalPossiblePathsForGeneration = pathsPerBlockThread * m_numBlockThreadsInPurturbDispatch;
        std::cout << "\tPaths to generate for pathbatch: " << pathBatch.numberOfPathsInBatch << std::endl;
        std::cout << "\tPaths to generate per block thread: " << pathsPerBlockThread << std::endl;
        std::cout << "\tpathsPerBlockThread * m_numBlockThreadsInPurturbDispatch: " << numTotalPossiblePathsForGeneration << std::endl;

        //TODO: Update this to be based on the number of segments in the path batch
        // We allocate and initialize in the reset
        CudaSafeErrorCheck(cudaMalloc((void**)&m_pPathBatchWrite_Device,
            sizeof(ExperimentSegmentTorsion) * m_experimentParams.numSegmentsPerCurve * numTotalPossiblePathsForGeneration),
            "Curve Segments Front cumalloc");

        {
            uint32_t numInitialSegments = 1;
            m_initialSegmentDataServer.resize(numInitialSegments * m_experimentParams.numSegmentsPerCurve);
            auto& segments = m_upInitialCurve->m_segments;
            for (uint32_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve * numInitialSegments; ++segmentIdx)
            {
                uint32_t initialCurveSegIdx = segmentIdx % m_experimentParams.numSegmentsPerCurve;
                m_initialSegmentDataServer[segmentIdx].m_curvature = segments[initialCurveSegIdx].m_curvature;
                //m_initialSegmentDataServer[segmentIdx].m_torsion = segments[initialCurveSegIdx].m_torsion;
            }


            // Set up the device memory for that
            CudaSafeErrorCheck(cudaMalloc((void**)&m_pInitialPathSegmentValues,
                sizeof(ExperimentSegmentTorsion) * m_experimentParams.numSegmentsPerCurve * numInitialSegments),
                "Initial path segment values"
            );


            {
                CudaSafeErrorCheck(cudaMemcpy(m_pInitialPathSegmentValues, m_initialSegmentDataServer.data(),
                    sizeof(ExperimentSegmentTorsion) * m_experimentParams.numSegmentsPerCurve * numInitialSegments,
                    cudaMemcpyHostToDevice),
                "Curve front alloc");
            }
        }

        // Finally, we want to create the path batch buffers
        {
            //m_pathBatchHostServer.resize(m_experimentParams.numSegmentsPerCurve * pathBatch.numberOfPathsInBatch);
            m_pathBatchReciever_Host.resize(m_experimentParams.numSegmentsPerCurve * numTotalPossiblePathsForGeneration);

            // Zero it out for now, but this could be removed later probably
            for (uint32_t i = 0; i < m_experimentParams.numSegmentsPerCurve * numTotalPossiblePathsForGeneration; ++i)
            {
                m_pathBatchReciever_Host[i].m_curvature = 0;
                //m_pathBatchReciever_Host[i].m_torsion = 0;
            }
        }
    }

    void ExperimentRunnerGpu::DoPathBatch(PathBatch& pathBatch)
    {
        printf("Doing Path Batch %d\n", pathBatch.index);

        ResetForPathBatch(pathBatch);

        // Generates a path batch set on the GPU
        GeneratePathBatch(pathBatch);

        // Copy that path batch back to CPU
        TransferPathBatch(pathBatch);
    }

    ExperimentRunner::ExperimentResults ExperimentRunnerGpu::RunExperiment()
    {
        /* So, it works like this...
            1. Dispatch a group of threads to generate purturbed paths
                We assume that the seed paths are set up correctly in near and back buffers from setup
        */

        if (!BeginPathBatchOutput())
        {
            printf("Failed to write out path batch data");
        }

        auto experimentStartTime = std::chrono::high_resolution_clock::now();

        // First, we calculate how many paths we need total
        // We have a number of curves we will discard from the seed curve... however, is this really needed?
        const uint32_t numTotalPaths = m_experimentParams.numPathsInExperiment;
        // Calulate the number of required path batches, this is the unit we work with at once on the gpu and for transfering back
        const uint32_t numPathBatchPasses = (numTotalPaths + m_experimentParams.maxPathBatchSize - 1) / m_experimentParams.maxPathBatchSize;

        printf("Num Required Path Batches: %d\n", numPathBatchPasses);

        // Currently, this entire process is serial
        uint32_t numPathsGenerated = 0;
        for (uint32_t pathBatchIdx = 0; pathBatchIdx < numPathBatchPasses; ++pathBatchIdx)
        {
            auto pathBatchStartTime = std::chrono::high_resolution_clock::now();

            PathBatch pathBatch;
            pathBatch.index = pathBatchIdx;
            pathBatch.numberOfPathsInBatch = m_experimentParams.maxPathBatchSize;
            const uint32_t numPathsLeft = m_experimentParams.numPathsInExperiment - numPathsGenerated;
            if (numPathsLeft < pathBatch.numberOfPathsInBatch)
            {
                pathBatch.numberOfPathsInBatch = numPathsLeft;
            }
            pathBatch.pathBatchSegments = KTSegments(pathBatch.numberOfPathsInBatch * m_experimentParams.numSegmentsPerCurve);
            pathBatch.perPathVailidity = std::vector<bool>(pathBatch.numberOfPathsInBatch);

            DoPathBatch(pathBatch);

            auto pathBatchEndTime = std::chrono::high_resolution_clock::now();

            auto elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(pathBatchEndTime - pathBatchStartTime);
            std::cout << "Time for path batch (ms): " << elapsedMS.count() << std::endl;
            auto elapsedS = std::chrono::duration_cast<std::chrono::seconds>(pathBatchEndTime - pathBatchStartTime);
            std::cout << "Time for path batch (s): " << elapsedS.count() << std::endl;

            OutputPathBatch(pathBatch);

            numPathsGenerated += pathBatch.numberOfPathsInBatch;
        }

        auto experimentEndTime = std::chrono::high_resolution_clock::now();
        auto elapsedMS = std::chrono::duration_cast<std::chrono::milliseconds>(experimentEndTime - experimentStartTime);
        std::cout << "Time for experiment (ms): " << elapsedMS.count() << std::endl;
        auto elapsedS = std::chrono::duration_cast<std::chrono::seconds>(experimentEndTime - experimentStartTime);
        std::cout << "Time for experiment (s): " << elapsedS.count() << std::endl;

        EndPathBatchOutput();

        return {};
    }

    void ExperimentRunnerGpu::Shutdown()
    {
        // We should be able to do this as there is no cuda dispatch going on
        // Free all cuda resources
        if (m_pPathBatchWrite_Device)
        {
            cudaFree(m_pPathBatchWrite_Device);

        }
        if (m_pInitialPathSegmentValues)
        {
            cudaFree(m_pInitialPathSegmentValues);

        }
        if (m_pSharedCurveInfo)
        {
            cudaFree(m_pSharedCurveInfo);

        }
    }

    // TODO: There is a memory bug in here somewhere I believe... Need to find it
    void ExperimentRunnerGpu::GeneratePathBatch(const PathBatch& pathBatch)
    {
        // Calculate the number of paths per thread which will be generated
        const uint32_t pathsPerBlockThread = (pathBatch.numberOfPathsInBatch + m_numBlockThreadsInPurturbDispatch - 1)
            / m_numBlockThreadsInPurturbDispatch;
        const uint32_t numTotalPossiblePathsForGeneration = pathsPerBlockThread * m_numBlockThreadsInPurturbDispatch;

        printf("\tNum paths in path batch: %d\n", pathBatch.numberOfPathsInBatch);
        printf("\tNum possible paths in path batch: %d\n", numTotalPossiblePathsForGeneration);
        printf("\tDispatching GeneratePathBatchPutrubations as <<<%d, %d>>>\n", m_gridSizePurturbKernel, m_blockSizePurturbKernel);
        
        // This is a dispatch which actually calls the functions to be exeuted on the gpu.
        // All pointers here must be accessable by the GPU
        GeneratePathBatchPutrubations <<<m_gridSizePurturbKernel, m_blockSizePurturbKernel>>>(
            pathBatch.numberOfPathsInBatch,
            pathsPerBlockThread,
            m_pPathBatchWrite_Device,
            m_pInitialPathSegmentValues,
            m_pSharedCurveInfo,
            m_pPerBlockThreadRandStates
        );
        CudaSafeErrorCheck(cudaPeekAtLastError(), "Purturb Launch");
        CudaSafeErrorCheck(cudaDeviceSynchronize(), "Purturb Sync");

        // We select the first device only
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        int maxActiveBlocksPurturbKernel;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPurturbKernel, GeneratePathBatchPutrubations, m_blockSizePurturbKernel, 0);
        float purturbKernelOccupancy = (maxActiveBlocksPurturbKernel * m_blockSizePurturbKernel / m_warpSize) / static_cast<float>(deviceProp.maxThreadsPerMultiProcessor / m_warpSize);
        printf("\tLaunched blocks of size %d\n", m_blockSizePurturbKernel);
        printf("\tPurturb Kernel occupancy: %f\n", purturbKernelOccupancy);
    }

    void ExperimentRunnerGpu::TransferPathBatch(PathBatch& pathBatch)
    {
        const uint32_t pathsPerBlockThread = (pathBatch.numberOfPathsInBatch + m_numBlockThreadsInPurturbDispatch - 1)
            / m_numBlockThreadsInPurturbDispatch;
        const uint32_t numTotalPossiblePathsForGeneration = pathsPerBlockThread * m_numBlockThreadsInPurturbDispatch;

        // Swap the front and back path batchs
        //std::swap(m_pPathBatchWrite_Device, m_pPathBatchRead_Device);
        CudaSafeErrorCheck(cudaMemcpy(&m_pathBatchReciever_Host[0], m_pPathBatchWrite_Device,
            sizeof(ExperimentSegmentTorsion) * m_experimentParams.numSegmentsPerCurve * numTotalPossiblePathsForGeneration,
            cudaMemcpyDeviceToHost),
            "Copy path batch from device to host"
        );

        // Read out the segments
        uint32_t pathBatchPathCounter = 0;

        // Once that is done, we can put all the paths into the path batch
        for (uint32_t perThreadIdx = 0; perThreadIdx < pathsPerBlockThread; ++perThreadIdx)
        {
            for (uint32_t blockThreadIdx = 0; blockThreadIdx < m_numBlockThreadsInPurturbDispatch; ++blockThreadIdx)
            {
                for (uint32_t segIdx = 0; segIdx < m_experimentParams.numSegmentsPerCurve; ++segIdx)
                {
                    pathBatch.pathBatchSegments[pathBatchPathCounter * m_experimentParams.numSegmentsPerCurve + segIdx] =
                        m_pathBatchReciever_Host[perThreadIdx * m_numBlockThreadsInPurturbDispatch * m_experimentParams.numSegmentsPerCurve + blockThreadIdx * m_experimentParams.numSegmentsPerCurve + segIdx];
                }
                pathBatch.perPathVailidity[pathBatchPathCounter] = true;
                pathBatchPathCounter++;
                if (pathBatchPathCounter >= pathBatch.numberOfPathsInBatch)
                {
                    break;
                }
            }
            if (pathBatchPathCounter >= pathBatch.numberOfPathsInBatch)
            {
                break;
            }
        }

        // We have extracted all the path batch info
    }
}