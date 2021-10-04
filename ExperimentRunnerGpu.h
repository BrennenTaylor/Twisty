#include "ExperimentRunner.h"

#include "PathWeightUtils.h"
#include "DeviceCurve.h"
#include "Range.h"

#include <curand.h>
#include <curand_kernel.h>

namespace twisty
{
    /**
     * @brief Implements an experiment runner which utalizes cuda and the GPU.
     *
     */
    class ExperimentRunnerGpu : public ExperimentRunner
    {
    public:
        ExperimentRunnerGpu(ExperimentRunner::ExperimentParameters& experimentparams,
            Bootstrapper& bootstrapper, uint32_t pathBatchSize);
        virtual ~ExperimentRunnerGpu();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

    private:
        bool SetupCudaDevice();
        bool SetupKernelDispatchParameters();
        bool SetupCuRandStates();
        bool SetupCurveDataStructures();

        void DoPathBatch(PathBatch& pathBatch);


        void ResetForPathBatch(const PathBatch& pathBatch);
        void GeneratePathBatch(const PathBatch& pathBatch);
        void TransferPathBatch(PathBatch& pathBatch);

    private:
        // Cuda device variables
        int32_t m_numSMs;
        int32_t m_warpSize;
        int32_t m_maxThreadsPerMultiprocessor;

        // Cuda Kernel Dispatch vars
        int m_gridSizeRandKernel;
        int m_blockSizeRandKernel;

        int m_gridSizePurturbKernel;
        int m_blockSizePurturbKernel;
        int m_numBlockThreadsInPurturbDispatch;

        // Experiment variables

        // We need a generated paths storage.
        // Basically, paths will be generated in blocks of N * NumThreads and read back to the cpu for weight generation.
        // During this time, we can dispatch the next set of blocks.

        // Back buffer, paths written to here
        ExperimentSegmentTorsion* m_pPathBatchWrite_Device;
        // Starting Segment Values, transfered to CPU
        ExperimentSegmentTorsion* m_pInitialPathSegmentValues;

        // Local copy of a path batch. This is a staging area for copying a path batch to or from
        //std::vector<ExperimentSegmentTorsion> m_pathBatchHostServer;
        std::vector<ExperimentSegmentTorsion> m_pathBatchReciever_Host;

        std::vector<ExperimentSegmentTorsion> m_initialSegmentDataServer;

        // Device Memory - Shared Among Threads
        DeviceCurve* m_pSharedCurveInfo;

        // Device Memory - Unique Per Thread
        curandState_t* m_pPerBlockThreadRandStates;
        float* m_pPerBlockThreadRotationScratchpad;

        // Device Memory - Unique Among Paths
        // This is read from in the purtubation stage
        float* m_pPerPathCurvatureAndTorsionFront;
        // This is written to in the purtubation stage
        float *m_pPerPathCurvatureAndTorsionBack;
    };
}