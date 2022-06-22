#pragma once

#include "Bootstrapper.h"

#include "PathWeightUtils.h"

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <chrono>
#include <string>
#include <sstream>


namespace twisty {
std::string format_duration(std::chrono::milliseconds ms);

std::string GetCurrentTimeForFileName();

void ProcessRandomSeed(uint32_t &seed);

class ExperimentRunner {
   public:
    struct PathBatch {
        std::vector<float> m_curvatures;
        std::vector<Farlor::Vector3> m_positions;
        std::vector<Farlor::Vector3> m_tangents;
        std::vector<bool> perPathVailidity;
        uint64_t numberOfPathsInBatch;
        uint64_t index;
    };

    enum class PerturbMethod { GeometricRandom, GeometricMinCurvature, GeometricCombined };

    struct ExperimentParameters {
        uint64_t numPathsInExperiment = 0;
        uint64_t numPathsPerBatch = 1000000;
        uint64_t numPathsToSkip = 0;
        uint32_t numSegmentsPerCurve = 3;
        float arclength = 0.0f;
        uint32_t exportPathBatchSize = 1000;
        bool exportGeneratedCurves = false;
        std::string experimentName = "Default_Experiment";
        std::string experimentDirPath = "./Default_Experiment/";
        std::string perExperimentDirSubfolder = "";
        std::string pathBatchPrepend = "";
        uint32_t bootstrapSeed = 0;
        uint32_t curvePurturbSeed = 0;
        twisty::WeightingParameters weightingParameters;
        float rotateInitialSeedCurveRadians = 0.0f;

        PerturbMethod perturbMethod = PerturbMethod::GeometricRandom;

        // Set to 0 if we default to max user machine threads
        int32_t maxPerturbThreads = 0;
        int32_t maxWeightThreads = 0;
        bool outputBigFloatWeights = false;
        bool outputPathBatches = false;

        bool useGpu = false;
    };

    struct ExperimentResults {
        // Double, float, whatever, weight value
        std::vector<boost::multiprecision::cpp_dec_float_100> experimentWeights;
        uint64_t totalPathsGenerated = 0;
        uint64_t numFailedPaths = 0;
        std::chrono::milliseconds::rep totalExperimentMs = 0;
        std::chrono::milliseconds::rep setupExperimentMs = 0;
        std::chrono::milliseconds::rep perturbExperimentMs = 0;
        std::chrono::milliseconds::rep weightingExperimentMs = 0;
    };

   public:
    ExperimentRunner(ExperimentParameters &experimentParams, Bootstrapper &bootstrapper);
    virtual ~ExperimentRunner();
    std::optional<ExperimentResults> RunExperiment();

    Curve *GetInitialCurvePtr() const { return m_upInitialCurve.get(); }

   protected:
    struct RunnerSpecificResults {
        std::optional<ExperimentResults> experimentResults;
        std::chrono::milliseconds::rep experimentRuntimeTotalMs = 0;
        std::chrono::milliseconds::rep setupMs = 0;
        std::chrono::milliseconds::rep runExperimentMs = 0;
        std::chrono::milliseconds::rep weightingMs = 0;
    };

    virtual RunnerSpecificResults RunnerSpecificRunExperiment() = 0;

    bool BeginPathBatchOutput();
    void OutputPathBatch(PathBatch &pathBatch);
    void EndPathBatchOutput();

    void StartWeightConvergenceWrite();
    void UpdateConvergenceWeight(const uint64_t numNewPaths,
          const boost::multiprecision::cpp_dec_float_100 weightContribution);
    void EndWeightConvergenceWrite();

   protected:
    ExperimentParameters &m_experimentParams;
    Bootstrapper &m_bootstrapper;
    std::unique_ptr<Curve> m_upInitialCurve;
    std::string m_pathBatchOutputPath;

    std::filesystem::path m_experimentDirPath;

    std::mutex m_exportPathBatchesMutex;
    std::ofstream m_curvesBinaryFile;
    std::ofstream m_curvesMetadataFile;

    std::ofstream m_weightConvergenceFile;
    uint64_t m_numWeightConvergencePaths = 0;
    boost::multiprecision::cpp_dec_float_100 m_weightConvergenceCombinedWeight = 0.0;

   private:
    nlohmann::json m_pathBatchJsonIndex;
};
}