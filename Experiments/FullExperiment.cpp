#include "FullExperimentRunnerOptimalPerturb.h"

#if defined(USE_CUDA)
#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"
#endif

#include <Bootstrapper.h>
#include <MathConsts.h>
#include <PathWeightUtils.h>

#include <FMath/Vector3.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <filesystem>
#include <string>

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(
      const nlohmann::json &experimentConfig)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Hardocded values
    experimentParams.rotateInitialSeedCurveRadians = 0.0f;

    // Values loaded from the config file
    experimentParams.numPathsInExperiment
          = experimentConfig["experiment"]["experimentParams"]["pathsToGenerate"];


    experimentParams.numPathsToSkip
          = experimentConfig["experiment"]["experimentParams"]["pathsToSkip"];
    experimentParams.experimentName = experimentConfig["experiment"]["experimentParams"]["name"];
    experimentParams.experimentDirPath
          = experimentConfig["experiment"]["experimentParams"]["experimentDir"];
    experimentParams.experimentDirPath += "/" + experimentParams.experimentName;


    experimentParams.maxPerturbThreads
          = experimentConfig["experiment"]["experimentParams"]["maxPerturbThreads"];
    experimentParams.maxWeightThreads
          = experimentConfig["experiment"]["experimentParams"]["maxWeightThreads"];

    experimentParams.outputBigFloatWeights
          = experimentConfig["experiment"]["experimentParams"]["outputBigFloatWeights"];
    experimentParams.outputPathBatches
          = experimentConfig["experiment"]["experimentParams"]["outputPathBatches"];
    experimentParams.useGpu = experimentConfig["experiment"]["experimentParams"]["useGpu"];

    experimentParams.numSegmentsPerCurve
          = experimentConfig["experiment"]["experimentParams"]["numSegments"];
    experimentParams.arclength = experimentConfig["experiment"]["experimentParams"]["arclength"];

    // Seeds
    experimentParams.bootstrapSeed
          = experimentConfig["experiment"]["experimentParams"]["random"]["bootstrapSeed"];
    experimentParams.curvePurturbSeed
          = experimentConfig["experiment"]["experimentParams"]["random"]["perturbSeed"];

    if (experimentParams.bootstrapSeed == 0) {
        experimentParams.bootstrapSeed = time(0);
    }
    if (experimentParams.curvePurturbSeed == 0) {
        experimentParams.curvePurturbSeed = time(0);
    }


    // Weighting parameter stuff
    int weightFunction
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["weightFunction"];
    switch (weightFunction) {
        // Radiative Transfer weight function
        case 0: {
            experimentParams.weightingParameters.weightingMethod
                  = twisty::WeightingMethod::RadiativeTransfer;
        } break;

        // Simplified Model
        case 1: {
            experimentParams.weightingParameters.weightingMethod
                  = twisty::WeightingMethod::SimplifiedModel;
        } break;

        // Default to the simplified model
        default: {
            experimentParams.weightingParameters.weightingMethod
                  = twisty::WeightingMethod::RadiativeTransfer;
        } break;
    }

    // Perturb method stuff
    int perturbMethod = experimentConfig["experiment"]["experimentParams"]["perturbMethod"];
    switch (perturbMethod) {
        // Simplified Model
        case 1: {
            experimentParams.perturbMethod
                  = twisty::ExperimentRunner::PerturbMethod::GeometricMinCurvature;
        } break;

        // Simplified Model
        case 2: {
            experimentParams.perturbMethod
                  = twisty::ExperimentRunner::PerturbMethod::GeometricCombined;
        } break;

        // Default to the simplified model
        case 0:
        default: {
            experimentParams.perturbMethod
                  = twisty::ExperimentRunner::PerturbMethod::GeometricRandom;
        } break;
    }

    // Weighting parameter stuff
    experimentParams.weightingParameters.mu
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["mu"];
    experimentParams.weightingParameters.eps
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["eps"];
    experimentParams.weightingParameters.numStepsInt
          = (int)experimentConfig["experiment"]["experimentParams"]["weighting"]["numStepsInt"];
    experimentParams.weightingParameters.numCurvatureSteps = (int)
          experimentConfig["experiment"]["experimentParams"]["weighting"]["numCurvatureSteps"];
    experimentParams.weightingParameters.absorbtion
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["absorbtion"];


    auto &scatterValuesLookup
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["scatterValues"];

    std::vector<double> scatterValues;
    for (auto &elem : scatterValuesLookup)
        scatterValues.push_back(elem);
    experimentParams.weightingParameters.scatterValues = scatterValues;

    // TODO: Should these be configurable in the file?
    experimentParams.weightingParameters.minBound = 0.0;
    experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;


    return experimentParams;
}

int main(int argc, char *argv[])
{
    {
        if (argc < 2) {
            std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
            return 1;
        }
        std::fstream configFile(argv[1]);
        if (!configFile.is_open()) {
            std::cout << "Failed to open: " << argv[1] << std::endl;
            return 1;
        }

        nlohmann::json experimentConfig;
        configFile >> experimentConfig;

        twisty::ExperimentRunner::ExperimentParameters experimentParams
              = ParseExperimentParamsFromConfig(experimentConfig);

        if (!std::filesystem::exists(experimentParams.experimentDirPath)) {
            std::filesystem::create_directories(experimentParams.experimentDirPath);
        }
        std::cout << "experimentDirPath: " << experimentParams.experimentDirPath << std::endl;
        const std::string experimentCfgCopyFilename
              = std::string(experimentParams.experimentDirPath) + "/"
              + experimentParams.experimentName + ".json";

        if (!std::filesystem::exists(experimentCfgCopyFilename)) {
            std::filesystem::copy_file(argv[1], experimentCfgCopyFilename,
                  std::filesystem::copy_options::overwrite_existing);
        }

        twisty::PerturbUtils::BoundaryConditions experimentGeometry;
        {
            float x = experimentConfig["experiment"]["geometry"]["startPos"]["x"];
            float y = experimentConfig["experiment"]["geometry"]["startPos"]["y"];
            float z = experimentConfig["experiment"]["geometry"]["startPos"]["z"];
            experimentGeometry.m_startPos = Farlor::Vector3(x, y, z);
        }
        {
            float x = experimentConfig["experiment"]["geometry"]["startDir"]["x"];
            float y = experimentConfig["experiment"]["geometry"]["startDir"]["y"];
            float z = experimentConfig["experiment"]["geometry"]["startDir"]["z"];
            experimentGeometry.m_startDir = Farlor::Vector3(x, y, z).Normalized();
        }

        {
            float x = experimentConfig["experiment"]["geometry"]["endPos"]["x"];
            float y = experimentConfig["experiment"]["geometry"]["endPos"]["y"];
            float z = experimentConfig["experiment"]["geometry"]["endPos"]["z"];
            experimentGeometry.m_endPos = Farlor::Vector3(x, y, z);
        }
        {
            float x = experimentConfig["experiment"]["geometry"]["endDir"]["x"];
            float y = experimentConfig["experiment"]["geometry"]["endDir"]["y"];
            float z = experimentConfig["experiment"]["geometry"]["endDir"]["z"];
            experimentGeometry.m_endDir = Farlor::Vector3(x, y, z).Normalized();
        }
        experimentGeometry.arclength = experimentParams.arclength;

        // We run the experiment as well
        std::string outputDataFilename
              = std::string(experimentParams.experimentDirPath) + std::string("/Results.dat");
        std::ofstream ofs(outputDataFilename);

        experimentParams.perExperimentDirSubfolder = std::string("main");

        twisty::Bootstrapper bootstrapper(experimentGeometry);

        std::cout << "Experiment Path Count: " << experimentParams.numPathsInExperiment
                  << std::endl;

        std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = nullptr;

#if defined(USE_CUDA)
        if (experimentParams.useGpu) {
            upExperimentRunner
                  = std::make_unique<twisty::FullExperimentRunnerOptimalPerturbOptimized_GPU>(
                        experimentParams, bootstrapper);
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb_Gpu"
                      << std::endl;
        } else {
            upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(
                  experimentParams, bootstrapper);
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb" << std::endl;
        }
#else
        if (experimentParams.useGpu) {
            std::cout << "Error, gpu requested but not supported on this platform; defaulting "
                         "to CPU"
                      << std::endl;
        }
        upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(
              experimentParams, bootstrapper);
#endif

        auto start = std::chrono::high_resolution_clock::now();
        std::optional<twisty::ExperimentRunner::ExperimentResults> optionalResults
              = upExperimentRunner->RunExperiment();
        auto end = std::chrono::high_resolution_clock::now();
        auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto timeSec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        if (!optionalResults.has_value()) {
            std::cout << "Experiment failed: no results returned." << std::endl;
            return 1;
        }
        const twisty::ExperimentRunner::ExperimentResults &results = optionalResults.value();

        std::cout << "Paths Generated: " << results.totalPathsGenerated << std::endl;

        auto experimentWeights = results.experimentWeights;

        for (int scatterIdx = 0; scatterIdx < results.experimentWeights.size(); scatterIdx++) {
            std::cout << "\tTotal experiment weight " << scatterIdx << ": "
                      << results.experimentWeights[scatterIdx] << std::endl;
            std::cout << "\tAvg path weight " << scatterIdx << ": "
                      << results.experimentWeights[scatterIdx] / results.totalPathsGenerated
                      << std::endl;
            ofs << "Scatter value: "
                << experimentParams.weightingParameters.scatterValues[scatterIdx] << std::endl;
            ofs << "\tTotal experiment weight " << scatterIdx << ": "
                << results.experimentWeights[scatterIdx] << std::endl;
            ofs << "\tAvg path weight " << scatterIdx << ": "
                << results.experimentWeights[scatterIdx] / results.totalPathsGenerated << std::endl;
            ofs << "\tTotal experiment time (ms) "
                << twisty::format_duration(std::chrono::milliseconds(results.totalExperimentMs))
                << std::endl;
            ofs << "\tSetup time (ms) "
                << twisty::format_duration(std::chrono::milliseconds(results.setupExperimentMs))
                << std::endl;
            ofs << "\tPerturb time (ms) "
                << twisty::format_duration(std::chrono::milliseconds(results.perturbExperimentMs))
                << std::endl;
            ofs << "\tWeighting time (ms) "
                << twisty::format_duration(std::chrono::milliseconds(results.weightingExperimentMs))
                << std::endl;
        }
        ofs << std::endl;

        // Retrieve Data we want from experiment
        upExperimentRunner.reset(nullptr);
        ofs.close();
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    _CrtDumpMemoryLeaks();
    _CrtMemDumpAllObjectsSince(NULL);
#endif

    return 0;
}
