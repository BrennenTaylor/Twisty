#include "FullExperimentRunnerOptimalPerturb.h"
#include "boost/multiprecision/cpp_dec_float.hpp"
#include <corecrt_math_defines.h>
#include <stdexcept>

#if defined(USE_CUDA)
#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"
#endif

#include "Bootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(
      const nlohmann::json &experimentConfig)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Values loaded from the config file
    experimentParams.numPathsInExperiment
          = experimentConfig["experiment"]["experimentParams"]["pathsToGenerate"];

    experimentParams.numPathsToSkip
          = experimentConfig["experiment"]["experimentParams"]["pathsToSkip"];
    experimentParams.experimentName = experimentConfig["experiment"]["experimentParams"]["name"];
    experimentParams.experimentDirPath
          = experimentConfig["experiment"]["experimentParams"]["experimentDir"];
    experimentParams.experimentDirPath += "/" + experimentParams.experimentName;
    experimentParams.experimentDirPath += +"/" + twisty::GetCurrentTimeForFileName() + "/";

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

    // Seeds
    experimentParams.bootstrapSeed
          = experimentConfig["experiment"]["experimentParams"]["random"]["bootstrapSeed"];
    experimentParams.curvePurturbSeed
          = experimentConfig["experiment"]["experimentParams"]["random"]["perturbSeed"];

    twisty::ProcessRandomSeed(experimentParams.bootstrapSeed);
    twisty::ProcessRandomSeed(experimentParams.curvePurturbSeed);

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
    experimentParams.weightingParameters.absorption
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["absorption"];

    experimentParams.weightingParameters.scatter
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["scatter"];

    // TODO: Should these be configurable in the file?
    experimentParams.weightingParameters.minBound = 0.0;
    experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;

    return experimentParams;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Call as: %s configFilename\n", argv[0]);
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

    const float sphereRadius
          = experimentConfig["experiment"]["beamSpreadExperiment"]["sphereRadius"];

    const float maxArclength
          = experimentConfig["experiment"]["beamSpreadExperiment"]["maxArclength"];

    const Farlor::Vector3 startPos(
          experimentConfig["experiment"]["beamSpreadExperiment"]["startPos"][0],
          experimentConfig["experiment"]["beamSpreadExperiment"]["startPos"][1],
          experimentConfig["experiment"]["beamSpreadExperiment"]["startPos"][2]);

    const Farlor::Vector3 startDir(
          experimentConfig["experiment"]["beamSpreadExperiment"]["startDir"][0],
          experimentConfig["experiment"]["beamSpreadExperiment"]["startDir"][1],
          experimentConfig["experiment"]["beamSpreadExperiment"]["startDir"][2]);

    const uint32_t numArclengths = 1;

    const uint32_t numInitialCurves
          = experimentConfig["experiment"]["beamSpreadExperiment"]["numInitialCurves"];
    const uint32_t numPerInitialCurve
          = experimentConfig["experiment"]["beamSpreadExperiment"]["numPerInitialCurve"];


    std::filesystem::path outputDirectoryPath = std::filesystem::current_path();
    outputDirectoryPath.append(experimentParams.experimentName);
    if (!std::filesystem::exists(outputDirectoryPath)) {
        std::filesystem::create_directory(outputDirectoryPath);
    }

    // Lets test uniform random directions on sampling hemisphere
    uint32_t targetSeed = experimentConfig["experiment"]["beamSpreadExperiment"]["targetSeed"];
    twisty::ProcessRandomSeed(targetSeed);

    std::mt19937 randomGenerator(targetSeed);
    std::uniform_real_distribution<float> uniformFloatGen(0.0f, 1.0f);

    const float n1 = (uniformFloatGen(randomGenerator) * 2.0f) - 1.0f;
    const float e1 = uniformFloatGen(randomGenerator);

    Farlor::Vector3 endDir(n1,
          std::cos(2 * M_PI * e1) * std::sqrt(1 - (n1 * n1)),
          std::sin(2 * M_PI * e1) * std::sqrt(1 - (n1 * n1)));
    Farlor::Vector3 endPos = startPos + sphereRadius * endDir;

    const float minArclength = twisty::Bootstrapper::CalculateMinimumArclength(
                                     experimentParams.numSegmentsPerCurve, startPos, endPos)
          * 1.01f;
    std::cout << "Min arclength: " << minArclength << std::endl;

    const float arclengthStepSize = (maxArclength - minArclength) / (numArclengths);

    boost::multiprecision::cpp_dec_float_100 outputValue = 0.0f;

    for (uint32_t arclengthIdx = 0; arclengthIdx < numArclengths; ++arclengthIdx) {
        double targetArclength = minArclength + arclengthStepSize * arclengthIdx;

        boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;

        std::mt19937 initialCurveGen(experimentParams.bootstrapSeed);
        for (uint32_t initialCurveIdx = 0; initialCurveIdx < numInitialCurves; ++initialCurveIdx) {
            uint32_t initialCurveSeed = 0;
            do {
                initialCurveSeed = initialCurveGen();
            } while (initialCurveSeed == 0);

            boost::multiprecision::cpp_dec_float_100 maxResult = 0.0;

            std::mt19937 perCurveGen(experimentParams.curvePurturbSeed);
            for (uint32_t perInitialCurveIdx = 0; perInitialCurveIdx < numPerInitialCurve;
                  ++perInitialCurveIdx) {
                uint32_t perCurveSeed = 0;
                do {
                    perCurveSeed = perCurveGen();
                } while (perCurveSeed == 0);

                twisty::PerturbUtils::BoundaryConditions experimentGeometry;
                experimentGeometry.m_startPos = startPos;
                experimentGeometry.m_startDir = startDir;
                experimentGeometry.m_endPos = endPos;
                experimentGeometry.m_endDir = endDir;
                experimentGeometry.arclength = targetArclength;

                experimentParams.arclength = experimentGeometry.arclength;

                twisty::Bootstrapper bootstrapper(experimentGeometry);

                std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = nullptr;
#if defined(USE_CUDA)
                if (experimentParams.useGpu) {
                    upExperimentRunner = std::make_unique<
                          twisty::FullExperimentRunnerOptimalPerturbOptimized_GPU>(
                          experimentParams, bootstrapper);
                    std::cout << "Selected Runner Method: "
                                 "FullExperimentRunnerOptimalPerturb_Gpu"
                              << std::endl;
                } else {
                    upExperimentRunner
                          = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(
                                experimentParams, bootstrapper);
                    std::cout << "Selected Runner Method: "
                                 "FullExperimentRunnerOptimalPerturb"
                              << std::endl;
                }
#else
                if (experimentParams.useGpu) {
                    std::cout << "Error, gpu requested but not supported on this "
                                 "platform; defaulting "
                                 "to CPU"
                              << std::endl;
                }
                upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(
                      experimentParams, bootstrapper);
#endif

                std::optional<twisty::ExperimentRunner::ExperimentResults> optionalResults
                      = upExperimentRunner->RunExperiment();
                if (!optionalResults.has_value()) {
                    std::cout << "Experiment failed: no results returned." << std::endl;
                    return 1;
                }
                const twisty::ExperimentRunner::ExperimentResults &results
                      = optionalResults.value();
                if (results.experimentWeight > maxResult) {
                    maxResult = results.experimentWeight;
                }
            }

            averagedResult += (maxResult * (1.0 / numInitialCurves));
        }
        outputValue += averagedResult * (1.0 / (numArclengths));
    }

    std::filesystem::path outputFilePath = outputDirectoryPath;
    outputFilePath.append("beamSpreadValues.dat");

    std::ofstream outputFile(outputFilePath.string());
    if (!outputFile.is_open()) {
        std::cout << "Failed to create output file: " << outputFilePath.string() << std::endl;
        throw std::runtime_error("Failed to create output file: " + outputFilePath.string());
    }

    std::cout << "Experiment done" << std::endl;

    return 0;
}
