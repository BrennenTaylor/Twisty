#include "FullExperimentRunnerOptimalPerturb.h"

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
    experimentParams.experimentDirPath += "/" + twisty::GetCurrentTimeForFileName() + "/";

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

    std::vector<float> scatterValues;
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


    const int normalGenSeed
          = (int)experimentConfig["experiment"]["noisyCircleExperiment"]["normalGenSeed"]
          ? (int)experimentConfig["experiment"]["noisyCircleExperiment"]["normalGenSeed"]
          : (int)time(0);

    const int startX = experimentConfig["experiment"]["noisyCircleExperiment"]["startX"];
    const int startY = experimentConfig["experiment"]["noisyCircleExperiment"]["startY"];

    const double frameLength
          = experimentConfig["experiment"]["noisyCircleExperiment"]["frameLength"];
    const uint32_t framePixelCount
          = experimentConfig["experiment"]["noisyCircleExperiment"]["framePixelCount"];

    const uint32_t numInitialCurves
          = experimentConfig["experiment"]["noisyCircleExperiment"]["numInitialCurves"];
    const uint32_t numPerInitialCurve
          = experimentConfig["experiment"]["noisyCircleExperiment"]["numPerInitialCurve"];

    // Ok, we want to kick off an experiment per pixel.
    const uint32_t numDirections
          = experimentConfig["experiment"]["noisyCircleExperiment"]["numDirections"];
    const uint32_t numArclengths
          = experimentConfig["experiment"]["noisyCircleExperiment"]["numArclengths"];
    const float distanceFromPlane
          = experimentConfig["experiment"]["noisyCircleExperiment"]["distanceFromPlane"];

    assert(startX < framePixelCount);
    assert(startY < framePixelCount);

    // Lets test uniform random directions on sampling hemisphere
    std::mt19937 normalGenerator(normalGenSeed);
    std::uniform_real_distribution<float> uniformFloat(0.0f, 1.0f);
    std::vector<boost::multiprecision::cpp_dec_float_100> framePixels(
          framePixelCount * framePixelCount);
    for (uint32_t r = 0; r < framePixelCount; r++) {
        for (uint32_t c = 0; c < framePixelCount; c++) {
            const uint32_t frameIdx = r * framePixelCount + c;
            framePixels[frameIdx] = 0.0;
        }
    }

    Farlor::Vector3 center(distanceFromPlane, 0.0f, 0.0f);
    Farlor::Vector3 bottomLeft
          = center - Farlor::Vector3(0.0f, frameLength / 2.0f, frameLength / 2.0f);

    // Bootstrap method
    const Farlor::Vector3 emitterStart { 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

    bool IsFirst = true;

    for (int32_t pixelIdxX = 0; pixelIdxX < ceil(framePixelCount / 2.0); ++pixelIdxX) {
        if (!(pixelIdxX == 4)) {
            continue;
        }
        std::cout << "Pixel Idx X: " << pixelIdxX << std::endl;

        for (int32_t pixelIdxY = 0; pixelIdxY < ceil(framePixelCount / 2.0); ++pixelIdxY) {
            if (!(pixelIdxY == 4)) {
                continue;
            }
            std::cout << "Pixel Idx Y: " << pixelIdxY << std::endl;


            if (IsFirst) {
                pixelIdxX = startX;
                pixelIdxY = startY;
                IsFirst = false;
            }

            const Farlor::Vector3 recieverPos = bottomLeft
                  + Farlor::Vector3(0.0f, pixelIdxY * (frameLength / framePixelCount),
                        pixelIdxX * (frameLength / framePixelCount))
                  + Farlor::Vector3(0.0f, (frameLength / framePixelCount) / 2.0f,
                        (frameLength / framePixelCount) / 2.0f);

            {
                const Farlor::Vector3 recieverDir = (recieverPos - emitterStart).Normalized();


                const float minArclength
                      = twisty::Bootstrapper::CalculateMinimumArclength(
                              experimentParams.numSegmentsPerCurve, emitterStart, recieverPos)
                      * 1.1f;
                const float maxArclength = minArclength * 1.5f;
                const float arclengthStepSize = (maxArclength - minArclength) / (numArclengths);

                for (uint32_t arclengthIdx = 0; arclengthIdx < numArclengths; ++arclengthIdx) {
                    double targetArclength = minArclength + arclengthStepSize * arclengthIdx;
                    experimentParams.arclength = targetArclength;

                    boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;

                    std::mt19937 initialCurveGen(experimentParams.bootstrapSeed);
                    for (uint32_t initialCurveIdx = 0; initialCurveIdx < numInitialCurves;
                          ++initialCurveIdx) {
                        int initialCurveSeed = initialCurveGen();
                        while (initialCurveSeed == 0) {
                            initialCurveSeed = initialCurveGen();
                        }

                        boost::multiprecision::cpp_dec_float_100 maxResult = 0.0;

                        std::mt19937 perCurveGen(experimentParams.curvePurturbSeed);
                        for (uint32_t perInitialCurveIdx = 0;
                              perInitialCurveIdx < numPerInitialCurve;
                              ++perInitialCurveIdx) {
                            int perCurveSeed = perCurveGen();
                            while (perCurveSeed == 0) {
                                perCurveSeed = perCurveGen();
                            }

                            twisty::PerturbUtils::BoundaryConditions experimentGeometry;
                            experimentGeometry.m_startPos = emitterStart;
                            experimentGeometry.m_startDir = emitterDir;
                            experimentGeometry.m_endPos = recieverPos;
                            experimentGeometry.m_endDir = recieverDir;
                            experimentGeometry.arclength = targetArclength;

                            twisty::Bootstrapper bootstrapper(experimentGeometry);

                            std::stringstream perExperimentSS;
                            perExperimentSS << "x_" << pixelIdxX << "_y_" << pixelIdxY << "_a_"
                                            << arclengthIdx << "_ic_" << initialCurveIdx << "_ci_"
                                            << perInitialCurveIdx;
                            experimentParams.perExperimentDirSubfolder = perExperimentSS.str();


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
                                upExperimentRunner = std::make_unique<
                                      twisty::FullExperimentRunnerOptimalPerturb>(
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
                            upExperimentRunner
                                  = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(
                                        experimentParams, bootstrapper);
#endif

                            std::optional<twisty::ExperimentRunner::ExperimentResults>
                                  optionalResults = upExperimentRunner->RunExperiment();
                            if (!optionalResults.has_value()) {
                                std::cout << "Experiment failed: no results returned." << std::endl;
                                return 1;
                            }
                            const twisty::ExperimentRunner::ExperimentResults &results
                                  = optionalResults.value();
                            if (results.experimentWeights[0] > maxResult) {
                                maxResult = results.experimentWeights[0];
                            }
                        }

                        averagedResult += (maxResult * (1.0 / numInitialCurves));
                    }

                    const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
                    framePixels[frameIdx] += averagedResult * (1.0 / (numArclengths));
                }
            }

            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
            std::cout << "Non-Rotated Pixel Weight: " << framePixels[frameIdx] << std::endl;
        }
    }

    std::filesystem::path outputDirectoryPath = experimentParams.experimentDirPath;
    if (!std::filesystem::exists(outputDirectoryPath)) {
        std::filesystem::create_directory(outputDirectoryPath);
    }

    // Export noisy circle pixel data
    {
        const std::filesystem::path rawDataFilepath
              = outputDirectoryPath / "noisyCircleNonRotated.dat";
        std::ofstream rawDataOutfile(rawDataFilepath.string());
        if (!rawDataOutfile.is_open()) {
            std::cout << "Failed to create rawDataOutfile: " << rawDataFilepath.string()
                      << std::endl;
            exit(1);
        }

        // X
        rawDataOutfile << framePixelCount << " " << framePixelCount << std::endl;

        // Write out the pixel data
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX) {
            for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
                // Output pixel
                const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
                rawDataOutfile << framePixels[frameIdx] << " ";
            }
            rawDataOutfile << std::endl;
        }
    }

    std::cout << "Experiment done" << std::endl;

    return 0;
}
