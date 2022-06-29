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

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateNormalizedFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights)
{
    if (rawFrameWeights.empty()) {
        return rawFrameWeights;
    }
    std::vector<boost::multiprecision::cpp_dec_float_100> result = rawFrameWeights;
    boost::multiprecision::cpp_dec_float_100 minimum
          = (*std::min_element(rawFrameWeights.begin(), rawFrameWeights.end()));

    boost::multiprecision::cpp_dec_float_100 maximum
          = (*std::max_element(rawFrameWeights.begin(), rawFrameWeights.end()));

    const boost::multiprecision::cpp_dec_float_100 range = maximum - minimum;

    std::for_each(result.begin(),
          result.end(),
          [minimum, range](
                boost::multiprecision::cpp_dec_float_100 &n) { n = ((n - minimum) / range); });
    return result;
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
    std::uniform_real_distribution<float> uniformFloat(0.0f, 1.0f);
    std::vector<boost::multiprecision::cpp_dec_float_100> framePixels(
          framePixelCount * framePixelCount);

    const float pixelLength = frameLength / framePixelCount;
    Farlor::Vector3 center(distanceFromPlane, 0.0f, 0.0f);

    // Bootstrap method
    const Farlor::Vector3 emitterStart { 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

    // First, we calculate the minimum possible arclength
    float minMinFrameArclength = distanceFromPlane * 2.0f;
    float maxMinFrameArclength = 0.0f;
    int32_t halfFrameWidth = framePixelCount / 2;
    for (int32_t pixelIdxZ = -halfFrameWidth; pixelIdxZ <= halfFrameWidth; ++pixelIdxZ) {
        for (int32_t pixelIdxY = -halfFrameWidth; pixelIdxY <= halfFrameWidth; ++pixelIdxY) {
            const Farlor::Vector3 recieverPos = center
                  + Farlor::Vector3(0.0f, pixelIdxY * pixelLength, pixelIdxZ * pixelLength);

            const Farlor::Vector3 recieverDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
            float minArclength = 0.0f;
            if (experimentParams.numSegmentsPerCurve == 4) {
                const float l = distanceFromPlane;
                const float x
                      = std::sqrt(abs(pixelIdxZ) * pixelLength * abs(pixelIdxZ) * pixelLength
                            + abs(pixelIdxY) * pixelLength * abs(pixelIdxY) * pixelLength);
                float minimumDs = (x * x) / (4.0 * l) + (l / 4.0);
                minArclength = minimumDs * 4.0 * 1.001f;
            } else {
                minArclength = (recieverPos - emitterStart).Magnitude() + 1.1;
            }
            if (minArclength < minMinFrameArclength) {
                minMinFrameArclength = minArclength;
            }
            if (minArclength > maxMinFrameArclength) {
                maxMinFrameArclength = minArclength;
            }
        }
    }

    minMinFrameArclength = 10.01;
    maxMinFrameArclength = 10.83 - (10.83 - 10.01) * 0.5;

    std::cout << "Min Pixel Minimuim Arclength For Frame: " << minMinFrameArclength << std::endl;
    std::cout << "Max Pixel Minimuim Arclength For Frame: " << maxMinFrameArclength << std::endl;

    const float minArclength = minMinFrameArclength;
    const float maxArclength = maxMinFrameArclength + (maxMinFrameArclength - minArclength);

    const float deltaArclength = (maxArclength - minArclength) / numArclengths;

    for (size_t arclengthIdx = 0; arclengthIdx < numArclengths; arclengthIdx++) {
        const float currentArclength = minArclength + deltaArclength * arclengthIdx;

        std::string currentArclengthString = std::to_string(currentArclength);
        std::replace(currentArclengthString.begin(), currentArclengthString.end(), '.', '_');


        for (uint32_t r = 0; r < framePixelCount; r++) {
            for (uint32_t c = 0; c < framePixelCount; c++) {
                const uint32_t frameIdx = r * framePixelCount + c;
                framePixels[frameIdx] = 0.0;
            }
        }

        int32_t halfFrameWidth = framePixelCount / 2;
        for (int32_t pixelIdxZ = -halfFrameWidth; pixelIdxZ <= halfFrameWidth; ++pixelIdxZ) {
            std::cout << "Pixel Idx X: " << pixelIdxZ << std::endl;
            for (int32_t pixelIdxY = -halfFrameWidth; pixelIdxY <= halfFrameWidth; ++pixelIdxY) {
                std::cout << "Pixel Idx Y: " << pixelIdxY << std::endl;

                const Farlor::Vector3 recieverPos = center
                      + Farlor::Vector3(0.0f, pixelIdxY * pixelLength, pixelIdxZ * pixelLength);

                const Farlor::Vector3 recieverDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

                float testMinArclength = 0.0f;
                if (experimentParams.numSegmentsPerCurve == 4) {
                    const float l = distanceFromPlane;
                    const float x
                          = std::sqrt(abs(pixelIdxZ) * pixelLength * abs(pixelIdxZ) * pixelLength
                                + abs(pixelIdxY) * pixelLength * abs(pixelIdxY) * pixelLength);
                    float minimumDs = (x * x) / (4.0 * l) + (l / 4.0);
                    testMinArclength = minimumDs * 4.0 * 1.001f;
                } else {
                    testMinArclength = (recieverPos - emitterStart).Magnitude() + 1.1;
                }

                if (testMinArclength > currentArclength) {
                    // The current pixel has no value.
                    continue;
                }

                experimentParams.arclength = currentArclength;

                boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;

                uint32_t localBootstrapSeed = experimentParams.bootstrapSeed;
                if (localBootstrapSeed == 0) {
                    localBootstrapSeed = time(0);
                }
                std::mt19937 initialCurveGen(localBootstrapSeed);
                for (uint32_t initialCurveIdx = 0; initialCurveIdx < numInitialCurves;
                      ++initialCurveIdx) {
                    int initialCurveSeed = initialCurveGen();
                    while (initialCurveSeed == 0) {
                        initialCurveSeed = initialCurveGen();
                    }

                    boost::multiprecision::cpp_dec_float_100 maxResult = 0.0;
                    uint32_t localCurvePerturbSeed = experimentParams.curvePurturbSeed;
                    if (localCurvePerturbSeed == 0) {
                        localCurvePerturbSeed = time(0);
                    }

                    std::mt19937 perCurveGen(localCurvePerturbSeed);
                    for (uint32_t perInitialCurveIdx = 0; perInitialCurveIdx < numPerInitialCurve;
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
                        experimentGeometry.arclength = currentArclength;

                        twisty::Bootstrapper bootstrapper(experimentGeometry);

                        std::stringstream perExperimentSS;
                        perExperimentSS << currentArclengthString << "/"
                                        << "x_" << pixelIdxZ << "_y_" << pixelIdxY << "_a_"
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
                        upExperimentRunner
                              = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(
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
                        if (results.experimentWeights[0] > maxResult) {
                            maxResult = results.experimentWeights[0];
                        }
                    }

                    averagedResult += (maxResult * (1.0 / numInitialCurves));


                    const uint32_t frameIdx = (pixelIdxY + halfFrameWidth) * framePixelCount
                          + (pixelIdxZ + halfFrameWidth);
                    framePixels[frameIdx] += averagedResult;
                }


                const uint32_t frameIdx = (pixelIdxY + halfFrameWidth) * framePixelCount
                      + (pixelIdxZ + halfFrameWidth);
                std::cout << "Pixel Weight: " << framePixels[frameIdx] << std::endl;
            }
        }

        std::filesystem::path outputDirectoryPath = experimentParams.experimentDirPath;
        if (!std::filesystem::exists(outputDirectoryPath)) {
            std::filesystem::create_directory(outputDirectoryPath);
        }

        // Export raw pixel data
        {
            std::filesystem::path rawDataFilepath = outputDirectoryPath;
            rawDataFilepath /= currentArclengthString;

            if (!std::filesystem::exists(rawDataFilepath)) {
                std::filesystem::create_directories(rawDataFilepath);
            }

            rawDataFilepath /= "noisyCircle.dat";
            std::ofstream rawDataOutfile(rawDataFilepath.string());
            if (!rawDataOutfile.is_open()) {
                std::cout << "Failed to create rawDataOutfile: " << rawDataFilepath.string()
                          << std::endl;
                exit(1);
            }

            // X
            rawDataOutfile << framePixelCount << " " << framePixelCount << std::endl;

            // Write out the pixel data
            for (uint32_t pixelIdxZ = 0; pixelIdxZ < framePixelCount; ++pixelIdxZ) {
                for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
                    // Output pixel
                    const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxZ;
                    rawDataOutfile << framePixels[frameIdx] << " ";
                }
                rawDataOutfile << std::endl;
            }
        }

        const std::vector<boost::multiprecision::cpp_dec_float_100> normalizedFrames
              = CalculateNormalizedFrames(framePixels);

        // Normalized pixel data
        {
            std::filesystem::path rawDataFilepath = outputDirectoryPath;
            rawDataFilepath /= currentArclengthString;

            if (!std::filesystem::exists(rawDataFilepath)) {
                std::filesystem::create_directories(rawDataFilepath);
            }

            rawDataFilepath /= "normalizedData.dat";
            std::ofstream rawDataOutfile(rawDataFilepath.string());
            if (!rawDataOutfile.is_open()) {
                std::cout << "Failed to create normalizedData: " << rawDataFilepath.string()
                          << std::endl;
                exit(1);
            }

            // X
            rawDataOutfile << framePixelCount << " " << framePixelCount << std::endl;

            // Write out the pixel data
            for (uint32_t pixelIdxZ = 0; pixelIdxZ < framePixelCount; ++pixelIdxZ) {
                for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
                    // Output pixel
                    const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxZ;
                    rawDataOutfile << normalizedFrames[frameIdx] << " ";
                }
                rawDataOutfile << std::endl;
            }
        }
    }
    std::cout << "Experiment done" << std::endl;

    return 0;
}
