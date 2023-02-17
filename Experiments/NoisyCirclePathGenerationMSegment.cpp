#include "ExperimentBase.h"

#include "CombinedWeightUtils.h"
#include "Curve.h"
#include "CurvePerturbUtils.h"
#include "ExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "boost/multiprecision/cpp_dec_float.hpp"


#include "Bootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "PathWeighters.h"

#include <FMath/Vector3.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateNormalizedFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights);

void OutputRawData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawCombined,
      const int32_t framePixelCount);

void OutputNormalizedData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &normalizedCombined,
      const int32_t framePixelCount);

struct NoisyCircleAngleIntegrationParams {
    int startX = 0;
    int startY = 0;

    float frameLength = 1.0;
    uint32_t framePixelCount = 1;

    // Ok, we want to kick off an experiment per pixel.
    uint32_t numDirections = 1;
    float arclengthStepSize = 1.0f;
    float distanceFromPlane = 1.0f;

    uint32_t numPhi1Vals = 1;
    uint32_t numTheta1Vals = 1;
    uint32_t numTheta2Vals = 1;
};

NoisyCircleAngleIntegrationParams ParseExperimentSpecificParams(nlohmann::json &experimentConfig)
{
    NoisyCircleAngleIntegrationParams params;
    params.startX = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["startX"];
    params.startY = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["startY"];

    params.frameLength
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["frameLength"];
    params.framePixelCount
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["framePixelCount"];

    // Ok, we want to kick off an experiment per pixel.
    params.numDirections
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["numDirections"];
    params.arclengthStepSize
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["arclengthStepSize"];
    params.distanceFromPlane
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["distanceFromPlane"];

    assert(params.startX < params.framePixelCount);
    assert(params.startY < params.framePixelCount);
    return params;
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
          = twisty::ExperimentRunner::ParseExperimentParamsFromConfig(experimentConfig);

    NoisyCircleAngleIntegrationParams experimentSpecificParams
          = ParseExperimentSpecificParams(experimentConfig);

    std::uniform_real_distribution<float> uniformFloat(0.0f, 1.0f);
    std::vector<boost::multiprecision::cpp_dec_float_100> framePixels(
          experimentSpecificParams.framePixelCount * experimentSpecificParams.framePixelCount);

    std::vector<boost::multiprecision::cpp_dec_float_100> combinedPixels(
          experimentSpecificParams.framePixelCount * experimentSpecificParams.framePixelCount);

    const float pixelLength = experimentSpecificParams.frameLength
          / static_cast<float>(experimentSpecificParams.framePixelCount);
    Farlor::Vector3 centerOfFrame(experimentSpecificParams.distanceFromPlane, 0.0f, 0.0f);

    // Bootstrap method
    const Farlor::Vector3 emitterStart { 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

    // First, we calculate the minimum possible arclength
    float minMinFrameArclength = experimentSpecificParams.distanceFromPlane * 2.0f;
    float maxMinFrameArclength = 0.0f;
    int32_t halfFrameWidth = experimentSpecificParams.framePixelCount / 2;
    for (int32_t pixelIdxZ = -halfFrameWidth; pixelIdxZ <= halfFrameWidth; ++pixelIdxZ) {
        for (int32_t pixelIdxY = -halfFrameWidth; pixelIdxY <= halfFrameWidth; ++pixelIdxY) {
            const Farlor::Vector3 recieverPos = centerOfFrame
                  + Farlor::Vector3(0.0f, pixelIdxY * pixelLength, pixelIdxZ * pixelLength);
            float minArclength = 0.0f;
            switch (experimentParams.numSegmentsPerCurve) {
                case 5: {
                    const float l = experimentSpecificParams.distanceFromPlane;
                    const float x
                          = std::sqrt(abs(pixelIdxZ) * pixelLength * abs(pixelIdxZ) * pixelLength
                                + abs(pixelIdxY) * pixelLength * abs(pixelIdxY) * pixelLength);
                    float minimumDs = (x * x) / ((double)experimentParams.numSegmentsPerCurve * l)
                          + (l / (double)experimentParams.numSegmentsPerCurve);
                    minArclength
                          = minimumDs * (double)experimentParams.numSegmentsPerCurve * 1.001f;
                } break;
                default: {
                    // Give an extra nubmer of segments
                    const uint32_t numExtraSegments = 1;
                    minArclength = (recieverPos - emitterStart).Magnitude()
                          * (static_cast<float>(
                                   experimentParams.numSegmentsPerCurve + numExtraSegments)
                                / static_cast<float>(experimentParams.numSegmentsPerCurve));
                } break;
            }
            minMinFrameArclength = std::min(minMinFrameArclength, minArclength);
            maxMinFrameArclength = std::max(maxMinFrameArclength, minArclength);
        }
    }

    std::cout << "Min Pixel Minimuim Arclength For Frame: " << minMinFrameArclength << std::endl;
    std::cout << "Max Pixel Minimuim Arclength For Frame: " << maxMinFrameArclength << std::endl;

    // TODO: Should we target the number of arclengths per pixel?
    // Currently, the minimum distance pixel gets twice the arclength exploration as the furthermost pixels
    const float minArclength = minMinFrameArclength;
    const float maxArclength = maxMinFrameArclength + (maxMinFrameArclength - minArclength);
    std::cout << "Min arclength: " << minArclength << std::endl;
    std::cout << "Max arclength: " << maxArclength << std::endl;


    uint32_t numArclengths
          = ceil((maxArclength - minArclength) / experimentSpecificParams.arclengthStepSize);
    if (numArclengths == 0) {
        numArclengths = 1;
        experimentSpecificParams.arclengthStepSize = (maxArclength - minArclength);
    }
    std::cout << "Num arclengths: " << numArclengths << std::endl;

    for (size_t arclengthIdx = 0; arclengthIdx < numArclengths; arclengthIdx++) {
        const float currentArclength
              = minArclength + experimentSpecificParams.arclengthStepSize * arclengthIdx;

        const float ds = currentArclength / experimentParams.numSegmentsPerCurve;

        std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;
        if (experimentParams.weightingParameters.weightingMethod
              == twisty::WeightingMethod::SimplifiedModel) {
            lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(
                  experimentParams.weightingParameters, ds);
        } else {
            lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
                  experimentParams.weightingParameters, ds);
        }
        lookupEvaluator->ExportValues(experimentParams.experimentDirPath);
        assert(lookupEvaluator);
        twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable = (*lookupEvaluator);

        std::string currentArclengthString = std::to_string(currentArclength);
        std::replace(currentArclengthString.begin(), currentArclengthString.end(), '.', '_');

        for (uint32_t r = 0; r < experimentSpecificParams.framePixelCount; r++) {
            for (uint32_t c = 0; c < experimentSpecificParams.framePixelCount; c++) {
                const uint32_t frameIdx = r * experimentSpecificParams.framePixelCount + c;
                framePixels[frameIdx] = 0.0;
            }
        }

        for (int32_t pixelIdxZ = -halfFrameWidth; pixelIdxZ <= halfFrameWidth; ++pixelIdxZ) {
            std::cout << "Pixel Idx X: " << pixelIdxZ << std::endl;
            for (int32_t pixelIdxY = -halfFrameWidth; pixelIdxY <= halfFrameWidth; ++pixelIdxY) {
                std::cout << "Pixel Idx Y: " << pixelIdxY << std::endl;

                const Farlor::Vector3 recieverPos = centerOfFrame
                      + Farlor::Vector3(0.0f, pixelIdxY * pixelLength, pixelIdxZ * pixelLength);
                const Farlor::Vector3 recieverDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

                float testMinArclength = 0.0f;
                if (experimentParams.numSegmentsPerCurve == 5) {
                    const float l = experimentSpecificParams.distanceFromPlane;
                    const float x
                          = std::sqrt(abs(pixelIdxZ) * pixelLength * abs(pixelIdxZ) * pixelLength
                                + abs(pixelIdxY) * pixelLength * abs(pixelIdxY) * pixelLength);
                    float minimumDs = (x * x) / ((double)experimentParams.numSegmentsPerCurve * l)
                          + (l / (double)experimentParams.numSegmentsPerCurve);
                    testMinArclength
                          = minimumDs * (double)experimentParams.numSegmentsPerCurve * 1.001f;
                } else {
                    testMinArclength = (recieverPos - emitterStart).Magnitude() + 1.1;
                }

                if (testMinArclength > currentArclength) {
                    // The current pixel has no value.
                    continue;
                }

                twisty::PerturbUtils::BoundaryConditions experimentGeometry;
                experimentGeometry.m_startPos = emitterStart;
                experimentGeometry.m_startDir = emitterDir;
                experimentGeometry.m_endPos = recieverPos;
                experimentGeometry.m_endDir = recieverDir;
                experimentGeometry.arclength = experimentParams.arclength = currentArclength;

                const twisty::PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer
                      = (experimentParams.weightingParameters.weightingMethod
                              != twisty::WeightingMethod::RadiativeTransfer)
                      ? 1.0
                      : twisty::PathWeighting::NormalizerStuff::Norm(
                            experimentParams.numSegmentsPerCurve, ds, experimentGeometry);

                const double pathNormalizerLog10 = (pathNormalizer != 0.0)
                      ? (double)boost::multiprecision::log10(pathNormalizer)
                      : 0.0;

                const uint32_t frameIdx
                      = (pixelIdxY + halfFrameWidth) * experimentSpecificParams.framePixelCount
                      + (pixelIdxZ + halfFrameWidth);
                const twisty::ExperimentBase::Result result
                      = twisty::ExperimentBase::MSegmentPathGenerationMC(
                            experimentParams.numPathsInExperiment,
                            experimentParams.numSegmentsPerCurve, experimentGeometry,
                            experimentParams, pathNormalizerLog10, weightLookupTable);

                std::cout << "Num valid paths: " << result.numValidPaths << "/"
                          << result.numPathsTotal << std::endl;
                std::cout << "Percent valid paths: "
                          << (result.numValidPaths / (float)result.numPathsTotal) * 100.0f << "%"
                          << std::endl;
                std::cout << "Total weight: " << result.totalWeight << std::endl;

                if (result.numValidPaths > 0) {
                    framePixels[frameIdx] = result.totalWeight;
                }


                std::cout << "Pixel Weight: " << framePixels[frameIdx] << std::endl;
            }
        }

        std::filesystem::path outputDirectoryPath = experimentParams.experimentDirPath;
        if (!std::filesystem::exists(outputDirectoryPath)) {
            std::filesystem::create_directories(outputDirectoryPath);
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
            rawDataOutfile << experimentSpecificParams.framePixelCount << " "
                           << experimentSpecificParams.framePixelCount << std::endl;

            // Write out the pixel data
            for (uint32_t pixelIdxZ = 0; pixelIdxZ < experimentSpecificParams.framePixelCount;
                  ++pixelIdxZ) {
                for (uint32_t pixelIdxY = 0; pixelIdxY < experimentSpecificParams.framePixelCount;
                      ++pixelIdxY) {
                    // Output pixel
                    const uint32_t frameIdx
                          = pixelIdxY * experimentSpecificParams.framePixelCount + pixelIdxZ;
                    rawDataOutfile << framePixels[frameIdx] << " ";
                    combinedPixels[frameIdx] += framePixels[frameIdx] * (1.0f / numArclengths);
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
            rawDataOutfile << experimentSpecificParams.framePixelCount << " "
                           << experimentSpecificParams.framePixelCount << std::endl;

            // Write out the pixel data
            for (uint32_t pixelIdxZ = 0; pixelIdxZ < experimentSpecificParams.framePixelCount;
                  ++pixelIdxZ) {
                for (uint32_t pixelIdxY = 0; pixelIdxY < experimentSpecificParams.framePixelCount;
                      ++pixelIdxY) {
                    // Output pixel
                    const uint32_t frameIdx
                          = pixelIdxY * experimentSpecificParams.framePixelCount + pixelIdxZ;
                    rawDataOutfile << normalizedFrames[frameIdx] << " ";
                }
                rawDataOutfile << std::endl;
            }
        }
    }

    const std::vector<boost::multiprecision::cpp_dec_float_100> normalizedCombined
          = CalculateNormalizedFrames(framePixels);

    std::filesystem::path outputDirectoryPath = experimentParams.experimentDirPath;
    if (!std::filesystem::exists(outputDirectoryPath)) {
        std::filesystem::create_directories(outputDirectoryPath);
    }

    OutputRawData(outputDirectoryPath, combinedPixels, experimentSpecificParams.framePixelCount);
    OutputNormalizedData(
          outputDirectoryPath, normalizedCombined, experimentSpecificParams.framePixelCount);

    std::cout << "Experiment done" << std::endl;

    return 0;
}

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateNormalizedFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights)
{
    if (rawFrameWeights.empty()) {
        return rawFrameWeights;
    }
    std::vector<boost::multiprecision::cpp_dec_float_100> result = rawFrameWeights;
    boost::multiprecision::cpp_dec_float_100 minimum
          = *std::min_element(rawFrameWeights.begin(), rawFrameWeights.end());

    boost::multiprecision::cpp_dec_float_100 maximum
          = *std::max_element(rawFrameWeights.begin(), rawFrameWeights.end());

    const boost::multiprecision::cpp_dec_float_100 range = maximum - minimum;

    std::for_each(result.begin(),
          result.end(),
          [&minimum, &range](
                boost::multiprecision::cpp_dec_float_100 &n) { n = ((n - minimum) / range); });
    return result;
}

void OutputRawData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawCombined,
      const int32_t framePixelCount)
{
    std::filesystem::path rawDataFilepath = outputDirectoryPath;
    rawDataFilepath /= "Combined/";
    if (!std::filesystem::exists(rawDataFilepath)) {
        std::filesystem::create_directories(rawDataFilepath);
    }

    rawDataFilepath /= "noisyCircle.dat";
    std::ofstream rawDataOutfile(rawDataFilepath.string());

    std::cout << "Combined raw Data Outfile path: " << rawDataFilepath << std::endl;

    if (!rawDataOutfile.is_open()) {
        std::cout << "Failed to create rawDataOutfile: " << rawDataFilepath.string() << std::endl;
        exit(1);
    }

    // X
    rawDataOutfile << framePixelCount << " " << framePixelCount << std::endl;

    // Write out the pixel data
    for (uint32_t pixelIdxZ = 0; pixelIdxZ < framePixelCount; ++pixelIdxZ) {
        for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxZ;
            rawDataOutfile << rawCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}

void OutputNormalizedData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &normalizedCombined,
      const int32_t framePixelCount)
{
    std::filesystem::path rawDataFilepath = outputDirectoryPath;
    rawDataFilepath /= "Combined/";

    if (!std::filesystem::exists(rawDataFilepath)) {
        std::filesystem::create_directories(rawDataFilepath);
    }

    rawDataFilepath /= "normalizedData.dat";
    std::ofstream rawDataOutfile(rawDataFilepath.string());
    if (!rawDataOutfile.is_open()) {
        std::cout << "Failed to create normalizedData: " << rawDataFilepath.string() << std::endl;
        exit(1);
    }

    // X
    rawDataOutfile << framePixelCount << " " << framePixelCount << std::endl;

    // Write out the pixel data
    for (uint32_t pixelIdxZ = 0; pixelIdxZ < framePixelCount; ++pixelIdxZ) {
        for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxZ;
            rawDataOutfile << normalizedCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}