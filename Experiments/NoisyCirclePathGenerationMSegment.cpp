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

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateOffsetFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights);

void OutputRawData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawCombined,
      const int32_t framePixelCount);

void OutputNormalizedData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &normalizedCombined,
      const int32_t framePixelCount);

void OutputOffsetData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &offsetCombined,
      const int32_t framePixelCount);

struct NoisyCircleParams {
    int startX = 0;
    int startY = 0;

    float frameLength = 1.0;
    uint32_t framePixelCount = 1;

    // Ok, we want to kick off an experiment per pixel.
    uint32_t numDirections = 1;
    float maxArclengthOffset = 1.0f;
    float distanceFromPlane = 1.0f;
};

NoisyCircleParams ParseExperimentSpecificParams(nlohmann::json &experimentConfig)
{
    NoisyCircleParams params;
    params.startX = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["startX"];
    params.startY = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["startY"];

    params.frameLength
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["frameLength"];
    params.framePixelCount
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["framePixelCount"];

    // Ok, we want to kick off an experiment per pixel.
    params.numDirections
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["numDirections"];
    params.maxArclengthOffset
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["maxArclengthOffset"];
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

    NoisyCircleParams experimentSpecificParams = ParseExperimentSpecificParams(experimentConfig);

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

    int32_t halfFrameWidth = experimentSpecificParams.framePixelCount / 2;

    // Experiment start time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Vector storing each runs time
    std::vector<uint64_t> runTimes;

    for (uint32_t r = 0; r < experimentSpecificParams.framePixelCount; r++) {
        for (uint32_t c = 0; c < experimentSpecificParams.framePixelCount; c++) {
            const uint32_t frameIdx = r * experimentSpecificParams.framePixelCount + c;
            framePixels[frameIdx] = 0.0;
        }
    }

    const auto uuid = experimentParams.weightingParameters.GenerateStringUUID();
    std::cout << "Weighting parameters hash: " << uuid.first << " \n"
              << uuid.second << '\n'
              << std::endl;

    // We are going to bake a big ol table, then use this whenever we need.
    const float minArclength = 10.0f;
    const float maxArclength = 20.0f;
    const float minDs = minArclength / experimentParams.numSegmentsPerCurve;
    const float maxDs = maxArclength / experimentParams.numSegmentsPerCurve;
    const uint32_t numArclengths = 1000;

    twisty::PathWeighting::CachedMultiArclengthWeightLookupTable objectCachedLookupTable(
          experimentParams.weightingParameters, minDs, maxDs, numArclengths);

    twisty::WeightingParameters environmentWeightingParams = experimentParams.weightingParameters;
    environmentWeightingParams.absorption = 0.001f;
    environmentWeightingParams.scatter = 0.001f;

    twisty::PathWeighting::CachedMultiArclengthWeightLookupTable environmentCachedLookupTable(
          environmentWeightingParams, minDs, maxDs, numArclengths);

    const Farlor::Vector3 planeNormal = Farlor::Vector3(-1.0f, 0.0f, 0.0f);
    const Farlor::Vector3 planeNormalO1 = Farlor::Vector3(0.0f, 1.0f, 0.0f);
    const Farlor::Vector3 planeNormalO2 = Farlor::Vector3(0.0f, 0.0f, 1.0f);

    const int32_t numRecieverDirections = experimentSpecificParams.numDirections;

    std::mt19937_64 rng(0);

    for (int32_t pixelIdxZ = -halfFrameWidth; pixelIdxZ <= halfFrameWidth; ++pixelIdxZ) {
        std::cout << "Pixel Idx X: " << pixelIdxZ << std::endl;
        for (int32_t pixelIdxY = -halfFrameWidth; pixelIdxY <= halfFrameWidth; ++pixelIdxY) {
            std::cout << "Pixel Idx Y: " << pixelIdxY << std::endl;

            const uint32_t frameIdx
                  = (pixelIdxY + halfFrameWidth) * experimentSpecificParams.framePixelCount
                  + (pixelIdxZ + halfFrameWidth);

            // Sample over a number of reciever directions
            for (int32_t recieverDirIdx = 0; recieverDirIdx < numRecieverDirections;
                  ++recieverDirIdx) {
                // https://alexanderameye.github.io/notes/sampling-the-hemisphere/
                // Reciever direction
                const float e0 = uniformFloat(rng);
                const float e1 = uniformFloat(rng);

                const float theta = std::acos(std::sqrt(e0));
                const float phi = 2.0f * twisty::TwistyPi * e1;

                // Flip the plane normal so we are facing the correct way
                const Farlor::Vector3 recieverDir = (-1.0f * planeNormal * std::cos(theta))
                      + planeNormalO1 * std::sin(theta) * std::cos(phi)
                      + planeNormalO2 * std::sin(theta) * std::sin(phi);

                const float pdf = std::cos(theta) / twisty::TwistyPi;


                const Farlor::Vector3 recieverPos = centerOfFrame
                      + planeNormalO1 * (pixelIdxY * pixelLength)
                      + planeNormalO2 * (pixelIdxZ * pixelLength);

                // Emitter direction
                const Farlor::Vector3 emitterDir = centerOfFrame.Normalized();

                twisty::PerturbUtils::BoundaryConditions experimentGeometry;
                experimentGeometry.m_startPos = emitterStart;
                experimentGeometry.m_startDir = emitterDir;
                experimentGeometry.m_endPos = recieverPos;
                experimentGeometry.m_endDir = recieverDir;
                experimentGeometry.arclength = 0.0f;

                const Farlor::Vector3 revserseDir = experimentGeometry.m_endDir * -1.0f;
                const float cosFactor = revserseDir.Dot(planeNormal);

                const double pathNormalizerLog10 = 0.0f;

                // Single run start time
                const auto startTime = std::chrono::high_resolution_clock::now();

                std::uniform_int_distribution<uint64_t> uniformInt(
                      0, std::numeric_limits<uint64_t>::max() - 250);
                const uint64_t rngSeed = uniformInt(rng);

                const twisty::ExperimentBase::Result result
                      = twisty::ExperimentBase::MSegmentPathGenerationMC(rngSeed,
                            experimentParams.numPathsInExperiment,
                            experimentParams.numSegmentsPerCurve, experimentGeometry,
                            experimentParams, pathNormalizerLog10, environmentCachedLookupTable,
                            objectCachedLookupTable, maxDs);
                const boost::multiprecision::cpp_dec_float_100 importanceSampledWeight
                      = result.totalWeight / pdf;

                // Single run end time
                const auto endTime = std::chrono::high_resolution_clock::now();
                // Add time difference to runTimes vector
                const auto timeDiff
                      = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                runTimes.push_back(timeDiff.count());

                std::cout << "Num valid paths: " << result.numValidPaths << "/"
                          << result.numPathsTotal << std::endl;
                std::cout << "Percent valid paths: "
                          << (result.numValidPaths / (float)result.numPathsTotal) * 100.0f << "%"
                          << std::endl;
                std::cout << "Total weight: " << importanceSampledWeight << std::endl;
                std::cout << "Total weight w/ cos factor: " << importanceSampledWeight * cosFactor
                          << std::endl;

                if (result.numValidPaths > 0) {
                    framePixels[frameIdx] += importanceSampledWeight * cosFactor;
                }
            }

            framePixels[frameIdx] /= numRecieverDirections;
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
                combinedPixels[frameIdx] += framePixels[frameIdx];
            }
            rawDataOutfile << std::endl;
        }
    }

    const std::vector<boost::multiprecision::cpp_dec_float_100> normalizedFrames
          = CalculateNormalizedFrames(framePixels);

    // Normalized pixel data
    {
        std::filesystem::path rawDataFilepath = outputDirectoryPath;

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

    const std::vector<boost::multiprecision::cpp_dec_float_100> normalizedCombined
          = CalculateNormalizedFrames(framePixels);
    const std::vector<boost::multiprecision::cpp_dec_float_100> offsetCombined
          = CalculateOffsetFrames(framePixels);

    // Experiment end time
    const auto endTime = std::chrono::high_resolution_clock::now();

    // Experiment duration
    const auto duration
          = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Experiment duration: " << duration.count() << "ms" << std::endl;
    // Experiment duration seconds
    const auto durationSeconds
          = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    std::cout << "Experiment duration: " << durationSeconds.count() << "s" << std::endl;

    // Experiment duration minutes
    const auto durationMinutes
          = std::chrono::duration_cast<std::chrono::minutes>(endTime - startTime);
    std::cout << "Experiment duration: " << durationMinutes.count() << "m" << std::endl;

    // Average runTimes
    uint64_t totalRunTime = 0;
    for (const auto &runTime : runTimes) {
        totalRunTime += runTime;
    }
    const uint64_t averageRunTime = totalRunTime / runTimes.size();

    std::cout << "Average run time: " << averageRunTime << "ms" << std::endl;
    // Avg run time in seconds
    const double averageRunTimeSeconds = averageRunTime / 1000.0;
    std::cout << "Average run time: " << averageRunTimeSeconds << "s" << std::endl;

    // Average run time in minutes
    const double averageRunTimeMinutes = averageRunTimeSeconds / 60.0;
    std::cout << "Average run time: " << averageRunTimeMinutes << "m" << std::endl;


    OutputRawData(outputDirectoryPath, combinedPixels, experimentSpecificParams.framePixelCount);
    OutputNormalizedData(
          outputDirectoryPath, normalizedCombined, experimentSpecificParams.framePixelCount);
    OutputOffsetData(outputDirectoryPath, offsetCombined, experimentSpecificParams.framePixelCount);

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

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateOffsetFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights)
{
    if (rawFrameWeights.empty()) {
        return rawFrameWeights;
    }
    std::vector<boost::multiprecision::cpp_dec_float_100> result = rawFrameWeights;

    boost::multiprecision::cpp_dec_float_100 maximum
          = *std::max_element(rawFrameWeights.begin(), rawFrameWeights.end());
    boost::multiprecision::cpp_dec_float_100 invMax
          = boost::multiprecision::cpp_dec_float_100(1.0) / maximum;

    std::for_each(result.begin(), result.end(),
          [&invMax](boost::multiprecision::cpp_dec_float_100 &n) { n = n * invMax; });
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

void OutputOffsetData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &offsetCombined,
      const int32_t framePixelCount)
{
    std::filesystem::path rawDataFilepath = outputDirectoryPath;
    rawDataFilepath /= "Combined/";

    if (!std::filesystem::exists(rawDataFilepath)) {
        std::filesystem::create_directories(rawDataFilepath);
    }

    rawDataFilepath /= "offsetData.dat";
    std::ofstream rawDataOutfile(rawDataFilepath.string());
    if (!rawDataOutfile.is_open()) {
        std::cout << "Failed to create offsetData: " << rawDataFilepath.string() << std::endl;
        exit(1);
    }

    // X
    rawDataOutfile << framePixelCount << " " << framePixelCount << std::endl;

    // Write out the pixel data
    for (uint32_t pixelIdxZ = 0; pixelIdxZ < framePixelCount; ++pixelIdxZ) {
        for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxZ;
            rawDataOutfile << offsetCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}
