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
#include "boost/multiprecision/detail/default_ops.hpp"

#include <FMath/Vector3.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateNormalizedFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights);

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateOffsetFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights);

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateQuadAvgFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights,
      const uint32_t totalWidth);

void OutputRawData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawCombined,
      const int32_t framePixelCount);

void OutputNormalizedData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &normalizedCombined,
      const int32_t framePixelCount);

void OutputOffsetData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &offsetCombined,
      const int32_t framePixelCount);

void OutputQuadAverageData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &combined,
      const int32_t framePixelCount);

class Camera {
   public:
    Camera(const glm::vec3 &position, const glm::vec3 &lookAt, const glm::vec3 &up, float fov,
          float aspectRatio, int imageWidth, int imageHeight)
        : position(position)
        , fov(fov)
        , aspectRatio(aspectRatio)
        , imageWidth(imageWidth)
        , imageHeight(imageHeight)
    {
        forward = glm::normalize(lookAt - position);
        right = glm::normalize(glm::cross(forward, up));
        this->up = glm::cross(right, forward);
        updateCameraVectors();
    }

    void setPosition(const glm::vec3 &newPosition)
    {
        position = newPosition;
        updateCameraVectors();
    }

    void setLookAt(const glm::vec3 &lookAt)
    {
        forward = glm::normalize(lookAt - position);
        right = glm::normalize(glm::cross(forward, up));
        up = glm::cross(right, forward);
        updateCameraVectors();
    }

    Farlor::Vector3 getPosition() const
    {
        return Farlor::Vector3(position.x, position.y, position.z);
    }

    Farlor::Vector3 getRayDirectionFromUV(float u, float v) const
    {
        const glm::vec3 norm
              = glm::normalize(lowerLeftCorner + u * horizontal + v * vertical - position);
        return Farlor::Vector3(norm.x, norm.y, norm.z);
    }

    Farlor::Vector3 getRayDirectionForPixel(int x, int y) const
    {
        const float u = (x + 0.5f) / static_cast<float>(imageWidth);
        const float v = (y + 0.5f) / static_cast<float>(imageHeight);
        return getRayDirectionFromUV(u, v);
    }

    Farlor::Vector3 getForward() const { return Farlor::Vector3(forward.x, forward.y, forward.z);
    }

   private:
    glm::vec3 position;
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
    float fov;
    float aspectRatio;
    int imageWidth;
    int imageHeight;

    glm::vec3 lowerLeftCorner;
    glm::vec3 horizontal;
    glm::vec3 vertical;

    void updateCameraVectors()
    {
        float theta = glm::radians(fov);
        float halfHeight = tan(theta / 2);
        float halfWidth = aspectRatio * halfHeight;

        lowerLeftCorner = position - halfWidth * right - halfHeight * up - forward;
        horizontal = 2 * halfWidth * right;
        vertical = 2 * halfHeight * up;
    }
};


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

    try {
        params.startX = experimentConfig.at("experiment")
                              .at("noisyCircleAngleIntegration")
                              .at("startX")
                              .get<int>();
        params.startY = experimentConfig.at("experiment")
                              .at("noisyCircleAngleIntegration")
                              .at("startY")
                              .get<int>();

        params.frameLength = experimentConfig.at("experiment")
                                   .at("noisyCircleAngleIntegration")
                                   .at("frameLength")
                                   .get<float>();
        params.framePixelCount = experimentConfig.at("experiment")
                                       .at("noisyCircleAngleIntegration")
                                       .at("framePixelCount")
                                       .get<uint32_t>();

        // Ok, we want to kick off an experiment per pixel.
        params.numDirections = experimentConfig.at("experiment")
                                     .at("noisyCircleAngleIntegration")
                                     .at("numDirections")
                                     .get<uint32_t>();
        params.maxArclengthOffset = experimentConfig.at("experiment")
                                          .at("noisyCircleAngleIntegration")
                                          .at("maxArclengthOffset")
                                          .get<float>();
        params.distanceFromPlane = experimentConfig.at("experiment")
                                         .at("noisyCircleAngleIntegration")
                                         .at("distanceFromPlane")
                                         .get<float>();
    } catch (std::exception &ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }

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
    try {
        experimentConfig = nlohmann::json::parse(configFile, nullptr, true, true);
    } catch (nlohmann::json::parse_error &ex) {
        std::cout << "Parse error at byte: " << ex.byte << std::endl;
        exit(1);
    }

    twisty::ExperimentRunner::ExperimentParameters experimentParams
          = twisty::ExperimentRunner::ParseExperimentParamsFromConfig(experimentConfig);

    NoisyCircleParams experimentSpecificParams = ParseExperimentSpecificParams(experimentConfig);

    std::uniform_real_distribution<float> uniformFloat(0.0f, 1.0f);
    std::vector<boost::multiprecision::cpp_dec_float_100> framePixels(
          experimentSpecificParams.framePixelCount * experimentSpecificParams.framePixelCount);
    std::fill(framePixels.begin(), framePixels.end(), 0.0);

    // Vector storing each runs time
    std::vector<uint64_t> runTimes;

    // We are going to bake a big ol table, then use this whenever we need.
    const float minArclength = 1.0f;
    const float maxArclength = 40.0f;
    const float minDs = minArclength / experimentParams.numSegmentsPerCurve;
    const float maxDs = maxArclength / experimentParams.numSegmentsPerCurve;
    const uint32_t numArclengths = 2000;

    std::filesystem::path outputDirectoryPath = experimentParams.experimentDirPath;
    if (!std::filesystem::exists(outputDirectoryPath)) {
        std::filesystem::create_directories(outputDirectoryPath);
    }

    {
        const auto uuid = experimentParams.weightingParameters.GenerateStringUUID();
        std::cout << "Object parameters hash: " << uuid.first << " \n"
                  << uuid.second << '\n'
                  << std::endl;
    }

    std::cout << "Object scatter: " << experimentParams.weightingParameters.scatter << std::endl;
    std::cout << "Object absorption: " << experimentParams.weightingParameters.absorption
              << std::endl;
    twisty::PathWeighting::CachedMultiArclengthWeightLookupTable objectCachedLookupTable(
          experimentParams.weightingParameters, minDs, maxDs, numArclengths);
    objectCachedLookupTable.GetWeightLookupTable(minDs)->ExportValues(
          outputDirectoryPath.string(), std::string("objectLookupTable_minDs.csv"));
    objectCachedLookupTable.GetWeightLookupTable(maxDs)->ExportValues(
          outputDirectoryPath.string(), std::string("objectLookupTable_maxDs.csv"));

    twisty::WeightingParameters environmentWeightingParams = experimentParams.weightingParameters;
    environmentWeightingParams.absorption = 0.0001f;
    environmentWeightingParams.scatter = 0.0001f;
    environmentWeightingParams.mu = 0.01f;

    {
        const auto uuid = environmentWeightingParams.GenerateStringUUID();
        std::cout << "Object parameters hash: " << uuid.first << " \n"
                  << uuid.second << '\n'
                  << std::endl;
    }

    twisty::PathWeighting::CachedMultiArclengthWeightLookupTable environmentCachedLookupTable(
          environmentWeightingParams, minDs, maxDs, numArclengths);

    environmentCachedLookupTable.GetWeightLookupTable(minDs)->ExportValues(
          outputDirectoryPath.string(), std::string("environmentLookupTable_minDs.csv"));
    environmentCachedLookupTable.GetWeightLookupTable(maxDs)->ExportValues(
          outputDirectoryPath.string(), std::string("environmentLookupTable_maxDs.csv"));

    // Example usage:
    glm::vec3 position(0.0f, 0.0f, 0.0f);
    glm::vec3 lookAt(0.0f, 0.0f, 1.0f);
    glm::vec3 up(0.0f, 1.0f, 0.0f);
    float fov = 90.0f;
    float aspectRatio = static_cast<float>(experimentSpecificParams.framePixelCount)
          / static_cast<float>(experimentSpecificParams.framePixelCount);

    Camera camera(position, lookAt, up, fov, aspectRatio, experimentSpecificParams.framePixelCount,
          experimentSpecificParams.framePixelCount);

    // Experiment start time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Setup random number generator
    const uint32_t overallRngSeed = time(0);
    std::cout << "Overall RNG Seed: " << overallRngSeed << std::endl;
    std::mt19937_64 rng(overallRngSeed);

    int64_t totalOpCount
          = experimentSpecificParams.framePixelCount * experimentSpecificParams.framePixelCount;
    int64_t currentOpCount = 0;

    for (int32_t pixelIdxY = 0; pixelIdxY < experimentSpecificParams.framePixelCount; ++pixelIdxY) {
        for (int32_t pixelIdxX = 0; pixelIdxX < experimentSpecificParams.framePixelCount;
              ++pixelIdxX) {
            const uint32_t frameIdx
                  = (pixelIdxY * experimentSpecificParams.framePixelCount) + pixelIdxX;
            const Farlor::Vector3 recieverPos = camera.getPosition();
            const Farlor::Vector3 recieverDir
                  = camera.getRayDirectionForPixel(pixelIdxX, pixelIdxY) * -1.0f;
            // Emitter direction
            const Farlor::Vector3 emitterDir(0.0f, 0.0f, -1.0f);

            twisty::PerturbUtils::BoundaryConditions experimentGeometry;
            experimentGeometry.m_startPos = Farlor::Vector3(0.0f, 0.0f, 10.0f);
            experimentGeometry.m_startDir = emitterDir;
            experimentGeometry.m_endPos = recieverPos;
            experimentGeometry.m_endDir = recieverDir;
            experimentGeometry.arclength = 0.0f;

            const Farlor::Vector3 revserseDir = experimentGeometry.m_endDir * -1.0f;
            const float cosFactor = revserseDir.Dot(camera.getForward());

            const double pathNormalizerLog10 = 0.0f;

            // Single run start time
            const auto singleRunStartTime = std::chrono::high_resolution_clock::now();

            std::uniform_int_distribution<uint64_t> uniformInt(
                  0, std::numeric_limits<uint64_t>::max() - 250);
            const uint64_t rngSeed = uniformInt(rng);

            const twisty::ExperimentBase::Result result
                  = twisty::ExperimentBase::MSegmentPathGenerationMC(rngSeed,
                        experimentParams.numPathsInExperiment, experimentParams.numSegmentsPerCurve,
                        experimentGeometry, experimentParams, pathNormalizerLog10,
                        environmentCachedLookupTable, objectCachedLookupTable, maxDs);

            currentOpCount++;

            // Single run end time
            const auto singleRunEndTime = std::chrono::high_resolution_clock::now();
            // Add time difference to runTimes vector
            const auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(
                  singleRunEndTime - singleRunStartTime);
            runTimes.push_back(timeDiff.count());

            std::ios_base::fmtflags flags = std::cout.flags();
            std::cout << "\rPercent Complete: " << std::fixed << std::setprecision(2)
                      << (100.0f * currentOpCount) / totalOpCount << "%";

            // Estimated time to completion
            std::chrono::high_resolution_clock::time_point currentTime
                  = std::chrono::high_resolution_clock::now();
            // Elapsed time in seconds
            const float elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(
                  currentTime - startTime)
                                               .count();
            const float secondsPerStep = elapsedSeconds / (currentOpCount + 1);
            const float secondsRemaining = secondsPerStep * (totalOpCount - currentOpCount);

            const float mins = std::floor(secondsRemaining / 60.0f);
            const float secs = secondsRemaining - mins * 60.0f;

            std::cout << " Time Remaining: " << std::fixed << std::setprecision(0) << mins << "m"
                      << secs << "s\t\t";
            std::cout.flags(flags);

            if (result.numValidPaths <= 0) {
                continue;
            }


            const boost::multiprecision::cpp_dec_float_100 importanceSampledWeight
                  = result.totalWeight;
            framePixels[frameIdx] += importanceSampledWeight * cosFactor;
        }
    }


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


    OutputRawData(outputDirectoryPath, framePixels, experimentSpecificParams.framePixelCount);

    const std::vector<boost::multiprecision::cpp_dec_float_100> normalizedCombined
          = CalculateNormalizedFrames(framePixels);

    OutputNormalizedData(
          outputDirectoryPath, normalizedCombined, experimentSpecificParams.framePixelCount);

    const std::vector<boost::multiprecision::cpp_dec_float_100> offsetCombined
          = CalculateOffsetFrames(framePixels);
    OutputOffsetData(outputDirectoryPath, offsetCombined, experimentSpecificParams.framePixelCount);

    const std::vector<boost::multiprecision::cpp_dec_float_100> quadAverageCombined
          = CalculateQuadAvgFrames(framePixels, experimentSpecificParams.framePixelCount);
    OutputQuadAverageData(
          outputDirectoryPath, quadAverageCombined, experimentSpecificParams.framePixelCount);

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
          [&minimum, &range](boost::multiprecision::cpp_dec_float_100 &n) {
              n = boost::multiprecision::pow(((n - minimum) / range), 1.0 / 2.2);
          });
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
          = boost::multiprecision::cpp_dec_float_100(100.0) / maximum;

    std::for_each(result.begin(), result.end(),
          [&invMax](boost::multiprecision::cpp_dec_float_100 &n) { n = n * invMax; });
    return result;
}

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateQuadAvgFrames(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights,
      const uint32_t totalWidth)
{
    if (rawFrameWeights.empty()) {
        return rawFrameWeights;
    }
    std::vector<boost::multiprecision::cpp_dec_float_100> result = rawFrameWeights;
    const uint32_t halfFrameWidth = totalWidth / 2;

    for (int xOffset = 0; xOffset <= halfFrameWidth; xOffset++) {
        for (int yOffset = 0; yOffset <= halfFrameWidth; yOffset++) {
            const uint32_t q0Idx = xOffset * totalWidth + yOffset;
            const uint32_t q1Idx = xOffset * totalWidth + (totalWidth - 1 - yOffset);
            const uint32_t q2Idx = (totalWidth - 1 - xOffset) * totalWidth + yOffset;
            const uint32_t q3Idx
                  = (totalWidth - 1 - xOffset) * totalWidth + (totalWidth - 1 - yOffset);
            boost::multiprecision::cpp_dec_float_100 avg
                  = (result[q0Idx] + result[q1Idx] + result[q2Idx] + result[q3Idx]) * 0.25;
            result[q0Idx] = avg;
            result[q1Idx] = avg;
            result[q2Idx] = avg;
            result[q3Idx] = avg;
        }
    }
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
    for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
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
    for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
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
    for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
            rawDataOutfile << offsetCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}

void OutputQuadAverageData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &quadAverageCombined,
      const int32_t framePixelCount)
{
    std::filesystem::path rawDataFilepath = outputDirectoryPath;
    rawDataFilepath /= "Combined/";

    if (!std::filesystem::exists(rawDataFilepath)) {
        std::filesystem::create_directories(rawDataFilepath);
    }

    rawDataFilepath /= "quadAverageData.dat";
    std::ofstream rawDataOutfile(rawDataFilepath.string());
    if (!rawDataOutfile.is_open()) {
        std::cout << "Failed to create offsetData: " << rawDataFilepath.string() << std::endl;
        exit(1);
    }

    // X
    rawDataOutfile << framePixelCount << " " << framePixelCount << std::endl;

    // Write out the pixel data
    for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
            rawDataOutfile << quadAverageCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}
