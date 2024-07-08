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

// Enable this for better exception messages, worth the tradeoff for us
#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

// #ifdef __linux__
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Composite.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class VDBVolume {
   public:
    VDBVolume(const float outerRadius, float innerRadius)
    {
        m_grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
              outerRadius, openvdb::Vec3f(0.0f, 0.0f, 0.0f), 0.01f);
        openvdb::FloatGrid::Ptr removeGrid
              = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
                    innerRadius, openvdb::Vec3f(0.0f, 0.0f, 0.0f), 0.01f);
        m_grid = openvdb::tools::csgDifferenceCopy(*m_grid, *removeGrid);
        if (m_grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
            openvdb::tools::sdfToFogVolume(*m_grid);
        }
    }

    VDBVolume(std::string filename, float scale, float xOffset, float yOffset, float zOffset)
    {
        openvdb::io::File vdbFile(filename);
        vdbFile.open();

        openvdb::GridBase::Ptr baseGrid = vdbFile.readGrid(vdbFile.beginName().gridName());
        vdbFile.close();

        m_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        if (m_grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
            openvdb::tools::sdfToFogVolume(*m_grid);
        }

        openvdb::math::Transform::Ptr linearTransform
              = openvdb::math::Transform::createLinearTransform(scale);
        linearTransform->postRotate(-openvdb::math::pi<double>() / 2.0, openvdb::math::Y_AXIS);
        linearTransform->postTranslate(openvdb::Vec3d(xOffset, yOffset, zOffset));
        m_grid->setTransform(linearTransform);
    }

    // We auto center with just the scale
    VDBVolume(std::string filename, float scale)
    {
        openvdb::io::File vdbFile(filename);
        vdbFile.open();

        openvdb::GridBase::Ptr baseGrid = vdbFile.readGrid(vdbFile.beginName().gridName());
        vdbFile.close();

        m_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        if (m_grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
            openvdb::tools::sdfToFogVolume(*m_grid);
        }

        openvdb::math::Transform::Ptr linearTransform
              = openvdb::math::Transform::createLinearTransform(scale);
        linearTransform->preRotate(-openvdb::math::pi<double>() / 2.0, openvdb::math::Y_AXIS);
        m_grid->setTransform(linearTransform);

        openvdb::math::CoordBBox bbox = m_grid->evalActiveVoxelBoundingBox();
        openvdb::Vec3d offset = m_grid->indexToWorld(bbox.max()) + m_grid->indexToWorld(bbox.min());
        offset *= 0.5;

        openvdb::math::Transform::Ptr offsetLinear
              = openvdb::math::Transform::createLinearTransform(scale);
        offsetLinear->preRotate(-openvdb::math::pi<double>() / 2.0, openvdb::math::Y_AXIS);
        offsetLinear->postTranslate(-offset + openvdb::Vec3d(8.0, 0.0, 0.0));
        m_grid->setTransform(offsetLinear);
    }

    void PrintMetadata() const
    {
        for (openvdb::MetaMap::MetaIterator iter = m_grid->beginMeta(); iter != m_grid->endMeta();
              ++iter) {
            const std::string &name = iter->first;
            openvdb::Metadata::Ptr value = iter->second;
            std::string valueAsString = value->str();
            std::cout << name << " = " << valueAsString << std::endl;
        }
    }

    void PrintWorldBB()
    {
        openvdb::math::CoordBBox bbox = m_grid->evalActiveVoxelBoundingBox();
        std::cout << m_grid->indexToWorld(bbox.min()) << ", " << m_grid->indexToWorld(bbox.max())
                  << std::endl;
    }

    openvdb::FloatGrid::Ptr AccessGrid() { return m_grid; }

   private:
    openvdb::FloatGrid::Ptr m_grid = nullptr;
};
// #endif

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateLogData(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights);

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateNormalizedData(
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawFrameWeights);

void OutputRawData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawCombined,
      const int32_t framePixelWidth, const int32_t framePixelHeight);

void OutputLogData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &offsetCombined,
      const int32_t framePixelWidth, const int32_t framePixelHeight);

void OutputNormalizedData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &normalizedCombined,
      const int32_t framePixelWidth, const int32_t framePixelHeight);

class Camera {
   public:
    Camera()
        : position(0.0f, 0.0f, 0.0f)
        , fov(0.0f)
        , aspectRatio(0.0f)
        , imageWidth(0)
        , imageHeight(0)
    {
    }

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

    Farlor::Vector3 getForward() const { return Farlor::Vector3(forward.x, forward.y, forward.z); }

    int getWidth() { return imageWidth; }
    int getHeight() { return imageHeight; }

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
    // Ok, we want to kick off an experiment per pixel.
    float maxArclengthOffset = 1.0f;
    float distanceFromPlane = 1.0f;
};

Camera LoadCameraFromConfig(nlohmann::json &experimentConfig)
{
    glm::vec3 position;
    position.x = experimentConfig.at("experiment").at("camera").at("eyePos").at("x").get<float>();
    position.y = experimentConfig.at("experiment").at("camera").at("eyePos").at("y").get<float>();
    position.z = experimentConfig.at("experiment").at("camera").at("eyePos").at("z").get<float>();

    glm::vec3 lookAt;
    lookAt.x = experimentConfig.at("experiment").at("camera").at("lookAt").at("x").get<float>();
    lookAt.y = experimentConfig.at("experiment").at("camera").at("lookAt").at("y").get<float>();
    lookAt.z = experimentConfig.at("experiment").at("camera").at("lookAt").at("z").get<float>();

    glm::vec3 up;
    up.x = experimentConfig.at("experiment").at("camera").at("up").at("x").get<float>();
    up.y = experimentConfig.at("experiment").at("camera").at("up").at("y").get<float>();
    up.z = experimentConfig.at("experiment").at("camera").at("up").at("z").get<float>();

    float fovDegrees = experimentConfig.at("experiment").at("camera").at("fovDegrees").get<float>();

    int framePixelWidth
          = experimentConfig.at("experiment").at("camera").at("framePixelWidth").get<int>();
    int framePixelHeight
          = experimentConfig.at("experiment").at("camera").at("framePixelHeight").get<int>();

    Camera camera(position, lookAt, up, fovDegrees,
          static_cast<float>(framePixelWidth) / static_cast<float>(framePixelHeight),
          framePixelWidth, framePixelHeight);
    return camera;
}

NoisyCircleParams ParseExperimentSpecificParams(nlohmann::json &experimentConfig)
{
    NoisyCircleParams params;
    try {
        params.maxArclengthOffset = experimentConfig.at("experiment")
                                          .at("noisyCircleAngleIntegration")
                                          .at("maxArclengthOffset")
                                          .get<float>();
        params.distanceFromPlane = experimentConfig.at("experiment")
                                         .at("noisyCircleAngleIntegration")
                                         .at("distanceFromPlane")
                                         .get<float>();
    } catch (const std::exception &ex) {
        std::cout << "Exception: " << ex.what() << std::endl;
    }
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

    // We are going to bake a big ol table, then use this whenever we need.
    const float minArclength = 10.0f;
    const float maxArclength = 25.0f;
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

    std::vector<Farlor::Vector3> emitterLocations;
    emitterLocations.push_back(Farlor::Vector3(0.0f, 0.0f, -10.0f));
    std::vector<Farlor::Vector3> emitterDirections;
    emitterDirections.push_back(Farlor::Vector3(0.0f, 0.0f, 1.0f));

    Camera camera;
    try {
        camera = LoadCameraFromConfig(experimentConfig);
    } catch (std::exception &ex) {
        std::cout << "ex: " << ex.what() << std::endl;
    }
    const int imageWidth = camera.getWidth();
    const int imageHeight = camera.getHeight();

    std::vector<boost::multiprecision::cpp_dec_float_100> framePixels(imageWidth * imageHeight);

    std::vector<boost::multiprecision::cpp_dec_float_100> combinedPixels(imageWidth * imageHeight);

    // Vector storing each runs time
    std::vector<uint64_t> runTimes(imageWidth * imageHeight);
    std::fill(runTimes.begin(), runTimes.end(), 0.0f);

    // #ifdef __linux__
    // Must call once
    openvdb::initialize();

    //     VDBVolume bunny("bunny.vdb", 1.0f / 75.0f);
    VDBVolume bunny(5.0f, 4.0f);
    bunny.PrintMetadata();
    bunny.PrintWorldBB();
    // #endif

    // Experiment start time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Setup random number generator
    const uint32_t overallRngSeed = time(0);
    std::cout << "Overall RNG Seed: " << overallRngSeed << std::endl;
    std::mt19937_64 rng(overallRngSeed);

    int64_t totalOpCount = imageWidth * imageHeight * emitterLocations.size();
    int64_t currentOpCount = 0;

    for (int32_t pixelIdxY = 0; pixelIdxY < imageHeight; ++pixelIdxY) {
        for (int32_t pixelIdxX = 0; pixelIdxX < imageWidth; ++pixelIdxX) {
            const uint32_t frameIdx = (pixelIdxY * imageWidth) + pixelIdxX;

            const Farlor::Vector3 recieverPos = camera.getPosition();
            const Farlor::Vector3 recieverDir
                  = camera.getRayDirectionForPixel(pixelIdxX, pixelIdxY);
            // std::cout << "Receiver Dir: " << recieverDir << std::endl;

            int64_t numValidEmitterSolutions = emitterLocations.size();

            for (int emitterIdx = 0; emitterIdx < emitterLocations.size(); emitterIdx++) {
                // Flip the plane normal so we are facing the correct way
                // Emitter direction
                twisty::PerturbUtils::BoundaryConditions experimentGeometry;
                experimentGeometry.m_startPos = emitterLocations[emitterIdx];
                experimentGeometry.m_startDir = emitterDirections[emitterIdx];
                experimentGeometry.m_endPos = recieverPos;
                experimentGeometry.m_endDir = recieverDir;
                experimentGeometry.arclength = 0.0f;

                const Farlor::Vector3 revserseDir = experimentGeometry.m_endDir * -1.0f;
                //     std::cout << "Reverse Dir: " << revserseDir << std::endl;
                const float cosFactor = revserseDir.Dot(camera.getForward());
                //     std::cout << "Cos Factor: " << cosFactor << std::endl;

                const double pathNormalizerLog10 = 0.0f;

                // Single run start time
                const auto singleRunStartTime = std::chrono::high_resolution_clock::now();

                std::uniform_int_distribution<uint64_t> uniformInt(
                      0, std::numeric_limits<uint64_t>::max() - 250);
                const uint64_t rngSeed = uniformInt(rng);

                // #ifdef __linux__
                //     std::cout << "Linux vdb bunny version" << std::endl;
                const twisty::ExperimentBase::Result result
                      = twisty::ExperimentBase::MSegmentPathGenerationMC_VDB(rngSeed,
                            experimentParams.numPathsInExperiment,
                            experimentParams.numSegmentsPerCurve, experimentGeometry,
                            experimentParams, pathNormalizerLog10, environmentCachedLookupTable,
                            objectCachedLookupTable, maxDs, bunny.AccessGrid());
                // #else
                //     const twisty::ExperimentBase::Result result
                //           = twisty::ExperimentBase::MSegmentPathGenerationMC(rngSeed,
                //                 experimentParams.numPathsInExperiment,
                //                 experimentParams.numSegmentsPerCurve, experimentGeometry,
                //                 experimentParams, pathNormalizerLog10, environmentCachedLookupTable,
                //                 objectCachedLookupTable, maxDs);
                // #endif
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
                const float elapsedSeconds
                      = std::chrono::duration_cast<std::chrono::duration<float>>(
                            currentTime - startTime)
                              .count();
                const float secondsPerStep = elapsedSeconds / (currentOpCount + 1);
                const float secondsRemaining = secondsPerStep * (totalOpCount - currentOpCount);

                const float mins = std::floor(secondsRemaining / 60.0f);
                const float secs = secondsRemaining - mins * 60.0f;

                std::cout << " Time Remaining: " << std::fixed << std::setprecision(0) << mins
                          << "m" << secs << "s\t\t";
                std::cout.flags(flags);

                if (result.numValidPaths <= 0) {
                    numValidEmitterSolutions--;
                    continue;
                }
                const boost::multiprecision::cpp_dec_float_100 importanceSampledWeight
                      = result.totalWeight;
                boost::multiprecision::cpp_dec_float_100 weight_log10
                      = boost::multiprecision::log10(importanceSampledWeight * cosFactor);
                framePixels[frameIdx] += importanceSampledWeight * cosFactor;
            }
            if (numValidEmitterSolutions > 0) {
                framePixels[frameIdx] /= numValidEmitterSolutions;
            }
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
    const uint64_t totalRunTime = std::accumulate(runTimes.begin(), runTimes.end(), 0.0);

    const uint64_t averageRunTime = totalRunTime / runTimes.size();
    std::cout << "Average run time: " << averageRunTime << "ms" << std::endl;

    // Avg run time in seconds
    const double averageRunTimeSeconds = averageRunTime / 1000.0;
    std::cout << "Average run time: " << averageRunTimeSeconds << "s" << std::endl;

    // Average run time in minutes
    const double averageRunTimeMinutes = averageRunTimeSeconds / 60.0;
    std::cout << "Average run time: " << averageRunTimeMinutes << "m" << std::endl;

    OutputRawData(outputDirectoryPath, framePixels, imageWidth, imageHeight);

    const std::vector<boost::multiprecision::cpp_dec_float_100> logData
          = CalculateLogData(framePixels);
    OutputLogData(outputDirectoryPath, CalculateLogData(framePixels), imageWidth, imageHeight);

    const std::vector<boost::multiprecision::cpp_dec_float_100> normalizedLogData
          = CalculateNormalizedData(logData);
    OutputNormalizedData(outputDirectoryPath, normalizedLogData, imageWidth, imageHeight);

    std::cout << "Experiment done" << std::endl;

    return 0;
}

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateLogData(
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

    std::for_each(
          result.begin(), result.end(), [&invMax](boost::multiprecision::cpp_dec_float_100 &n) {
              n = boost::multiprecision::log10(n);
          });
    return result;
}

std::vector<boost::multiprecision::cpp_dec_float_100> CalculateNormalizedData(
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
      const int32_t framePixelWidth, const int32_t framePixelHeight)
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
    rawDataOutfile << framePixelWidth << " " << framePixelHeight << std::endl;

    // Write out the pixel data
    for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelHeight; ++pixelIdxY) {
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelWidth; ++pixelIdxX) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelWidth + pixelIdxX;
            rawDataOutfile << rawCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}

void OutputLogData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &offsetCombined,
      const int32_t framePixelWidth, const int32_t framePixelHeight)
{
    std::filesystem::path rawDataFilepath = outputDirectoryPath;
    rawDataFilepath /= "Combined/";

    if (!std::filesystem::exists(rawDataFilepath)) {
        std::filesystem::create_directories(rawDataFilepath);
    }

    rawDataFilepath /= "logData.dat";
    std::ofstream rawDataOutfile(rawDataFilepath.string());
    if (!rawDataOutfile.is_open()) {
        std::cout << "Failed to create logData: " << rawDataFilepath.string() << std::endl;
        exit(1);
    }

    // X
    rawDataOutfile << framePixelWidth << " " << framePixelHeight << std::endl;

    // Write out the pixel data
    for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelHeight; ++pixelIdxY) {
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelWidth; ++pixelIdxX) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelWidth + pixelIdxX;
            rawDataOutfile << offsetCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}

void OutputNormalizedData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &normalizedCombined,
      const int32_t framePixelWidth, const int32_t framePixelHeight)
{
    std::filesystem::path rawDataFilepath = outputDirectoryPath;
    rawDataFilepath /= "Combined/";

    if (!std::filesystem::exists(rawDataFilepath)) {
        std::filesystem::create_directories(rawDataFilepath);
    }

    rawDataFilepath /= "normalizedLogData.dat";
    std::ofstream rawDataOutfile(rawDataFilepath.string());
    if (!rawDataOutfile.is_open()) {
        std::cout << "Failed to create normalizedData: " << rawDataFilepath.string() << std::endl;
        exit(1);
    }

    // X
    rawDataOutfile << framePixelWidth << " " << framePixelHeight << std::endl;

    // Write out the pixel data
    for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelHeight; ++pixelIdxY) {
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelWidth; ++pixelIdxX) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelWidth + pixelIdxX;
            rawDataOutfile << normalizedCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}