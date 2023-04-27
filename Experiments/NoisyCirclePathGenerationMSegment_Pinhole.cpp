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

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Composite.h>

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

        openvdb::v10_0::math::CoordBBox bbox = m_grid->evalActiveVoxelBoundingBox();
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
        openvdb::v10_0::math::CoordBBox bbox = m_grid->evalActiveVoxelBoundingBox();
        std::cout << m_grid->indexToWorld(bbox.min()) << ", " << m_grid->indexToWorld(bbox.max())
                  << std::endl;
    }

    openvdb::FloatGrid::Ptr AccessGrid() { return m_grid; }

   private:
    openvdb::FloatGrid::Ptr m_grid = nullptr;
};

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

    Farlor::Vector3 cameraCenter(experimentSpecificParams.distanceFromPlane, 0.0f, 0.0f);
    Farlor::Vector3 centerOfFocalFrame(
          experimentSpecificParams.distanceFromPlane - 15.0, 0.0f, 0.0f);

    std::vector<Farlor::Vector3> emitterLocations;
//     emitterLocations.push_back(Farlor::Vector3(5.0f, 10.0f, 0.0f));
//     emitterLocations.push_back(Farlor::Vector3(5.0f, -10.0f, 0.0f));
//     emitterLocations.push_back(Farlor::Vector3(5.0f, 0.0f, 10.0f));
//     emitterLocations.push_back(Farlor::Vector3(5.0f, 0.0f, -10.0f)); 

        emitterLocations.push_back(Farlor::Vector3(0.0f, 8.0f, 0.0f));
        emitterLocations.push_back(Farlor::Vector3(0.0f, -8.0f, 0.0f));
        emitterLocations.push_back(Farlor::Vector3(0.0f, 0.0f, 8.0f));
        emitterLocations.push_back(Farlor::Vector3(0.0f, 0.0f, -8.0f));
            
            // emitterLocations.push_back(Farlor::Vector3(0.0f, 0.0f, 0.0f));
            // emitterLocations.push_back(Farlor::Vector3(0.0f, 0.0f, 0.0f));
            // emitterLocations.push_back(Farlor::Vector3(0.0f, 0.0f, 0.0f));
            // emitterLocations.push_back(Farlor::Vector3(0.0f, 0.0f, 0.0f));

        //     emitterLocations.push_back(Farlor::Vector3(24.0f, 0.0f, 0.0f));

        std::vector<Farlor::Vector3> emitterDirections;
        emitterDirections.push_back(Farlor::Vector3(0.0f, -1.0f, 0.0f));
        emitterDirections.push_back(Farlor::Vector3(0.0f, 1.0f, 0.0f));
        emitterDirections.push_back(Farlor::Vector3(0.0f, 0.0f, -1.0f));
        emitterDirections.push_back(Farlor::Vector3(0.0f, 0.0f, 1.0f));

            // emitterDirections.push_back(Farlor::Vector3(0.0f, 1.0f, 0.0f));
            // emitterDirections.push_back(Farlor::Vector3(0.0f, -1.0f, 0.0f));
            // emitterDirections.push_back(Farlor::Vector3(0.0f, 0.0f, 1.0f));
            // emitterDirections.push_back(Farlor::Vector3(0.0f, 0.0f, -1.0f));
        //     emitterDirections.push_back(Farlor::Vector3(-1.0f, 0.0f, 0.0f));

        int32_t halfFrameWidth = experimentSpecificParams.framePixelCount / 2;

        // Vector storing each runs time
        std::vector<uint64_t> runTimes;

        for (uint32_t r = 0; r < experimentSpecificParams.framePixelCount; r++) {
        for (uint32_t c = 0; c < experimentSpecificParams.framePixelCount; c++) {
            const uint32_t frameIdx = r * experimentSpecificParams.framePixelCount + c;
            framePixels[frameIdx] = 0.0;
        }
    }

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
      std::cout << "Object absorption: " << experimentParams.weightingParameters.absorption << std::endl;

      // experimentParams.weightingParameters.bias = 10.0;//twisty::TwistyPi;

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

    const Farlor::Vector3 planeNormal = Farlor::Vector3(-1.0f, 0.0f, 0.0f);
    const Farlor::Vector3 planeNormalO1 = Farlor::Vector3(0.0f, 1.0f, 0.0f);
    const Farlor::Vector3 planeNormalO2 = Farlor::Vector3(0.0f, 0.0f, 1.0f);

    // Must call once
    openvdb::initialize();

    VDBVolume bunny("bunny.vdb", 1.0f / 75.0f);
    // VDBVolume bunny(OuterSphereRadius, InnerSphereRadius);
    bunny.PrintMetadata();
    bunny.PrintWorldBB();

    // Experiment start time
    auto startTime = std::chrono::high_resolution_clock::now();

      // Setup random number generator
    const uint32_t overallRngSeed = time(0);
    std::cout << "Overall RNG Seed: " << overallRngSeed << std::endl;
     std::mt19937_64 rng(overallRngSeed);

    int64_t totalOpCount = experimentSpecificParams.framePixelCount
          * experimentSpecificParams.framePixelCount * emitterLocations.size();
    int64_t currentOpCount = 0;

    for (int emitterIdx = 0; emitterIdx < emitterLocations.size(); emitterIdx++) {
        for (int32_t pixelIdxZ = -halfFrameWidth; pixelIdxZ <= halfFrameWidth; ++pixelIdxZ) {
            //std::cout << "Pixel Idx X: " << pixelIdxZ << std::endl;
            for (int32_t pixelIdxY = -halfFrameWidth; pixelIdxY <= halfFrameWidth; ++pixelIdxY) {
                //std::cout << "Pixel Idx Y: " << pixelIdxY << std::endl;

                const uint32_t frameIdx
                      = (pixelIdxY + halfFrameWidth) * experimentSpecificParams.framePixelCount
                      + (pixelIdxZ + halfFrameWidth);


                // Flip the plane normal so we are facing the correct way

                const Farlor::Vector3 pixelCenter = centerOfFocalFrame
                      + planeNormalO1 * (pixelIdxY * pixelLength)
                      + planeNormalO2 * (pixelIdxZ * pixelLength);

                const Farlor::Vector3 recieverDir = (cameraCenter - pixelCenter).Normalized();

                // Emitter direction
                twisty::PerturbUtils::BoundaryConditions experimentGeometry;
                experimentGeometry.m_startPos = emitterLocations[emitterIdx];
                experimentGeometry.m_startDir = emitterDirections[emitterIdx];
                experimentGeometry.m_endPos = cameraCenter;
                experimentGeometry.m_endDir = recieverDir;
                experimentGeometry.arclength = 0.0f;

                const Farlor::Vector3 revserseDir = experimentGeometry.m_endDir * -1.0f;
                const float cosFactor = revserseDir.Dot(planeNormal);

                const double pathNormalizerLog10 = 0.0f;

                // Single run start time
                const auto singleRunStartTime = std::chrono::high_resolution_clock::now();

                std::uniform_int_distribution<uint64_t> uniformInt(
                      0, std::numeric_limits<uint64_t>::max() - 250);
                const uint64_t rngSeed = uniformInt(rng);

            //     const twisty::ExperimentBase::Result result
            //           = twisty::ExperimentBase::MSegmentPathGenerationMC_VDB(rngSeed,
            //                 experimentParams.numPathsInExperiment,
            //                 experimentParams.numSegmentsPerCurve, experimentGeometry,
            //                 experimentParams, pathNormalizerLog10, environmentCachedLookupTable,
            //                 objectCachedLookupTable, maxDs, bunny.AccessGrid());
                const twisty::ExperimentBase::Result result
                      = twisty::ExperimentBase::MSegmentPathGenerationMC(rngSeed,
                            experimentParams.numPathsInExperiment,
                            experimentParams.numSegmentsPerCurve, experimentGeometry,
                            experimentParams, pathNormalizerLog10, environmentCachedLookupTable,
                            objectCachedLookupTable, maxDs);
                const boost::multiprecision::cpp_dec_float_100 importanceSampledWeight
                      = result.totalWeight;

                currentOpCount++;

                // Single run end time
                const auto singleRunEndTime = std::chrono::high_resolution_clock::now();
                // Add time difference to runTimes vector
                const auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(
                      singleRunEndTime - singleRunStartTime);
                runTimes.push_back(timeDiff.count());
                /*
                std::cout << "Num valid paths: " << result.numValidPaths << "/"
                          << result.numPathsTotal << std::endl;
                std::cout << "Percent valid paths: "
                          << (result.numValidPaths / (float)result.numPathsTotal) * 100.0f << "%"
                          << std::endl;
                std::cout << "Total weight: " << importanceSampledWeight << std::endl;
                std::cout << "Total weight w/ cos factor: " << importanceSampledWeight * cosFactor
                          << std::endl;
*/


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

                boost::multiprecision::cpp_dec_float_100 weight_log10
                      = boost::multiprecision::log10(importanceSampledWeight * cosFactor);
                // if (weight_log10 > -10) {
                //     std::cout << "Odd bug encountered, lets toss the path" << std::endl;
                // } else {
                if (result.numValidPaths > 0) {
                    framePixels[frameIdx]
                          += importanceSampledWeight * cosFactor * (1.0f / emitterLocations.size());
                }
                // }
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
    for (uint32_t pixelIdxZ = 0; pixelIdxZ < framePixelCount; ++pixelIdxZ) {
        for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY) {
            // Output pixel
            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxZ;
            rawDataOutfile << quadAverageCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}
