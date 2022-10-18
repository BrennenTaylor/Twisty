#include "CombinedWeightUtils.h"
#include "Curve.h"
#include "CurvePerturbUtils.h"
#include "ExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "boost/multiprecision/cpp_dec_float.hpp"

#if defined(USE_CUDA)
#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"
#endif

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

boost::multiprecision::cpp_dec_float_100 AngleIntegration(const uint32_t numPhi1Vals,
      const uint32_t numTheta1Vals, const uint32_t numTheta2Vals,
      const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
      const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
      const boost::multiprecision::cpp_dec_float_100 &pathNormalizer,
      const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable);

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

    params.numPhi1Vals
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["numPhi1Vals"];
    params.numTheta1Vals
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["numTheta1Vals"];
    params.numTheta2Vals
          = experimentConfig["experiment"]["noisyCircleAngleIntegration"]["numTheta2Vals"];
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

                const uint32_t frameIdx
                      = (pixelIdxY + halfFrameWidth) * experimentSpecificParams.framePixelCount
                      + (pixelIdxZ + halfFrameWidth);
                framePixels[frameIdx] = AngleIntegration(experimentSpecificParams.numPhi1Vals,
                      experimentSpecificParams.numTheta1Vals,
                      experimentSpecificParams.numTheta2Vals, experimentParams, experimentGeometry,
                      pathNormalizer, weightLookupTable);

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

boost::multiprecision::cpp_dec_float_100 AngleIntegration(const uint32_t numPhi1Vals,
      const uint32_t numTheta1Vals, const uint32_t numTheta2Vals,
      const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
      const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
      const boost::multiprecision::cpp_dec_float_100 &pathNormalizer,
      const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
{
    std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;

    const float ds
          = experimentGeometry.arclength / static_cast<float>(experimentParams.numSegmentsPerCurve);

    // Ok, generate the curve.
    Farlor::Vector3 point0 = experimentGeometry.m_startPos;
    Farlor::Vector3 point1 = point0 + experimentGeometry.m_startDir * ds;

    Farlor::Vector3 point5 = experimentGeometry.m_endPos;
    Farlor::Vector3 point4 = point5 - experimentGeometry.m_endDir * ds;

    // Calculate the second point using theta and cos values.

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = twisty::TwistyPi;
    const float dPhi1 = (phi1Max - phi1Min) / static_cast<float>(numPhi1Vals);

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const float dTheta1 = (theta1Max - theta1Min) / static_cast<float>(numTheta1Vals);

    // Azimuthal
    const float theta2Min = -twisty::TwistyPi;
    const float theta2Max = twisty::TwistyPi;
    const float dTheta2 = (theta2Max - theta2Min) / static_cast<float>(numTheta2Vals);

    uint64_t numValidPaths = 0;
    twisty::CombinedWeightValues_C activeWeightValue;
    twisty::CombinedWeightValues_C_Reset(activeWeightValue);

    for (size_t phi1Idx = 0; phi1Idx < numPhi1Vals; phi1Idx++) {
        const float phi1 = phi1Min + phi1Idx * dPhi1;

        for (size_t theta1Idx = 0; theta1Idx < numTheta1Vals; theta1Idx++) {
            const float theta1 = theta1Min + theta1Idx * dTheta1;

            /*
                  x = ρsinφcosθ
                  y = ρsinφsinθ
                  z = ρcosφ 
            */

            const float sinPhi1 = std::sin(phi1);
            const float cosPhi1 = std::cos(phi1);
            const float sinTheta1 = std::sin(theta1);
            const float cosTheta1 = std::cos(theta1);

            // Calculate the first segment position
            const Farlor::Vector3 segment1Dir(sinPhi1 * cosTheta1, sinPhi1 * sinTheta1, cosPhi1);
            const Farlor::Vector3 point2 = point1 + segment1Dir * ds;

            if (const float remainingDistance2 = (point4 - point2).SqrMagnitude();
                  (4 * ds * ds) < remainingDistance2) {
                continue;
            }

            // If not, we keep going through the possible combinations
            for (size_t theta2Idx = 0; theta2Idx < numTheta2Vals; theta2Idx++) {
                const float theta2 = theta2Min + static_cast<float>(theta2Idx) * dTheta2;
                // TODO: Finish out the remaining piece

                const Farlor::Vector3 x_p = (point2 + point4) * 0.5;
                const Farlor::Vector3 lineUnitDir = (point4 - point2).Normalized();

                Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
                if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
                    otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
                }

                const Farlor::Vector3 normalToLine = lineUnitDir.Cross(otherCrossVec).Normalized();

                // We should have an even number of segments remaining
                const float hypot = ds;
                const float D_2 = (point4 - point2).Magnitude() * 0.5f;
                assert(D_2 > hypot && "This should never be reached due to earlier check.");

                const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
                Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

                // Now rotate randomly theta amount around the axis.
                {
                    const float sinRotAngle = std::sinf(theta2 / 2.0f);
                    std::array<float, 4> quaternionRotation
                          = { std::cosf(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
                                lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


                    Farlor::Vector3 shiftedPoint = x_t - point2;
                    // Rotate and stuff back in shifted point
                    twisty::RotateVectorByQuaternion(
                          quaternionRotation.data(), shiftedPoint.m_data.data());
                    // Update the point with the rotated version
                    x_t = shiftedPoint + point2;
                }
                const Farlor::Vector3 point3 = x_t;

                std::array<Farlor::Vector3, 6> points
                      = { point0, point1, point2, point3, point4, point5 };
                std::array<Farlor::Vector3, 5> tangents;
                std::array<float, 4> curvatures;

                twisty::PerturbUtils::UpdateTangentsFromPos(
                      points.data(), tangents.data(), 5, experimentGeometry);
                twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                      tangents.data(), curvatures.data(), 5, experimentGeometry);

                double scatteringWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                      curvatures.data(), 4, weightLookupTable);
                if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                    twisty::CombinedWeightValues_C_AddValue(
                          activeWeightValue, scatteringWeightLog10);
                } else {
                    combinedWeightValues.push_back(activeWeightValue);
                    twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                    twisty::CombinedWeightValues_C_AddValue(
                          activeWeightValue, scatteringWeightLog10);
                }
                numValidPaths++;
            }
        }
    }


    boost::multiprecision::cpp_dec_float_100 pathIntegralResult = 0.0;
    for (const auto &combinedWeightValue : combinedWeightValues) {
        pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue) * pathNormalizer;
    }

    return (numValidPaths > 0)
          ? boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths)
          : 0.0;
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