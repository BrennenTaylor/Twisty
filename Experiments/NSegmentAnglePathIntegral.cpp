#include <ExperimentRunner.h>
#include <PathWeighters.h>
#include "CombinedWeightUtils.h"

#include "MathConsts.h"
#include "boost/multiprecision/detail/default_ops.hpp"

#include <FMath/FMath.h>

#include <nlohmann/json.hpp>
#include <string>
#include <chrono>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

std::string format_duration(std::chrono::milliseconds ms)
{
    using namespace std::chrono;
    auto secs = duration_cast<seconds>(ms);
    ms -= duration_cast<milliseconds>(secs);
    auto mins = duration_cast<minutes>(secs);
    secs -= duration_cast<seconds>(mins);
    auto hour = duration_cast<hours>(mins);
    mins -= duration_cast<minutes>(hour);

    std::stringstream ss;
    ss << hour.count() << " Hours : " << mins.count() << " Minutes : " << secs.count()
       << " Seconds : " << ms.count() << " Milliseconds";
    return ss.str();
}

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
    experimentParams.weightingParameters.minBound = 0.0f;
    experimentParams.weightingParameters.maxBound
          = 10.0f / experimentParams.weightingParameters.eps;

    return experimentParams;
}

int main(int argc, char *argv[])
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
    std::cout << "Segment Count: " << experimentParams.numSegmentsPerCurve << std::endl;

    if (!std::filesystem::exists(experimentParams.experimentDirPath)) {
        std::filesystem::create_directories(experimentParams.experimentDirPath);
    }
    std::cout << "experimentDirPath: " << experimentParams.experimentDirPath << std::endl;
    const std::string experimentCfgCopyFilename = std::string(experimentParams.experimentDirPath)
          + "/" + experimentParams.experimentName + ".json";

    if (!std::filesystem::exists(experimentCfgCopyFilename)) {
        std::filesystem::copy_file(argv[1], experimentCfgCopyFilename,
              std::filesystem::copy_options::overwrite_existing);
    }

    std::ofstream resultsOFS(experimentParams.experimentDirPath + "/Results.txt");
    if (!resultsOFS.is_open()) {
        std::cout << "Failed to open results file: " << experimentParams.experimentDirPath
                  << "/Results.txt" << std::endl;
        return 1;
    }

    twisty::PerturbUtils::BoundaryConditions experimentGeometry;
    {
        float x = experimentConfig["experiment"]["sixSegmentDoF"]["startPos"][0];
        float y = experimentConfig["experiment"]["sixSegmentDoF"]["startPos"][1];
        float z = experimentConfig["experiment"]["sixSegmentDoF"]["startPos"][2];
        experimentGeometry.m_startPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["sixSegmentDoF"]["startDir"][0];
        float y = experimentConfig["experiment"]["sixSegmentDoF"]["startDir"][1];
        float z = experimentConfig["experiment"]["sixSegmentDoF"]["startDir"][2];
        experimentGeometry.m_startDir = Farlor::Vector3(x, y, z).Normalized();
    }

    {
        float x = experimentConfig["experiment"]["sixSegmentDoF"]["endPos"][0];
        float y = experimentConfig["experiment"]["sixSegmentDoF"]["endPos"][1];
        float z = experimentConfig["experiment"]["sixSegmentDoF"]["endPos"][2];
        experimentGeometry.m_endPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["sixSegmentDoF"]["endDir"][0];
        float y = experimentConfig["experiment"]["sixSegmentDoF"]["endDir"][1];
        float z = experimentConfig["experiment"]["sixSegmentDoF"]["endDir"][2];
        experimentGeometry.m_endDir = Farlor::Vector3(x, y, z).Normalized();
    }
    // Force to a value
    experimentGeometry.arclength = experimentParams.arclength
          = experimentConfig["experiment"]["sixSegmentDoF"]["arclength"];
    std::cout << "Arclength: " << experimentGeometry.arclength << std::endl;

    const uint32_t numPhi1Vals = experimentConfig["experiment"]["sixSegmentDoF"]["numPhi1Vals"];
    const uint32_t numPhi2Vals = experimentConfig["experiment"]["sixSegmentDoF"]["numPhi2Vals"];
    const uint32_t numTheta1Vals = experimentConfig["experiment"]["sixSegmentDoF"]["numTheta1Vals"];
    const uint32_t numTheta2Vals = experimentConfig["experiment"]["sixSegmentDoF"]["numTheta2Vals"];
    const uint32_t numTheta3Vals = experimentConfig["experiment"]["sixSegmentDoF"]["numTheta3Vals"];

    auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();


    const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

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
    twisty::PathWeighting::BaseWeightLookupTable &weightingIntegralsRawPointer = (*lookupEvaluator);

    const twisty::PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer
          = (experimentParams.weightingParameters.weightingMethod
                  != twisty::WeightingMethod::RadiativeTransfer)
          ? 1.0
          : twisty::PathWeighting::NormalizerStuff::Norm(
                experimentParams.numSegmentsPerCurve, ds, experimentGeometry);
    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    double pathNormalizerLog10 = (double)boost::multiprecision::log10(pathNormalizer);

    // Ok, generate the curve.
    Farlor::Vector3 point0 = experimentGeometry.m_startPos;
    Farlor::Vector3 point1 = point0 + experimentGeometry.m_startDir * ds;

    Farlor::Vector3 point6 = experimentGeometry.m_endPos;
    Farlor::Vector3 point5 = point6 - experimentGeometry.m_endDir * ds;

    // Calculate the second point using theta and cos values.

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = twisty::TwistyPi;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    // Polar angle
    const float phi2Min = 0.0f;
    const float phi2Max = twisty::TwistyPi;
    const float dPhi2 = (phi2Max - phi2Min) / numPhi2Vals;

    // Azimuthal
    const float theta2Min = -twisty::TwistyPi;
    const float theta2Max = twisty::TwistyPi;
    const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

    // Azimuthal
    const float theta3Min = -twisty::TwistyPi;
    const float theta3Max = twisty::TwistyPi;
    const float dTheta3 = (theta3Max - theta3Min) / numTheta3Vals;


    std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;

    uint64_t numValidPaths = 0;
    twisty::CombinedWeightValues_C activeWeightValue;
    twisty::CombinedWeightValues_C_Reset(activeWeightValue);

    for (int phi1Idx = 0; phi1Idx < numPhi1Vals; phi1Idx++) {
        const float phi1 = phi1Min + phi1Idx * dPhi1;

        for (int theta1Idx = 0; theta1Idx < numTheta1Vals; theta1Idx++) {
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

            const float remainingDistance3_2 = (point5 - point2).SqrMagnitude();

            if ((9.0 * ds * ds) < remainingDistance3_2) {
                continue;
            }

            for (int phi2Idx = 0; phi2Idx < numPhi2Vals; phi2Idx++) {
                const float phi2 = phi2Min + phi2Idx * dPhi2;

                for (int theta2Idx = 0; theta2Idx < numTheta2Vals; theta2Idx++) {
                    const float theta2 = theta2Min + theta2Idx * dTheta2;

                    const float sinPhi2 = std::sin(phi2);
                    const float cosPhi2 = std::cos(phi2);
                    const float sinTheta2 = std::sin(theta2);
                    const float cosTheta2 = std::cos(theta2);

                    // Calculate the first segment position
                    const Farlor::Vector3 segment2Dir(
                          sinPhi2 * cosTheta2, sinPhi2 * sinTheta2, cosPhi2);
                    const Farlor::Vector3 point3 = point2 + segment2Dir * ds;

                    const float remainingDistance2_2 = (point5 - point3).SqrMagnitude();

                    if ((4.0 * ds * ds) < remainingDistance2_2) {
                        continue;
                    }


                    // If not, we keep going through the possible combinations
                    for (int theta3Idx = 0; theta3Idx < numTheta3Vals; theta3Idx++) {
                        const float theta3 = theta3Min + theta3Idx * dTheta3;

                        const Farlor::Vector3 x_p = (point3 + point5) * 0.5;
                        const Farlor::Vector3 lineUnitDir = (point5 - point3).Normalized();

                        Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
                        if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
                            otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
                        }

                        const Farlor::Vector3 normalToLine
                              = lineUnitDir.Cross(otherCrossVec).Normalized();

                        // We should have an even number of segments remaining
                        const float hypot = ds;
                        const float D_2 = (point5 - point3).Magnitude() * 0.5f;
                        assert(D_2 > hypot && "This should never be reached due to earlier check.");

                        const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
                        Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

                        // Now rotate randomly theta amount around the axis.
                        {
                            const float sinRotAngle = std::sinf(theta3 / 2.0f);
                            float quaternionRotation[4]
                                  = { std::cosf(theta3 / 2.0f), lineUnitDir.x * sinRotAngle,
                                        lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


                            Farlor::Vector3 shiftedPoint = x_t - point3;
                            // Rotate and stuff back in shifted point
                            twisty::RotateVectorByQuaternion(
                                  quaternionRotation, shiftedPoint.m_data.data());
                            // Update the point with the rotated version
                            x_t = shiftedPoint + point3;
                        }
                        const Farlor::Vector3 point4 = x_t;

                        std::array<Farlor::Vector3, RequiredNumSegments + 1> points
                              = { point0, point1, point2, point3, point4, point5, point6 };
                        std::array<Farlor::Vector3, RequiredNumSegments> tangents;
                        std::array<float, RequiredNumSegments - 1> curvatures;

                        twisty::PerturbUtils::UpdateTangentsFromPos(points.data(), tangents.data(),
                              RequiredNumSegments, experimentGeometry);
                        twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                              tangents.data(), curvatures.data(), RequiredNumSegments,
                              experimentGeometry);

                        double scatteringWeightLog10
                              = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                                    curvatures.data(), RequiredNumSegments - 1,
                                    weightingIntegralsRawPointer, experimentParams.weightingParameters.absorption);
                        scatteringWeightLog10 += pathNormalizerLog10;

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
        }
    }

    boost::multiprecision::cpp_dec_float_100 pathIntegralResult = 0.0;
    for (const auto &combinedWeightValue : combinedWeightValues) {
        pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
    }

    boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;
    if (numValidPaths > 0) {
        finalResult = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
    }

    const uint64_t numTotalDoF
          = numPhi1Vals * numPhi2Vals * numTheta1Vals * numTheta2Vals * numTheta3Vals;

    auto runExperimentTimeEnd = std::chrono::high_resolution_clock::now();

    std::cout << "Experiment Time Reporting: " << std::endl;
    auto runExperimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
          runExperimentTimeEnd - runExperimentTimeStart);

    resultsOFS << "Experiment time: " << format_duration(runExperimentTimeMs) << std::endl;
    resultsOFS << "Experiment time (ms): " << runExperimentTimeMs.count() << std::endl;
    resultsOFS << "Converged final weight combined weight: " << finalResult << std::endl;
    resultsOFS << "Ratio of valid paths to total: " << numValidPaths << " / " << numTotalDoF
               << " = " << ((double)numValidPaths / (double)numTotalDoF) << std::endl;

    std::cout << "Experiment time: " << format_duration(runExperimentTimeMs) << std::endl;
    std::cout << "Experiment time (ms): " << runExperimentTimeMs.count() << std::endl;
    std::cout << "Converged final weight: " << finalResult << std::endl;
    std::cout << "Ratio of valid paths to total: " << numValidPaths << " / " << numTotalDoF << " = "
              << ((double)numValidPaths / (double)numTotalDoF) << std::endl;

    std::cout << "Done" << std::endl;
}