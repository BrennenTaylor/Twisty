#include <ExperimentRunner.h>
#include <PathWeighters.h>

#include "MathConsts.h"

#include <FMath/FMath.h>

#include <nlohmann/json.hpp>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

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
    experimentParams.weightingParameters.absorbtion
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["absorbtion"];
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
    constexpr uint32_t RequiredNumSegments = 6;
    assert((experimentParams.numSegmentsPerCurve == RequiredNumSegments)
          && "Must only target 6 segment curve configurations");

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
    const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

    const uint64_t numGreatestPathsToExtract
          = experimentConfig["experiment"]["sixSegmentDoF"]["numGreatestPathsToExtract"];
    std::cout << "NumGreatestPathsToExtract: " << numGreatestPathsToExtract << std::endl;

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
    const double pathNormaizerLog10
          = static_cast<double>(boost::multiprecision::log10(pathNormalizer));

    // Ok, generate the curve.
    // We have 7 points to work with
    Farlor::Vector3 point0 = experimentGeometry.m_startPos;
    Farlor::Vector3 point1 = point0 + experimentGeometry.m_startDir * ds;

    Farlor::Vector3 point6 = experimentGeometry.m_endPos;
    Farlor::Vector3 point5 = point6 - experimentGeometry.m_endDir * ds;

    // Calculate the second point using theta and cos values.

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = twisty::TwistyPi;
    const uint32_t numPhi1Vals = 50;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const uint32_t numTheta1Vals = 50;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    // Polar angle
    const float phi2Min = 0.0f;
    const float phi2Max = twisty::TwistyPi;
    const uint32_t numPhi2Vals = 50;
    const float dPhi2 = (phi2Max - phi2Min) / numPhi2Vals;

    // Azimuthal
    const float theta2Min = -twisty::TwistyPi;
    const float theta2Max = twisty::TwistyPi;
    const uint32_t numTheta2Vals = 50;
    const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

    // Azimuthal
    const float theta3Min = -twisty::TwistyPi;
    const float theta3Max = twisty::TwistyPi;
    const uint32_t numTheta3Vals = 50;
    const float dTheta3 = (theta3Max - theta3Min) / numTheta3Vals;

    struct Pixel {
        bool validPath = false;
        double weight = 0.0f;
    };
    std::vector<std::vector<std::vector<std::vector<std::vector<Pixel>>>>> values(numPhi1Vals);

    for (auto &vector1 : values) {
        vector1.resize(numTheta2Vals);
        for (auto &vector2 : vector1) {
            vector2.resize(numPhi1Vals);
            for (auto &vector3 : vector2) {
                vector3.resize(numTheta2Vals);
                for (auto &vector4 : vector3) {
                    vector4.resize(numTheta3Vals);
                }
            }
        }
    }

    double minVal = 100000000;
    double maxValue = -10000000;

    struct SortStruct {
        double value = -10000000;
        Farlor::Vector3 point0;
        Farlor::Vector3 point1;
        Farlor::Vector3 point2;
        Farlor::Vector3 point3;
        Farlor::Vector3 point4;
        Farlor::Vector3 point5;
        Farlor::Vector3 point6;
    };
    std::vector<SortStruct> unsortedPaths;

    boost::multiprecision::cpp_dec_float_100 pathIntegralResult = 0.0;
    uint64_t numValidPaths = 0;

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
                                    weightingIntegralsRawPointer);
                        double normalizedPathWeight
                              = std::pow(10.0, scatteringWeightLog10 + pathNormaizerLog10);

                        if (normalizedPathWeight > maxValue) {
                            maxValue = normalizedPathWeight;
                        }

                        if (normalizedPathWeight < minVal) {
                            minVal = normalizedPathWeight;
                        }

                        SortStruct newValue;
                        newValue.value = normalizedPathWeight;
                        newValue.point0 = point0;
                        newValue.point1 = point1;
                        newValue.point2 = point2;
                        newValue.point3 = point3;
                        newValue.point4 = point4;
                        newValue.point5 = point5;
                        newValue.point6 = point6;
                        unsortedPaths.push_back(newValue);

                        values[phi1Idx][theta1Idx][phi2Idx][theta2Idx][theta3Idx].validPath = true;
                        values[phi1Idx][theta1Idx][phi2Idx][theta2Idx][theta3Idx].weight
                              = normalizedPathWeight;
                        pathIntegralResult += normalizedPathWeight;
                        numValidPaths++;
                    }
                }
            }
        }
    }

    std::cout << "Converged final weight: " << (pathIntegralResult / numValidPaths) << std::endl;
    {
        const std::string resultsFilepath
              = std::string(experimentParams.experimentDirPath) + "/Results.dat";
        std::ofstream resultsOFS(resultsFilepath);
        resultsOFS << "Converged final weight: " << (pathIntegralResult / numValidPaths)
                   << std::endl;
    }

    std::filesystem::path generatedCurvesDirPath = experimentParams.experimentDirPath;
    generatedCurvesDirPath /= "GeneratedCurves";
    if (!std::filesystem::exists(generatedCurvesDirPath)) {
        std::filesystem::create_directories(generatedCurvesDirPath);
    }

    // First, we need to write the boundary conditions here
    // Export geometry
    {
        std::filesystem::path outputDirectoryPath
              = std::filesystem::path(generatedCurvesDirPath) / "BoundaryConditions.bcf";

        std::ofstream boundaryConditionFile(outputDirectoryPath.string(), std::ios::binary);
        if (!boundaryConditionFile.is_open()) {
            std::cout << "Failed to open " << outputDirectoryPath.string() << std::endl;
            return false;
        }

        boundaryConditionFile.write(
              (char *)experimentGeometry.m_startPos.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write(
              (char *)experimentGeometry.m_startDir.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write(
              (char *)experimentGeometry.m_endPos.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write(
              (char *)experimentGeometry.m_endDir.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write((char *)&experimentGeometry.arclength, sizeof(float));
        boundaryConditionFile.write(
              (char *)&experimentParams.numSegmentsPerCurve, sizeof(uint32_t));
    }

    {
        std::stringstream indexJsonSS;
        indexJsonSS << "index.json";

        nlohmann::json experimentJson;
        experimentJson["experimentName"] = "SixSegmentExploreDoF";

        std::filesystem::path indexJsonPath = generatedCurvesDirPath;
        indexJsonPath.append(indexJsonSS.str());


        std::ofstream jsonOfstream(indexJsonPath.string());
        jsonOfstream << std::setw(4) << experimentJson << std::endl;
        jsonOfstream.close();
    }

    std::stringstream pathBinaryFilenameSS;
    pathBinaryFilenameSS << experimentParams.pathBatchPrepend;
    pathBinaryFilenameSS << "Paths_Binary"
                         << ".pbd";
    std::filesystem::path binaryFilePath = generatedCurvesDirPath;
    binaryFilePath.append(pathBinaryFilenameSS.str());


    std::sort(unsortedPaths.begin(), unsortedPaths.end(),
          [](const SortStruct &l, const SortStruct &r) { return l.value > r.value; });
    std::cout << "Highest weight: " << unsortedPaths.front().value << std::endl;
    std::cout << "Lowest weight: " << unsortedPaths.back().value << std::endl;
    std::vector<Farlor::Vector3> sortedPaths(numGreatestPathsToExtract * (RequiredNumSegments + 1));

    int idx = 0;
    for (int i = 0; i < numGreatestPathsToExtract; i++) {
        sortedPaths[idx + 0] = unsortedPaths[i].point0;
        sortedPaths[idx + 1] = unsortedPaths[i].point1;
        sortedPaths[idx + 2] = unsortedPaths[i].point2;
        sortedPaths[idx + 3] = unsortedPaths[i].point3;
        sortedPaths[idx + 4] = unsortedPaths[i].point4;
        sortedPaths[idx + 5] = unsortedPaths[i].point5;
        sortedPaths[idx + 6] = unsortedPaths[i].point6;
        idx += 7;
    }

    std::cout << "Writing num paths: " << unsortedPaths.size() << std::endl;
    std::ofstream curvesBinaryFile(binaryFilePath, std::ios::binary);
    curvesBinaryFile.write(
          (char *)sortedPaths.data(), sizeof(Farlor::Vector3) * sortedPaths.size());

    std::cout << "Done" << std::endl;
}