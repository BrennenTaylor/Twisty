#include <ExperimentRunner.h>

#include "MathConsts.h"

#include <FMath/FMath.h>

#include <nlohmann/json.hpp>

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

    //     if (experimentParams.bootstrapSeed == 0) {
    //         experimentParams.bootstrapSeed = time(0);
    //     }
    //     if (experimentParams.curvePurturbSeed == 0) {
    //         experimentParams.curvePurturbSeed = time(0);
    //     }

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
    assert((experimentParams.numSegmentsPerCurve == 5)
          && "Must only target 5 segment curve configurations");

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
        float x = experimentConfig["experiment"]["fiveSegmentDoF"]["startPos"][0];
        float y = experimentConfig["experiment"]["fiveSegmentDoF"]["startPos"][1];
        float z = experimentConfig["experiment"]["fiveSegmentDoF"]["startPos"][2];
        experimentGeometry.m_startPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["fiveSegmentDoF"]["startDir"][0];
        float y = experimentConfig["experiment"]["fiveSegmentDoF"]["startDir"][1];
        float z = experimentConfig["experiment"]["fiveSegmentDoF"]["startDir"][2];
        experimentGeometry.m_startDir = Farlor::Vector3(x, y, z).Normalized();
    }

    {
        float x = experimentConfig["experiment"]["fiveSegmentDoF"]["endPos"][0];
        float y = experimentConfig["experiment"]["fiveSegmentDoF"]["endPos"][1];
        float z = experimentConfig["experiment"]["fiveSegmentDoF"]["endPos"][2];
        experimentGeometry.m_endPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["fiveSegmentDoF"]["endDir"][0];
        float y = experimentConfig["experiment"]["fiveSegmentDoF"]["endDir"][1];
        float z = experimentConfig["experiment"]["fiveSegmentDoF"]["endDir"][2];
        experimentGeometry.m_endDir = Farlor::Vector3(x, y, z).Normalized();
    }
    // Force to a value
    experimentGeometry.arclength = experimentParams.arclength
          = experimentConfig["experiment"]["fiveSegmentDoF"]["arclength"];
    std::cout << "Arclength: " << experimentGeometry.arclength << std::endl;

    const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;
    // Ok, generate the curve.
    Farlor::Vector3 point0 = experimentGeometry.m_startPos;
    Farlor::Vector3 point1 = point0 + experimentGeometry.m_startDir * ds;

    Farlor::Vector3 point5 = experimentGeometry.m_endPos;
    Farlor::Vector3 point4 = point5 - experimentGeometry.m_endDir * ds;

    // Calculate the second point using theta and cos values.

    const float phi1Min = 0.0f;
    const float phi1Max = twisty::TwistyPi;
    const uint32_t numPhi1Vals = 10;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    const float theta1Min = 0.0f;
    const float theta1Max = 2.0f * twisty::TwistyPi;
    const uint32_t numTheta1Vals = 10;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    const float theta2Min = 0.0f;
    const float theta2Max = 2.0f * twisty::TwistyPi;
    const uint32_t numTheta2Vals = 10;
    const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

    std::vector<std::vector<Farlor::Vector3>> gridOfImages(numPhi1Vals);

    for (auto &image : gridOfImages) {
        image.resize(numTheta1Vals * numTheta2Vals);
    }

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

            const float remainingDistance2 = (point4 - point2).SqrMagnitude();

            if ((4 * ds * ds) < remainingDistance2) {
                continue;
            }

            // If not, we keep going through the possible combinations
            for (int theta2Idx = 0; theta2Idx < numTheta2Vals; theta2Idx++) {
                const float theta2 = theta2Min + theta2Idx * dTheta2;
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
                    float quaternionRotation[4]
                          = { std::cosf(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
                                lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


                    Farlor::Vector3 shiftedPoint = x_t - point2;
                    // Rotate and stuff back in shifted point
                    twisty::RotateVectorByQuaternion(
                          quaternionRotation, shiftedPoint.m_data.data());
                    // Update the point with the rotated version
                    x_t = shiftedPoint + point2;
                }
                const Farlor::Vector3 point3 = x_t;
            }
        }
    }
}