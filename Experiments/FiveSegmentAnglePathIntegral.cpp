#include "ExperimentBase.h"

#include <ExperimentRunner.h>
#include <PathWeighters.h>
#include "CombinedWeightUtils.h"

#include "MathConsts.h"
#include "boost/multiprecision/detail/default_ops.hpp"


#include <FMath/FMath.h>

#include <omp.h>

#include <nlohmann/json.hpp>
#include <string>

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
            std::cout << "Using simplified model weighting function" << std::endl;
            experimentParams.weightingParameters.weightingMethod
                  = twisty::WeightingMethod::SimplifiedModel;
        } break;

        // Default to the simplified model
        default: {
            std::cout << "Error: Unknown weighting function specified";
            exit(1);
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

    std::ofstream resultsOFS(experimentParams.experimentDirPath + "/Results.txt");
    if (!resultsOFS.is_open()) {
        std::cout << "Failed to open results file: " << experimentParams.experimentDirPath
                  << "/Results.txt" << std::endl;
        return 1;
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

    const uint32_t numPhi1Vals = experimentConfig["experiment"]["fiveSegmentDoF"]["numPhi1Vals"];
    const uint32_t numTheta1Vals
          = experimentConfig["experiment"]["fiveSegmentDoF"]["numTheta1Vals"];
    const uint32_t numTheta2Vals
          = experimentConfig["experiment"]["fiveSegmentDoF"]["numTheta2Vals"];

    const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

    std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;
    if (experimentParams.weightingParameters.weightingMethod
          == twisty::WeightingMethod::SimplifiedModel) {
        std::cout << "Simplified Weight Lookup Table" << std::endl;
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
          = twisty::PathWeighting::NormalizerStuff::Norm(
                experimentParams.numSegmentsPerCurve, ds, experimentGeometry);
    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    double pathNormalizerLog10 = (double)boost::multiprecision::log10(pathNormalizer);

    const twisty::ExperimentBase::Result result
          = twisty::ExperimentBase::FiveSegmentAngleIntegration(numPhi1Vals, numTheta1Vals,
                numTheta2Vals, experimentGeometry, experimentParams, pathNormalizerLog10,
                weightingIntegralsRawPointer);

    resultsOFS << "Num valid paths: " << result.numValidPaths << "/" << result.numPathsTotal
               << std::endl;
    resultsOFS << "Percent valid paths: "
               << (result.numValidPaths / (float)result.numPathsTotal) * 100.0f << "%" << std::endl;

    std::cout << "Num valid paths: " << result.numValidPaths << "/" << result.numPathsTotal
              << std::endl;
    std::cout << "Percent valid paths: "
              << (result.numValidPaths / (float)result.numPathsTotal) * 100.0f << "%" << std::endl;

    resultsOFS << "Converged final weight: " << result.totalWeight << std::endl;
    std::cout << "Converged final weight: " << result.totalWeight << std::endl;


    resultsOFS << "Min path weight log10: " << result.minPathWeightLog10 << std::endl;
    resultsOFS << "Max path weight log10: " << result.maxPathWeightLog10 << std::endl;

    std::cout << "Min path weight log10: " << result.minPathWeightLog10 << std::endl;
    std::cout << "Max path weight log10: " << result.maxPathWeightLog10 << std::endl;

    // Big float decompressed versions
    const double overallMinPathWeightLog10Decompressed
          = std::pow(10.0f, (double)result.minPathWeightLog10);
    const double overallMaxPathWeightLog10Decompressed
          = std::pow(10.0f, (double)result.maxPathWeightLog10);

    resultsOFS << "Min path weight: " << overallMinPathWeightLog10Decompressed << std::endl;
    resultsOFS << "Max path weight: " << overallMaxPathWeightLog10Decompressed << std::endl;

    std::cout << "Min path weight: " << overallMinPathWeightLog10Decompressed << std::endl;
    std::cout << "Max path weight: " << overallMaxPathWeightLog10Decompressed << std::endl;

    std::cout << "Calculating histogram" << std::endl;

    const int maxThreads = omp_get_max_threads();

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = 1.0f;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    // Azimuthal
    const float theta2Min = -twisty::TwistyPi;
    const float theta2Max = twisty::TwistyPi;
    const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

    const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
    const Farlor::Vector3 point1
          = experimentGeometry.m_startPos + experimentGeometry.m_startDir * ds;
    const Farlor::Vector3 point5 = experimentGeometry.m_endPos;
    const Farlor::Vector3 point4 = experimentGeometry.m_endPos - experimentGeometry.m_endDir * ds;

    // Histogram per thread
    const uint64_t numBins = 500;
    std::vector<std::vector<uint64_t>> histogramPerThread(maxThreads);
    for (int i = 0; i < maxThreads; i++) {
        histogramPerThread[i].reserve(numBins);
        for (int j = 0; j < numBins; j++) {
            histogramPerThread[i][j] = 0;
        }
    }

#pragma omp parallel for num_threads(maxThreads) default(none) shared(histogramPerThread, \
      numPhi1Vals, dPhi1, numTheta1Vals, dTheta1, numTheta2Vals, dTheta2,\
       ds, point0, point1, point4, point5, experimentGeometry, experimentParams, \
       weightingIntegralsRawPointer, pathNormalizerLog10, result)
    for (int phi1Idx = 0; phi1Idx < numPhi1Vals; phi1Idx++) {
        const int threadId = omp_get_thread_num();

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
            const Farlor::Vector3 segment1Dir
                  = Farlor::Vector3(sinPhi1 * cosTheta1, sinPhi1 * sinTheta1, cosPhi1);
            const Farlor::Vector3 point2 = point1 + segment1Dir * ds;

            const float remainingDistance2 = (point4 - point2).SqrMagnitude();

            if ((4 * ds * ds) < remainingDistance2) {
                continue;
            }

            // If not, we keep going through the possible combinations
            for (int theta2Idx = 0; theta2Idx < numTheta2Vals; theta2Idx++) {
                const float theta2 = theta2Min + theta2Idx * dTheta2;

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
                assert(D_2 < hypot && "This should never be reached due to earlier check.");

                const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
                Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

                // Now rotate randomly theta amount around the axis.
                {
                    const float sinRotAngle = std::sin(theta2 / 2.0f);
                    float quaternionRotation[4]
                          = { std::cos(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
                                lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


                    Farlor::Vector3 shiftedPoint = x_t - point2;
                    // Rotate and stuff back in shifted point
                    twisty::RotateVectorByQuaternion(
                          quaternionRotation, shiftedPoint.m_data.data());
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

                if (experimentParams.weightingParameters.weightingMethod
                      == twisty::WeightingMethod::RadiativeTransfer) {
                    twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                          tangents.data(), curvatures.data(), 5, experimentGeometry);
                } else {
                    twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
                          tangents.data(), curvatures.data(), 5, experimentGeometry);
                }

                const double scatteringWeightLog10
                      = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                              curvatures.data(), 4, weightingIntegralsRawPointer, experimentParams.weightingParameters.absorption)
                      + pathNormalizerLog10;
                // Decompressed weight big float
                const double scatteringWeightLog10Decompressed
                      = std::pow(10.0, scatteringWeightLog10);


                const uint64_t binIdx = (scatteringWeightLog10 - result.minPathWeightLog10)
                      / (result.maxPathWeightLog10 - result.minPathWeightLog10) * numBins;
                histogramPerThread[threadId][binIdx]++;
            }
        }
    }

    // Combine bins
    std::vector<uint64_t> histogram(numBins);
    for (int i = 0; i < maxThreads; i++) {
        for (int j = 0; j < numBins; j++) {
            histogram[j] += histogramPerThread[i][j];
        }
    }

    // Print histogram to file with bucket range and count
    resultsOFS << "Histogram" << std::endl;
    for (int i = 0; i < numBins; i++) {
        // Big float min
        const double binMin = overallMinPathWeightLog10Decompressed
              + (overallMaxPathWeightLog10Decompressed - overallMinPathWeightLog10Decompressed) * i
                    / numBins;
        // Big float max
        const double binMax = overallMinPathWeightLog10Decompressed
              + (overallMaxPathWeightLog10Decompressed - overallMinPathWeightLog10Decompressed)
                    * (i + 1) / numBins;

        resultsOFS << binMin << " " << binMax << " " << histogram[i] << std::endl;
    }

    std::cout << "Done" << std::endl;
}