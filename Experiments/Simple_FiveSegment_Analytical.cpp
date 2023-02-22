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
#include <random>

const float PI = 3.14159265358979323846f;

double G_4(const Farlor::Vector3 &q, const Farlor::Vector3 &betaHat, const double alpha,
      const double ds, const twisty::PerturbUtils::BoundaryConditions &bc)
{
    const float qSqrMag = q.SqrMagnitude();
    const float qMag = sqrt(qSqrMag);

    // Handles the heaviside step function
    if ((2.0 - qMag) <= 0.0) {
        return 0.0;
    }

    double firstTerm = (2.0 * PI) / (ds * ds * ds);
    firstTerm *= (1.0 / qMag);
    firstTerm *= std::exp(-4.0 * alpha);

    double secondTermArg = (alpha / 2.0) * (qSqrMag + q.Dot(bc.m_startDir + betaHat));
    const double secondTerm = std::exp(secondTermArg);

    const double besilFuncArg
          = alpha * sqrt(1 - (qSqrMag / 4)) * (betaHat - bc.m_startDir).Magnitude();
    const double besilEval = std::cyl_bessel_i(0, besilFuncArg);

    const double g4 = firstTerm * secondTerm * besilEval;
    return g4;
}

double G_5(const Farlor::Vector3 &q, const Farlor::Vector3 &betaHat, const double alpha,
      const double ds, const twisty::PerturbUtils::BoundaryConditions &bc)
{
    // Integration over unit sphere
    const int numPhi1Vals = 100000;
    const int numTheta1Vals = 100000;

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = 1.0f;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    double g5 = 0.0;

    for (int phi1Idx = 0; phi1Idx < numPhi1Vals; phi1Idx++) {
        const float phi1Mapped = phi1Min + phi1Idx * dPhi1;
        const float phi1 = std::acos(1.0 - 2.0 * phi1Mapped);

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
            const Farlor::Vector3 betaPrime
                  = Farlor::Vector3(sinPhi1 * cosTheta1, sinPhi1 * sinTheta1, cosPhi1);

            const Farlor::Vector3 qPrime = q - betaPrime;
            const double g4Eval = G_4(qPrime, betaPrime, alpha, ds, bc);
            const double segmentWeight = std::exp(alpha * betaHat.Dot(betaPrime));

            g5 += (g4Eval * segmentWeight * dPhi1 * dTheta1);
        }
    }

    g5 *= std::exp(-alpha);

    return g5;
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
        float x = experimentConfig["experiment"]["basicExperiment"]["startPos"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["startPos"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["startPos"][2];
        experimentGeometry.m_startPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["basicExperiment"]["startDir"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["startDir"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["startDir"][2];
        experimentGeometry.m_startDir = Farlor::Vector3(x, y, z).Normalized();
    }

    {
        float x = experimentConfig["experiment"]["basicExperiment"]["endPos"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["endPos"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["endPos"][2];
        experimentGeometry.m_endPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["basicExperiment"]["endDir"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["endDir"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["endDir"][2];
        experimentGeometry.m_endDir = Farlor::Vector3(x, y, z).Normalized();
    }

    experimentGeometry.arclength = experimentParams.arclength
          = experimentConfig["experiment"]["basicExperiment"]["arclength"];
    std::cout << "Arclength: " << experimentGeometry.arclength << std::endl;

    const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

    // To calculate the analytical derivative, we need to know a few things
    if (experimentParams.weightingParameters.scatter * experimentParams.weightingParameters.mu * ds
          == 0.0f) {
        std::cout
              << "Something is wrong and we cannot compute alpha. Check your weighing parameters"
              << std::endl;
        return 1;
    }

    const double alpha = 1.0
          / (experimentParams.weightingParameters.scatter * experimentParams.weightingParameters.mu
                * ds);
    const Farlor::Vector3 qVec
          = (experimentGeometry.m_endPos - experimentGeometry.m_startPos) / (ds)
          - (experimentGeometry.m_endDir + experimentGeometry.m_startDir);
    const Farlor::Vector3 &betaHat = experimentGeometry.m_endDir;
    twisty::ExperimentBase::Result result { 0 };
    result.totalWeight = G_5(qVec, betaHat, alpha, ds, experimentGeometry);
    result.numPathsTotal = 1;
    result.numValidPaths = 1;

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

    std::cout << "Done" << std::endl;
}