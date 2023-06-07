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

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(
      const nlohmann::json &experimentConfig, bool tackOnDate = true)
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
    experimentParams.experimentDirPath += "/" + experimentParams.experimentName + "/";
    if (tackOnDate) {
        experimentParams.experimentDirPath += twisty::GetCurrentTimeForFileName() + "/";
    }
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
    if (argc < 4) {
        std::cout << "Call as: " << argv[0] << " configFilename arclength angle" << std::endl;
        return 1;
    }
    std::fstream configFile(argv[1]);
    if (!configFile.is_open()) {
        std::cout << "Failed to open: " << argv[1] << std::endl;
        return 1;
    }

    nlohmann::json experimentConfig;
    configFile >> experimentConfig;

    const float arclength = std::stof(argv[2]);
    const float angle = std::stof(argv[3]);
    const float angle_rad = angle * (PI / 180.0f);

    twisty::ExperimentRunner::ExperimentParameters experimentParams
          = ParseExperimentParamsFromConfig(experimentConfig, false);
    std::cout << "Number of segments: " << experimentParams.numSegmentsPerCurve << std::endl;

    // tack on the info for experiment path
    experimentParams.experimentDirPath += "/arclength_" + std::to_string((int)arclength);
    experimentParams.experimentDirPath += "/angle_" + std::to_string((int)angle) + "/";

    experimentParams.experimentName
          = "/arclength_" + std::to_string((int)arclength) + "_angle_" + std::to_string((int)angle);

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
        // Compute end dir from angle
        experimentGeometry.m_endDir
              = Farlor::Vector3(std::cos(angle_rad), std::sin(angle_rad), 0.0f);
    }
    // Force to a value
    experimentGeometry.arclength = arclength;
    std::cout << "Arclength: " << experimentGeometry.arclength << std::endl;

    const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

    const uint64_t numExperimentPaths = experimentParams.numPathsInExperiment;

    const float minDs = 9.0f / experimentParams.numSegmentsPerCurve;
    const float maxDs = 21.0f / experimentParams.numSegmentsPerCurve;
    const uint32_t numArclengths = 100;
    twisty::PathWeighting::CachedMultiArclengthWeightLookupTable cachedLookupTable(
          experimentParams.weightingParameters, minDs, maxDs, numArclengths);

    twisty::PathWeighting::BaseWeightLookupTable &weightingIntegralsRawPointer
          = *cachedLookupTable.GetWeightLookupTable(ds);

    const twisty::PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = 1.0f;
    //     = twisty::PathWeighting::NormalizerStuff::Norm(
    //           experimentParams.numSegmentsPerCurve, ds, experimentGeometry);
    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    const double pathNormalizerLog10 = (double)boost::multiprecision::log10(pathNormalizer);

    const twisty::ExperimentBase::Result result = twisty::ExperimentBase::MSegmentPathGenerationMC(
          numExperimentPaths, experimentParams.numSegmentsPerCurve, experimentGeometry,
          experimentParams, pathNormalizerLog10, weightingIntegralsRawPointer);

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