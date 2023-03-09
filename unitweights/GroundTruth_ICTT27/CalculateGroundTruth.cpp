#include <stdexcept>
#include <exception>
#include <iomanip>
#define _USE_MATH_DEFINES
#include "ExperimentRunner.h"

#include <FMath/FMath.h>

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(
      const nlohmann::json &configJson)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Values loaded from the config file
    experimentParams.numPathsInExperiment
          = configJson["experiment"]["experimentParams"]["pathsToGenerate"];


    experimentParams.numPathsToSkip = configJson["experiment"]["experimentParams"]["pathsToSkip"];
    experimentParams.experimentName = configJson["experiment"]["experimentParams"]["name"];
    experimentParams.experimentDirPath
          = configJson["experiment"]["experimentParams"]["experimentDir"];

    experimentParams.experimentDirPath += "/" + experimentParams.experimentName;

    experimentParams.numSegmentsPerCurve
          = configJson["experiment"]["experimentParams"]["numSegments"];
    experimentParams.arclength = configJson["experiment"]["experimentParams"]["arclength"];

    // Seeds
    experimentParams.bootstrapSeed
          = configJson["experiment"]["experimentParams"]["random"]["bootstrapSeed"];
    experimentParams.curvePurturbSeed
          = configJson["experiment"]["experimentParams"]["random"]["perturbSeed"];

    if (experimentParams.bootstrapSeed == 0) {
        experimentParams.bootstrapSeed = time(0);
    }
    if (experimentParams.curvePurturbSeed == 0) {
        experimentParams.curvePurturbSeed = time(0);
    }

    // Weighting parameter stuff
    experimentParams.weightingParameters.mu
          = configJson["experiment"]["experimentParams"]["weighting"]["mu"];
    experimentParams.weightingParameters.eps
          = configJson["experiment"]["experimentParams"]["weighting"]["eps"];
    experimentParams.weightingParameters.numStepsInt
          = (int)configJson["experiment"]["experimentParams"]["weighting"]["numStepsInt"];
    experimentParams.weightingParameters.numCurvatureSteps
          = (int)configJson["experiment"]["experimentParams"]["weighting"]["numCurvatureSteps"];
    experimentParams.weightingParameters.absorption
          = configJson["experiment"]["experimentParams"]["weighting"]["absorption"];
    experimentParams.weightingParameters.scatter
          = configJson["experiment"]["experimentParams"]["weighting"]["scatter"];

    // TODO: Should these be configurable in the file?
    experimentParams.weightingParameters.minBound = 0.0;
    experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;

    return experimentParams;
}

long double HeavisideStepFunction(const long double x)
{
    if (x > 0) {
        return 1.0;
    }

    return 0.0;
}

double G_3(const Farlor::Vector3 &qVec, const Farlor::Vector3 &bVec, double ds, double alpha,
      const Farlor::Vector3 &n0)
{
    const double q = qVec.Magnitude();

    // First terms
    const double firstTermEval
          = (2.0 * M_PI) / (ds * ds * ds) * HeavisideStepFunction(2.0 - q) / q * exp(-4.0 * alpha);

    // Second term
    const double dotProductEval = qVec.Dot(n0 + bVec);
    const double secondTermEval = exp((alpha / 2.0) * (q * q + dotProductEval));

    // Bessel evaluation piece
    const double magVectorPiece = (bVec - qVec).Magnitude();
    double besselOrder = 0.0;
    const double besselEval
          = std::cyl_bessel_i(besselOrder, alpha * sqrt(1.0 - (q * q / 4.0)) * magVectorPiece);

    double finalTerm = firstTermEval * secondTermEval * besselEval;

    if (isnan(finalTerm)) {
        if (firstTermEval == 0.0 && isnan(besselEval)) {
            finalTerm = 0.0;
        } else {
            std::cout << "Weird case" << std::endl;
            std::cout << "First term: " << firstTermEval << std::endl;
            std::cout << "Bessel term: " << besselEval << std::endl;
        }
    }
    return finalTerm;
}

// Spherical coordinates used for integeration
// https://tutorial.math.lamar.edu/classes/calcii/sphericalcoords.aspx
double G_4(const Farlor::Vector3 &q, const Farlor::Vector3 &beta, double ds, double alpha,
      const Farlor::Vector3 &n0)
{
    const uint32_t numThetaSteps = 100;
    const double thetaMin = 0.0;
    const double thetaMax = 2.0 * M_PI;
    const double thetaStepSize = (thetaMax - thetaMin) / (numThetaSteps - 1);

    const double dTheta = ((thetaMax - thetaMin) / numThetaSteps);

    const uint32_t numSSteps = 100;
    const double sMin = -1.0;
    const double sMax = 1.0;
    const double sStepSize = (sMax - sMin) / (numSSteps - 1);

    const double dS = ((sMax - sMin) / numSSteps);

    double result = 0.0;

    for (uint32_t thetaIdx = 0; thetaIdx < numThetaSteps; thetaIdx++) {
        const double theta = thetaMin + thetaIdx * thetaStepSize;
        for (uint32_t sIdx = 0; sIdx < numSSteps; sIdx++) {
            const double s = sMin + sIdx * sStepSize;
            const double ss = sqrt(1.0 - s * s);

            Farlor::Vector3 evalVector
                  = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();
            // Function eval
            double functionEval = G_3(q - evalVector, evalVector, ds, alpha, n0);
            result += exp(-alpha) * functionEval * dTheta * dS * exp(alpha * beta.Dot(evalVector));
        }
    }
    return result;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
        return 1;
    }

    // try {
    std::string configFilename(argv[1]);

    std::ifstream configFile(configFilename);
    if (!configFile.is_open()) {
        std::string error = "Failed to open file ";
        error += configFilename;
        throw std::runtime_error(error);
    }

    nlohmann::json configJson;
    configFile >> configJson;

    twisty::ExperimentRunner::ExperimentParameters experimentParams
          = ParseExperimentParamsFromConfig(configJson);
    if (!std::filesystem::exists(experimentParams.experimentDirPath)) {
        std::filesystem::create_directories(experimentParams.experimentDirPath);
    }
    const std::string experimentCfgCopyFilename = std::string(experimentParams.experimentDirPath)
          + "/" + experimentParams.experimentName + ".cfg";

    if (!std::filesystem::exists(experimentCfgCopyFilename)) {
        std::filesystem::copy_file(configFilename, experimentCfgCopyFilename,
              std::filesystem::copy_options::overwrite_existing);
    }
    const uint32_t numEmitterDirections
          = configJson["experiment"]["smallSegmentExperiment"]["numEmitterDirections"];
    const float distanceFromPlane
          = configJson["experiment"]["smallSegmentExperiment"]["distanceFromPlane"];

    const double ds = experimentParams.arclength / experimentParams.numSegmentsPerCurve;
    const double alpha = 1.0
          / (experimentParams.weightingParameters.scatter * experimentParams.weightingParameters.mu
                * ds);

    Farlor::Vector3 startPos(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 startDir(0.0f, 0.0f, 1.0f);

    Farlor::Vector3 endPos(0.0f, 0.0f, distanceFromPlane);
    Farlor::Vector3 endDir(0.0f, 0.0f, 1.0f);

    const Farlor::Vector3 q = (endPos - startPos) * (1.0 / ds) - (startDir + endDir);

    const uint32_t numSSteps = numEmitterDirections;
    const double sMin = -1.0;
    const double sMax = 1.0;
    const double sStepSize = (sMax - sMin) / (numSSteps - 1);

    std::string outputDataFilename
          = std::string(experimentParams.experimentDirPath) + std::string("/Results.dat");
    std::ofstream ofs(outputDataFilename);

    const double theta = 0.0;
    for (uint32_t sIdx = 0; sIdx < numSSteps; sIdx++) {
        const double s = sMin + sIdx * sStepSize;
        const double ss = sqrt(1.0 - s * s);

        std::cout << "S: " << s << std::endl;

        Farlor::Vector3 evalVector
              = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();

        double g4Result = G_4(q, evalVector, ds, alpha, startDir);

        ofs << s << ", " << g4Result << std::endl;

        std::cout << "\tEnd Dir: " << evalVector << std::endl;
        std::cout << "\tEval: " << g4Result << std::endl;
    }
    ofs.close();
    // }
    // catch (std::exception &ex)
    // {
    //     std::cout << ex.what() << std::endl;
    //     return 1;
    // }

    return 0;
}