#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <map>
#include <filesystem>

#include <libconfig.h++>

#include <FMath/FMath.h>

#include "ExperimentRunner.h"

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(const libconfig::Config& config)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Hardcoded values
    experimentParams.maximumBootstrapCurveError = 0.5f;
    experimentParams.rotateInitialSeedCurveRadians = 0.0f;

    // Values loaded from the config file
    experimentParams.numPathsInExperiment = (int)config.lookup("experiment.experimentParams.pathsToGenerate");
    experimentParams.numPathsToSkip = (int)config.lookup("experiment.experimentParams.pathsToSkip");
    experimentParams.experimentName = config.lookup("experiment.experimentParams.name").c_str();
    experimentParams.experimentDirPath = config.lookup("experiment.experimentParams.experimentDir").c_str();

    experimentParams.experimentDirPath += "/" + experimentParams.experimentName;

    experimentParams.numSegmentsPerCurve = (int)config.lookup("experiment.experimentParams.numSegments");
    experimentParams.arclength = config.lookup("experiment.experimentParams.arclength");

    // Seeds
    experimentParams.bootstrapSeed = (int)config.lookup("experiment.experimentParams.random.bootstrapSeed");
    experimentParams.curvePurturbSeed = (int)config.lookup("experiment.experimentParams.random.perturbSeed");

    if (experimentParams.bootstrapSeed == 0)
    {
        experimentParams.bootstrapSeed = time(0);
    }
    if (experimentParams.curvePurturbSeed == 0)
    {
        experimentParams.curvePurturbSeed = time(0);
    }

    // Weighting parameter stuff
    experimentParams.weightingParameters.mu = config.lookup("experiment.experimentParams.weighting.mu");
    experimentParams.weightingParameters.eps = config.lookup("experiment.experimentParams.weighting.eps");
    experimentParams.weightingParameters.numStepsInt = (int)config.lookup("experiment.experimentParams.weighting.numStepsInt");
    experimentParams.weightingParameters.numCurvatureSteps = (int)config.lookup("experiment.experimentParams.weighting.numCurvatureSteps");
    experimentParams.weightingParameters.absorbtion = config.lookup("experiment.experimentParams.weighting.absorbtion");
    experimentParams.weightingParameters.scatter = config.lookup("experiment.experimentParams.weighting.scatter");

    // TODO: Should these be configurable in the file?
    experimentParams.weightingParameters.minBound = 0.0;
    experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;

    return experimentParams;
}

bool LoadConfigFile(const std::string& filename, libconfig::Config& experimentConfig)
{
    try
    {
        experimentConfig.readFile(filename);
    }
    catch (const libconfig::FileIOException &fioex)
    {
        std::cout << "I/O error while reading file." << std::endl;
        return false;
    }
    catch (const libconfig::ParseException &pex)
    {
        std::cout << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                    << " - " << pex.getError() << std::endl;
        return false;
    }
    return true;
}

long double HeavisideStepFunction(const long double x)
{
   if (x > 0)
   {
      return 1.0;
   }

   return 0.0;
}

double G_3(const Farlor::Vector3& qVec, const Farlor::Vector3& bVec, double ds, double alpha, const Farlor::Vector3& n0)
{
   const double q = qVec.Magnitude();

   // First terms
   const double firstTermEval = (2.0 * M_PI) / (ds * ds * ds) * HeavisideStepFunction(2.0 - q) / q * exp(-4.0 * alpha);

   // Second term
   const double dotProductEval = qVec.Dot(n0 + bVec);
   const double secondTermEval = exp((alpha / 2.0) * (q * q + dotProductEval));

   // Bessel evaluation piece
   const double magVectorPiece = (bVec - qVec).Magnitude();
   double besselOrder = 0.0;
   const double besselEval = std::cyl_bessel_i(besselOrder,
      alpha * sqrt(1.0 - (q * q / 4.0)) * magVectorPiece
   );

   double finalTerm = firstTermEval * secondTermEval * besselEval;

   if (isnan(finalTerm))
   {
      if (firstTermEval == 0.0 && isnan(besselEval))
      {
         finalTerm = 0.0;
      }
      else
      {
         std::cout << "Weird case" << std::endl;
         std::cout << "First term: " << firstTermEval << std::endl;
         std::cout << "Bessel term: " << besselEval << std::endl;
      }
   }
   return finalTerm;
}

// Spherical coordinates used for integeration
// https://tutorial.math.lamar.edu/classes/calcii/sphericalcoords.aspx
double G_4(const Farlor::Vector3& q, const Farlor::Vector3& beta, double ds, double alpha, const Farlor::Vector3& n0)
{
    const uint32_t numThetaSteps = 1000;
    const double thetaMin = 0.0;
    const double thetaMax = 2.0 * M_PI;
    const double thetaStepSize = (thetaMax - thetaMin) / (numThetaSteps - 1);

    const double dTheta = ((thetaMax - thetaMin) / numThetaSteps);

    const uint32_t numSSteps = 1000;
    const double sMin = -1.0;
    const double sMax= 1.0;
    const double sStepSize= (sMax - sMin) / (numSSteps - 1);

    const double dS = ((sMax - sMin) / numSSteps);

    double result = 0.0;

    for (uint32_t thetaIdx = 0; thetaIdx < numThetaSteps; thetaIdx++)
    {
        const double theta = thetaMin + thetaIdx * thetaStepSize;
        for (uint32_t sIdx = 0; sIdx < numSSteps; sIdx++)
        {
            const double s = sMin+ sIdx * sStepSize;
            const double ss = sqrt(1.0 - s * s);

            Farlor::Vector3 evalVector = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();
            // Function eval 
            double functionEval = G_3(q - evalVector, evalVector, ds, alpha, n0);
            result += exp(-alpha) * functionEval * dTheta * dS * exp(alpha * beta.Dot(evalVector));
        }
    }
    return result;
}

int main(int argc, char* argv[])
{
   if (argc < 2)
   {
      std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
      return 1;
   }
   std::string configFilename(argv[1]);
   libconfig::Config experimentConfig;
   if (!LoadConfigFile(configFilename, experimentConfig)) {
      std::cout << "Failed to load config file: " << configFilename << std::endl;
      return false;
   }

   twisty::ExperimentRunner::ExperimentParameters experimentParams = ParseExperimentParamsFromConfig(experimentConfig);
   if (!std::filesystem::exists(experimentParams.experimentDirPath))
   {
      std::filesystem::create_directories(experimentParams.experimentDirPath);
   }
   const std::string experimentCfgCopyFilename = std::string(experimentParams.experimentDirPath) + "/" + experimentParams.experimentName + ".cfg";

   if (!std::filesystem::exists(experimentCfgCopyFilename)) {
       std::filesystem::copy_file(configFilename, experimentCfgCopyFilename, std::filesystem::copy_options::overwrite_existing);
   }
   const uint32_t numEmitterDirections = (int)experimentConfig.lookup("experiment.smallSegmentExperiment.numEmitterDirections");
   const float distanceFromPlane = experimentConfig.lookup("experiment.smallSegmentExperiment.distanceFromPlane");

   const double ds = experimentParams.arclength / experimentParams.numSegmentsPerCurve;
   const double alpha = 1.0 / (experimentParams.weightingParameters.scatter * experimentParams.weightingParameters.mu * ds);

   Farlor::Vector3 startPos(0.0f, 0.0f, 0.0f);
   Farlor::Vector3 startDir(0.0f, 0.0f, 1.0f);

   Farlor::Vector3 endPos(0.0f, 0.0f, distanceFromPlane);
   Farlor::Vector3 endDir(0.0f, 0.0f, 1.0f);

   const Farlor::Vector3 q = (endPos - startPos) * (1.0 / ds) - (startDir + endDir);

   const uint32_t numSSteps = numEmitterDirections;
   const double sMin = -1.0;
   const double sMax = 1.0;
   const double sStepSize = (sMax - sMin) / (numSSteps - 1);

   std::string outputDataFilename = std::string(experimentParams.experimentDirPath) + std::string("/Results.dat");
   std::ofstream ofs(outputDataFilename);

   const double theta = 0.0;
   for (uint32_t sIdx = 0; sIdx < numSSteps; sIdx++)
   {
      const double s = sMin + sIdx * sStepSize;
      const double ss = sqrt(1.0 - s * s);

      Farlor::Vector3 evalVector = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();

      double g4Result = G_4(q, evalVector, ds, alpha, startDir);

      ofs << s << ", " << g4Result << std::endl;

      std::cout << "S: " << s << std::endl;
      std::cout << "\tEnd Dir: " << evalVector << std::endl;
      std::cout << "\tEval: " << g4Result << std::endl;
   }
   ofs.close();

   return 0;
}