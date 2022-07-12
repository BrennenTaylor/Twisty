/*

    This experiment executes the monte carlo feynman path integral with a specified number of segments.
    The target number of segments should be small as the specified weight function for this corresponds to the
    simplified weighting function derived by Jerry.

*/

#include "FullExperimentRunnerOptimalPerturb.h"

#include <libconfig.h++>

#include "Bootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(
      const libconfig::Config &config)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Values loaded from the config file
    experimentParams.numPathsInExperiment
          = (int)config.lookup("experiment.experimentParams.pathsToGenerate");
    experimentParams.numPathsToSkip = (int)config.lookup("experiment.experimentParams.pathsToSkip");
    experimentParams.experimentName = config.lookup("experiment.experimentParams.name").c_str();
    experimentParams.experimentDirPath
          = config.lookup("experiment.experimentParams.experimentDir").c_str();
    experimentParams.numSegmentsPerCurve
          = (int)config.lookup("experiment.experimentParams.numSegments");
    experimentParams.arclength = config.lookup("experiment.experimentParams.arclength");

    // Seeds
    experimentParams.bootstrapSeed
          = (int)config.lookup("experiment.experimentParams.random.bootstrapSeed");
    experimentParams.curvePurturbSeed
          = (int)config.lookup("experiment.experimentParams.random.perturbSeed");

    if (experimentParams.bootstrapSeed == 0) {
        experimentParams.bootstrapSeed = time(0);
    }
    if (experimentParams.curvePurturbSeed == 0) {
        experimentParams.curvePurturbSeed = time(0);
    }

    // Weighting parameter stuff
    experimentParams.weightingParameters.mu
          = config.lookup("experiment.experimentParams.weighting.mu");
    experimentParams.weightingParameters.eps
          = config.lookup("experiment.experimentParams.weighting.eps");
    experimentParams.weightingParameters.numStepsInt
          = (int)config.lookup("experiment.experimentParams.weighting.numStepsInt");
    experimentParams.weightingParameters.numCurvatureSteps
          = (int)config.lookup("experiment.experimentParams.weighting.numCurvatureSteps");
    experimentParams.weightingParameters.absorbtion
          = config.lookup("experiment.experimentParams.weighting.absorbtion");
    experimentParams.weightingParameters.scatter
          = config.lookup("experiment.experimentParams.weighting.scatter");

    // TODO: Should these be configurable in the file?
    experimentParams.weightingParameters.minBound = 0.0;
    experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;

    return experimentParams;
}

bool LoadConfigFile(const std::string &filename, libconfig::Config &experimentConfig)
{
    try {
        experimentConfig.readFile(filename);
    } catch (const libconfig::FileIOException &fioex) {
        std::cout << "I/O error while reading file." << std::endl;
        return false;
    } catch (const libconfig::ParseException &pex) {
        std::cout << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - "
                  << pex.getError() << std::endl;
        return false;
    }
    return true;
}

// Shoot along the z-axis
int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
        return 1;
    }
    std::string configFilename(argv[1]);
    libconfig::Config experimentConfig;
    if (!LoadConfigFile(configFilename, experimentConfig)) {
        std::cout << "Failed to load config file: " << configFilename << std::endl;
        return false;
    }

    twisty::ExperimentRunner::ExperimentParameters experimentParams
          = ParseExperimentParamsFromConfig(experimentConfig);
    if (!std::filesystem::exists(experimentParams.experimentDirPath)) {
        std::filesystem::create_directories(experimentParams.experimentDirPath);
    }
    const std::string experimentCfgCopyFilename
          = std::string(experimentParams.experimentDirPath) + "/parameters.cfg";
    std::filesystem::copy_file(configFilename, experimentCfgCopyFilename,
          std::filesystem::copy_options::overwrite_existing);

    // Experiment specific parameters
    const uint32_t numInitialCurves
          = (int)experimentConfig.lookup("experiment.smallSegmentExperiment.numInitialCurves");
    const uint32_t numPerInitialCurve
          = (int)experimentConfig.lookup("experiment.smallSegmentExperiment.numPerInitialCurve");
    const uint32_t numEmitterDirections
          = (int)experimentConfig.lookup("experiment.smallSegmentExperiment.numEmitterDirections");
    const float distanceFromPlane
          = experimentConfig.lookup("experiment.smallSegmentExperiment.distanceFromPlane");
    const uint32_t numPathsPerInternal
          = (int)experimentConfig.lookup("experiment.smallSegmentExperiment.numPathsPerInternal");
    const uint32_t numPathsSkipPerInternal = (int)experimentConfig.lookup(
          "experiment.smallSegmentExperiment.numPathsSkipPerInternal");

    // Ok, we want the ray emitter
    const Farlor::Vector3 emitterStart { 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(0.0f, 0.0f, 1.0f).Normalized();
    twisty::RayGeometry rayEmitter(emitterStart, emitterDir);

    const Farlor::Vector3 recieverPos = Farlor::Vector3(0.0f, 0.0f, distanceFromPlane);
    const Farlor::Vector3 recieverDir = Farlor::Vector3(0.0f, 0.0f, 1.0f).Normalized();
    twisty::RayGeometry rayReciever(recieverPos, recieverDir);

    // Start at somehting close to 1.05 times the minimum arclength and go to 1.5 times the arclength
    if (experimentParams.arclength < 3.0) {
        std::cout << "Overwritting arclength as it was too small" << std::endl;
        experimentParams.arclength = 3.0;
    }

    std::stringstream innerBlockSS;
    innerBlockSS << "<Converged Value>";
    std::cout << innerBlockSS.str() << std::endl;

    boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;

    // Cache the initial seeds
    uint32_t cachedInitialBootstrapSeed = experimentParams.bootstrapSeed;
    uint32_t cachedInitialPerturbSeed = experimentParams.curvePurturbSeed;

    std::mt19937 initialCurveGen(cachedInitialBootstrapSeed);
    for (uint32_t initialCurveIdx = 0; initialCurveIdx < numInitialCurves; ++initialCurveIdx) {
        int initialCurveSeed = initialCurveGen();
        while (initialCurveSeed == 0) {
            initialCurveSeed = initialCurveGen();
        }

        boost::multiprecision::cpp_dec_float_100 maxResult = 0.0;

        std::mt19937 perCurveGen(cachedInitialPerturbSeed);
        for (uint32_t perInitialCurveIdx = 0; perInitialCurveIdx < numPerInitialCurve;
              ++perInitialCurveIdx) {
            // Go ahead and overwrite the per curve seed
            int perCurveSeed = perCurveGen();
            while (perCurveSeed == 0) {
                perCurveSeed = perCurveGen();
            }

            // Update random seeds
            experimentParams.bootstrapSeed = initialCurveSeed;
            experimentParams.curvePurturbSeed = perCurveSeed;

            twisty::Bootstrapper bootstrapper(rayEmitter, rayReciever);
            std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner
                  = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(
                        experimentParams, bootstrapper);
            bool result = upExperimentRunner->Setup();

            if (!result) {
                upExperimentRunner->Shutdown();
                std::cout << "Failed to setup experiment runner." << std::endl;
                return 1;
            }

            twisty::ExperimentRunner::ExperimentResults results
                  = upExperimentRunner->RunExperiment();
            if (results.experimentWeight > maxResult) {
                maxResult = results.experimentWeight;
            }
            upExperimentRunner->Shutdown();
        }

        averagedResult += (maxResult * (1.0 / (numInitialCurves)));
    }

    std::cout << "Converged Value: " << averagedResult << std::endl;

    // Export freeze frame pixel data
    {
        std::filesystem::path converedValuesOutputPath = experimentParams.experimentDirPath;
        converedValuesOutputPath.append("Converged.dat");

        std::ofstream convergedValuesOutputStream(converedValuesOutputPath.string());
        if (!convergedValuesOutputStream.is_open()) {
            std::cout << "Failed to create converged values outfile" << std::endl;
            exit(1);
        }

        convergedValuesOutputStream << averagedResult << std::endl;
    }

    return 0;
}
