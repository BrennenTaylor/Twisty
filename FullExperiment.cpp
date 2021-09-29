#include "FullExperimentRunnerOptimalPerturb.h"

#include <libconfig.h++>

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <filesystem>
#include <string>

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(const libconfig::Config& config)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Hardocded values
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

    experimentParams.bootstrapSeed = (int)config.lookup("experiment.experimentParams.random.bootstrapSeed");
    experimentParams.curvePurturbSeed = (int)config.lookup("experiment.experimentParams.random.perturbSeed");

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

int main(int argc, char *argv[])
{
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

        // Override params
        //experimentParams.numPathsInExperiment = 10000;
        //experimentParams.numPathsToSkip = 100;

        if (!std::filesystem::exists(experimentParams.experimentDirPath))
        {
            std::filesystem::create_directories(experimentParams.experimentDirPath);
        }
        std::cout << "experimentDirPath: " << experimentParams.experimentDirPath << std::endl;
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
        const twisty::RayGeometry rayEmitter(startPos, startDir);

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

            std::cout << "S: " << s << std::endl;

            experimentParams.perExperimentDirSubfolder = std::string("s_") + std::to_string(sIdx);

            Farlor::Vector3 endPos(0.0f, 0.0f, distanceFromPlane);
            Farlor::Vector3 evalVector = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();
            twisty::RayGeometry rayReciever(endPos, evalVector);

            twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever);

            std::cout << "Experiment Path Count: " << experimentParams.numPathsInExperiment << std::endl;

            std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = nullptr;

            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb" << std::endl;
            upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);

            bool result = upExperimentRunner->Setup();
            if (!result)
            {
                upExperimentRunner->Shutdown();
                std::cout << "Failed to setup experiment runner." << std::endl;
                return 1;
            }
            auto start = std::chrono::high_resolution_clock::now();
            twisty::ExperimentRunner::ExperimentResults results = upExperimentRunner->RunExperiment();
            auto end = std::chrono::high_resolution_clock::now();
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            auto timeSec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

            std::cout << "Paths Generated: " << results.totalPathsGenerated << std::endl;
            std::cout << "Total experiment weight: " << results.experimentWeight << std::endl;
            std::cout << "Avg path weight: " << results.experimentWeight / results.totalPathsGenerated << std::endl;

            // Retrieve Data we want from experiment
            upExperimentRunner->Shutdown();
            upExperimentRunner.reset(nullptr);

            ofs << s << ", " << results.experimentWeight << std::endl;

            std::cout << "\tEnd Dir: " << evalVector << std::endl;
            std::cout << "\tEval: " << results.experimentWeight << std::endl;
        }
        ofs.close();
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    _CrtDumpMemoryLeaks();
    _CrtMemDumpAllObjectsSince(NULL);
#endif

    return 0;
}
