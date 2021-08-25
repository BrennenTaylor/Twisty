// #include "FullExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"
// #include "FullExperimentRunnerOptimalPerturbOptimized.h"

#include <libconfig.h++>

//#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"

//#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
//#include "GpuFullExperimentRunnerGeneral.h"
//#include "GpuFullExperimentRunnerGeneral2.h"
//#endif

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <filesystem>

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
        if (!std::filesystem::exists(experimentParams.experimentDirPath))
        {
            std::filesystem::create_directories(experimentParams.experimentDirPath);
        }
        const std::string experimentCfgCopyFilename = std::string(experimentParams.experimentDirPath) + "/parameters.cfg";
        std::filesystem::copy_file(configFilename, experimentCfgCopyFilename, std::filesystem::copy_options::overwrite_existing);

        // Parse experiment specific parameters
        uint32_t runnerVersion = experimentConfig.lookup("experiment.runnerVersion");
        
        // Bootstrap method
        Farlor::Vector3 emitterStart;
        emitterStart.x = experimentConfig.lookup("experiment.geometry.startPos.x");
        emitterStart.y = experimentConfig.lookup("experiment.geometry.startPos.y");
        emitterStart.z = experimentConfig.lookup("experiment.geometry.startPos.z");

        Farlor::Vector3 emitterDir;
        emitterDir.x = experimentConfig.lookup("experiment.geometry.startDir.x");
        emitterDir.y = experimentConfig.lookup("experiment.geometry.startDir.y");
        emitterDir.z = experimentConfig.lookup("experiment.geometry.startDir.z");
        emitterDir.Normalize();
        const twisty::RayGeometry rayEmitter(emitterStart, emitterDir);

        const double zMin = 0.0;
        const double zMax = 10.0;

        const uint32_t numZValues = 10;
        const double deltaZ = (zMax - zMin) / (numZValues - 1);
        const uint32_t zIdx = 2;
        const double receiverZ = zMin + deltaZ * zIdx;
        const double ringRadius = 1.75;

        const Farlor::Vector3 recieverPos = Farlor::Vector3(ringRadius, 0.0f, receiverZ);
        const Farlor::Vector3 recieverDir = (recieverPos - emitterStart).Normalized();

        twisty::RayGeometry rayReciever(recieverPos, recieverDir);

        float targetArclength = (recieverPos - emitterStart).Magnitude() * 1.1f;
        targetArclength = std::max(targetArclength, 3.0f);

        twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever);

        std::cout << "Experiment Path Count: " << experimentParams.numPathsInExperiment << std::endl;

        std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = nullptr;

        std::cout << "Runner version: " << runnerVersion << std::endl;

// #if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
        // switch (runnerVersion)
        // {
        // case 0:
        // {
        //     std::cout << "Selected Runner Method: FullExperimentRunner" << std::endl;
        //     upExperimentRunner = std::make_unique<twisty::FullExperimentRunner>(experimentParams, bootstrapper);
        // } break;
        // case 1:
        // {
        std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb" << std::endl;
        upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);
        // } break;
        // case 2:
        // {
        //     std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturbOptimized" << std::endl;
        //     upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturbOptimized>(experimentParams, bootstrapper, 1, 1);
        // } break;

        //case 3:
        //{
        //    std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturbOptimized_GPU" << std::endl;
        //    upExperimentRunner = std::make_unique<FullExperimentRunnerOptimalPerturbOptimized_GPU>(experimentParams, bootstrapper);
        //} break;

        // Debug modes
        // case 66:
        // {
        //     upExperimentRunner = std::make_unique<FullExperimentRunnerOldMethodBridge>(experimentParams, bootstrapper, kdsRange, tdsRange);
        // } break;
        // default:
        // {
        //     std::cout << "Invalid experiment runner method selected" << std::endl;
        //     exit(1);
        // }
        // }
// #else
        // upExperimentRunner = std::make_unique<FullExperimentRunner>(experimentParams, bootstrapper, kdsRange, tdsRange);
// #endif

        std::cout << "Spot 1 - Num hardware threads: " << std::thread::hardware_concurrency() << std::endl;


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
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    _CrtDumpMemoryLeaks();
    _CrtMemDumpAllObjectsSince(NULL);
#endif

    return 0;
}
