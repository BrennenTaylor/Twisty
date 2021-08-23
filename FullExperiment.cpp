#include "FullExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "FullExperimentRunnerOptimalPerturbOptimized.h"

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

int main(int argc, char *argv[])
{
    {
        if (argc < 2)
        {
            std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
            return 1;
        }

        const std::string configFilename(argv[1]);

        libconfig::Config experimentConfig;
        try
        {
            experimentConfig.readFile(configFilename);
        }
        catch (const libconfig::FileIOException &fioex)
        {
            std::cout << "I/O error while reading file." << std::endl;
            return (1);
        }
        catch (const libconfig::ParseException &pex)
        {
            std::cout << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                      << " - " << pex.getError() << std::endl;
            return (1);
        }

        uint32_t numPathsToGenerate = experimentConfig.lookup("experiment.pathsToGenerate");
        uint32_t numPathsToSkip = experimentConfig.lookup("experiment.pathsToSkip");
        std::string experimentName = experimentConfig.lookup("experiment.name");
        std::string experimentDirPath = experimentConfig.lookup("experiment.experimentDir");
        uint32_t numExperimentSegments = experimentConfig.lookup("experiment.numSegments");
        float minTargetArclength = experimentConfig.lookup("experiment.minArclength");
        float maxTargetArclength = experimentConfig.lookup("experiment.maxArclength");
        uint32_t runnerVersion = experimentConfig.lookup("experiment.runnerVersion");
        uint32_t boostrapperSeed = experimentConfig.lookup("experiment.random.bootstrapperSeed");
        uint32_t perturbSeed = experimentConfig.lookup("experiment.random.perturbSeed");

        std::cout << "Command line args: " << std::endl;
        std::cout << "\tNum paths to gen: " << numPathsToGenerate << std::endl;
        std::cout << "\tExperiment name: " << experimentName << std::endl;
        std::cout << "\tBootstrapper Seed: " << boostrapperSeed << std::endl;
        std::cout << "\tPerturb Seed: " << perturbSeed << std::endl;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
        _CrtDumpMemoryLeaks();
        _CrtMemDumpAllObjectsSince(NULL);
#endif

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

        twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, targetArclength, boostrapperSeed);

        std::cout << "Experiment Path Count: " << numPathsToGenerate << std::endl;

        twisty::ExperimentRunner::ExperimentParameters experimentParams;
        experimentParams.numPathsInExperiment = numPathsToGenerate;
        experimentParams.numPathsToSkip = numPathsToSkip;
        experimentParams.exportGeneratedCurves = true;
        experimentParams.experimentName = experimentName;
        experimentParams.experimentDirPath = experimentDirPath;
        experimentParams.numSegmentsPerCurve = numExperimentSegments;
        experimentParams.maximumBootstrapCurveError = 0.5f;
        experimentParams.curvePerturbMethod = twisty::ExperimentRunner::CurvePerturbMethod::SimpleGeometry;
        experimentParams.curvePurturbSeed = perturbSeed;
        experimentParams.rotateInitialSeedCurveRadians = 0.0f;

        experimentParams.weightingParameters.mu = experimentConfig.lookup("experiment.weighting.mu");
        experimentParams.weightingParameters.eps = experimentConfig.lookup("experiment.weighting.eps");
        experimentParams.weightingParameters.numStepsInt = experimentConfig.lookup("experiment.weighting.numStepsInt");
        experimentParams.weightingParameters.minBound = 0.0;
        experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;
        experimentParams.weightingParameters.numCurvatureSteps =  experimentConfig.lookup("experiment.weighting.numCurvatureSteps");;
        // Lets give some absorbtion as well
        // Absorbtion 1/20 off the time
        experimentParams.weightingParameters.absorbtion = experimentConfig.lookup("experiment.weighting.absorbtion");
        // 1/5 scatter means one event every 5 units, thus 2 scattering events in the shortest
        // or 5 in the longest 100 unit path
        experimentParams.weightingParameters.scatter = experimentConfig.lookup("experiment.weighting.scatter");

        std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = nullptr;

        std::cout << "Runner version: " << runnerVersion << std::endl;

// #if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
        switch (runnerVersion)
        {
        case 0:
        {
            std::cout << "Selected Runner Method: FullExperimentRunner" << std::endl;
            upExperimentRunner = std::make_unique<twisty::FullExperimentRunner>(experimentParams, bootstrapper);
        } break;
        case 1:
        {
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb" << std::endl;
            upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);
        } break;
        case 2:
        {
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturbOptimized" << std::endl;
            upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturbOptimized>(experimentParams, bootstrapper, 1, 1);
        } break;

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
        default:
        {
            std::cout << "Invalid experiment runner method selected" << std::endl;
            exit(1);
        }
        }
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
