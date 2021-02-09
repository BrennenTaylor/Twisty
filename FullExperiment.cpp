#include "FullExperimentRunner.h"
#include "FullExperimentRunner2.h"
// #include "FullExperimentRunnerOldMethodBridge.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "FullExperimentRunnerOptimalPerturbOptimized.h"

//#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h"

//#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
//#include "GpuFullExperimentRunnerGeneral.h"
//#include "GpuFullExperimentRunnerGeneral2.h"
//#endif

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

using namespace twisty;

int main(int argc, char *argv[])
{
    {
        if (argc < 11)
        {
            std::cout << "Call as: " << argv[0] << " numPathsToGenerate numPathsToSkip experimentName experimentPath numSegments minArclength maxArclength runnerVersion bootstrapperSeed perturbSeed" << std::endl;
            return 1;
        }

        const uint64_t numPathsToGenerate = std::stoll(argv[1]);
        const uint64_t numPathsToSkip = std::stoll(argv[2]);
        const std::string experimentName(argv[3]);
        const std::string experimentDirPath(argv[4]);
        const uint32_t numExperimentSegments = std::stoi(argv[5]);
        const float minTargetArclength = std::stof(argv[6]);
        const float maxTargetArclength = std::stof(argv[7]);
        const uint32_t runnerVersion = std::stoi(argv[8]);
        const uint32_t boostrapperSeed = std::stoi(argv[9]);
        const uint32_t perturbSeed = std::stoi(argv[10]);

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
        const Range defaultBounds = { -1.0f, 1.0f };
        const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
        const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
        RayGeometry rayEmitter(emitterStart, emitterDir);

        const Farlor::Vector3 recieverPos{ 10.0f, 0.0f, 0.0f };
        const Farlor::Vector3 recieverDir{ 1.0, 0.0f, 0.0f };

        RayGeometry rayReciever(recieverPos, recieverDir);

        const Range arclengthRange = { minTargetArclength, maxTargetArclength };

        GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, arclengthRange, boostrapperSeed);

        std::cout << "Experiment Path Count: " << numPathsToGenerate << std::endl;

        // This is the range we want to meet
        // The range of actual curvature/torsion * ds is below
        Range kdsRange = { 0.0f, 2.0f };
        Range tdsRange = { -1.0f, 1.0f };

        ExperimentRunner::ExperimentParameters experimentParams;
        experimentParams.numPathsInExperiment = numPathsToGenerate;
        experimentParams.numPathsToSkip = numPathsToSkip;
        experimentParams.exportGeneratedCurves = true;
        experimentParams.experimentName = experimentName;
        experimentParams.experimentDirPath = experimentDirPath;
        experimentParams.numSegmentsPerCurve = numExperimentSegments;
        experimentParams.maximumBootstrapCurveError = 0.5f;
        experimentParams.curvePerturbMethod = ExperimentRunner::CurvePerturbMethod::SimpleGeometry;
        experimentParams.curvePurturbSeed = perturbSeed;
        experimentParams.rotateInitialSeedCurveRadians = 0.0f;

        experimentParams.weightingParameters.mu = 0.1;
        experimentParams.weightingParameters.eps = 0.1;
        experimentParams.weightingParameters.numStepsInt = 2000;
        experimentParams.weightingParameters.minBound = 0.0;
        experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;
        experimentParams.weightingParameters.numCurvatureSteps = 10000;
        // Lets give some absorbtion as well
        // Absorbtion 1/20 off the time
        experimentParams.weightingParameters.absorbtion = 0.05;
        // 1/5 scatter means one event every 5 units, thus 2 scattering events in the shortest
        // or 5 in the longest 100 unit path
        experimentParams.weightingParameters.scatter = 0.2;

        std::unique_ptr<ExperimentRunner> upExperimentRunner = nullptr;

        std::cout << "Runner version: " << runnerVersion << std::endl;

// #if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
        switch (runnerVersion)
        {
        case 0:
        {
            std::cout << "Selected Runner Method: FullExperimentRunner" << std::endl;
            upExperimentRunner = std::make_unique<FullExperimentRunner>(experimentParams, bootstrapper, kdsRange, tdsRange);
        } break;
        case 1:
        {
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb" << std::endl;
            upExperimentRunner = std::make_unique<FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper, kdsRange, tdsRange);
        } break;
        case 2:
        {
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturbOptimized" << std::endl;
            upExperimentRunner = std::make_unique<FullExperimentRunnerOptimalPerturbOptimized>(experimentParams, bootstrapper, 1, 1);
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
