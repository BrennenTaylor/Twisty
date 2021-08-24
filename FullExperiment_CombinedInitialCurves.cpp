#include "FullExperimentRunner.h"
//#include "FullExperimentRunnerOldMethodBridge.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "FullExperimentRunnerOptimalPerturbOptimized.h"

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

using namespace twisty;

int main(int argc, char *argv[])
{
    if (argc < 12)
    {
        std::cout << "Call as: " << argv[0] << " numPathsToGenerate numPathsToSkip experimentName experimentPath numSegments targetArclength runnerVersion numInitialCurves numRunsPerInitialCurve bootstrapperSeed perturbSeed" << std::endl;
        return 1;
    }

    const uint64_t numPathsToGenerate = std::stoll(argv[1]);
    const uint64_t numPathsToSkip = std::stoll(argv[2]);
    const std::string experimentName(argv[3]);
    const std::string experimentDirPath(argv[4]);
    const uint32_t numExperimentSegments = std::stoi(argv[5]);
    const float targetArclength = std::stof(argv[6]);
    const uint32_t runnerVersion = std::stoi(argv[7]);
    const uint32_t numInitialCurves = std::stoi(argv[8]);
    const uint32_t numRunsPerInitialCurve = std::stoi(argv[9]);
    const uint32_t bootstrapperSeed = std::stoi(argv[10]);
    const uint32_t perturbSeed = std::stoi(argv[11]);

    std::cout << "Command line args: " << std::endl;
    std::cout << "\tNum paths to gen: " << numPathsToGenerate << std::endl;
    std::cout << "\tExperiment name: " << experimentName << std::endl;
    std::cout << "\tBootstrapper Seed: " << bootstrapperSeed << std::endl;
    std::cout << "\tPerturb Seed: " << perturbSeed << std::endl;
    
    
    std::vector<boost::multiprecision::cpp_dec_float_100> initialCurveWeights(numInitialCurves);


    boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;

    std::mt19937 initialCurveGen(bootstrapperSeed);
    for (uint32_t initialCurveIdx = 0; initialCurveIdx < numInitialCurves; ++initialCurveIdx)
    {

        int initialCurveSeed = initialCurveGen();
        while (initialCurveSeed == 0)
        {
            initialCurveSeed = initialCurveGen();
        }

        boost::multiprecision::cpp_dec_float_100 maxResult = 0.0;

        std::mt19937 perCurveGen(perturbSeed);
        for (uint32_t perInitialCurveIdx = 0; perInitialCurveIdx < numRunsPerInitialCurve; ++perInitialCurveIdx)
        {
            int perCurveSeed = perCurveGen();
            while (perCurveSeed == 0)
            {
                perCurveSeed = perCurveGen();
            }


            // Bootstrap method
            const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
            const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
            RayGeometry rayEmitter(emitterStart, emitterDir);

            const Farlor::Vector3 recieverPos{ 10.0f, 0.0f, 0.0f };
            const Farlor::Vector3 recieverDir{ 1.0, 0.0f, 0.0f };

            RayGeometry rayReciever(recieverPos, recieverDir);

            GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, targetArclength, initialCurveSeed);

            std::cout << "Experiment Path Count: " << numPathsToGenerate << std::endl;

            ExperimentRunner::ExperimentParameters experimentParams;
            experimentParams.numPathsInExperiment = numPathsToGenerate;
            experimentParams.numPathsToSkip = numPathsToSkip;
            experimentParams.exportGeneratedCurves = true;
            experimentParams.experimentName = experimentName;
            experimentParams.experimentDirPath = experimentDirPath;
            experimentParams.numSegmentsPerCurve = numExperimentSegments;
            experimentParams.maximumBootstrapCurveError = 0.5f;
            experimentParams.curvePurturbSeed = perCurveSeed;
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
            std::cout << "Seeds: " << initialCurveSeed << ", " << perCurveSeed << std::endl;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
            switch (runnerVersion)
            {
            case 0:
            {
                std::cout << "\tSelected Runner Method: FullExperimentRunner" << std::endl;
                upExperimentRunner = std::make_unique<FullExperimentRunner>(experimentParams, bootstrapper);
            } break;
            case 1:
            {
                std::cout << "\tSelected Runner Method: FullExperimentRunnerOptimalPerturb" << std::endl;
                upExperimentRunner = std::make_unique<FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);
            } break;
            case 2:
            {
                std::cout << "\tSelected Runner Method: FullExperimentRunnerOptimalPerturbOptimized" << std::endl;
                upExperimentRunner = std::make_unique<FullExperimentRunnerOptimalPerturbOptimized>(experimentParams, bootstrapper, 1, 1);
            } break;

            // Debug modes
            //case 66:
            //{
            //    upExperimentRunner = std::make_unique<FullExperimentRunnerOldMethodBridge>(experimentParams, bootstrapper, kdsRange, tdsRange);
            //} break;
            default:
            {
                std::cout << "Invalid experiment runner method selected" << std::endl;
                exit(1);
            }
            }
#else
            upExperimentRunner = std::make_unique<FullExperimentRunner>(experimentParams, bootstrapper, kdsRange, tdsRange);
#endif

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

            std::cout << "\tPaths Generated: " << results.totalPathsGenerated << std::endl;
            std::cout << "\tTotal experiment weight: " << results.experimentWeight << std::endl;
            std::cout << "\tAvg path weight: " << results.experimentWeight / results.totalPathsGenerated << std::endl;

            // Retrieve Data we want from experiment
            upExperimentRunner->Shutdown();
            upExperimentRunner.reset(nullptr);


            if (results.experimentWeight > maxResult)
            {
                maxResult = results.experimentWeight;
            }
        }

        initialCurveWeights[initialCurveIdx] = maxResult;
        averagedResult += maxResult * (1.0 / numInitialCurves);
    }

    std::cout << "IC Average Result: " << averagedResult << std::endl;
    
    boost::multiprecision::cpp_dec_float_100 icVar = 0.0;
    for (auto& value : initialCurveWeights)
    {
        icVar += boost::multiprecision::pow((value - averagedResult), 2.0);
    }
    icVar /= (initialCurveWeights.size() - 1);
    std::cout << "IC Var: " << icVar << std::endl;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    _CrtDumpMemoryLeaks();
#endif

    return 0;
}
