#include "FullExperimentRunner.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "GpuFullExperimentRunnerGeneral.h"
#endif

#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"

#include <FMath/Vector3.h>

#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <memory>

using namespace twisty;

int main(int argc, char *argv[])
{
    if (argc < 10)
    {
        fmt::print("Call as: {} numPathsToGenerate numPathsToSkip experimentName numSegments minArclength maxArclength useGpu bootstrapperSeed perturbSeed\n", argv[0]);
        return 1;
    }

    const uint32_t numPathsToGenerate = std::stoi(argv[1]);
    const uint32_t numPathsToSkip = std::stoi(argv[2]);
    const std::string experimentName(argv[3]);
    const uint32_t numExperimentSegments = std::stoi(argv[4]);
    const float minTargetArclength = std::stof(argv[5]);
    const float maxTargetArclength = std::stof(argv[6]);
    const bool useGpu = std::stoi(argv[7]);
    const int boostrapperSeed = std::stoi(argv[8]);
    const int perturbSeed = std::stoi(argv[9]);

    std::cout << "Command line args: " << std::endl;
    std::cout << "\tNum paths to gen: " << numPathsToGenerate << std::endl;
    std::cout << "\tExperiment name: " << experimentName << std::endl;
    std::cout << "\tBootstrapper Seed: " << boostrapperSeed << std::endl;
    std::cout << "\tPerturb Seed: " << perturbSeed << std::endl;


    // Bootstrap method
    const Range defaultBounds = {-1.0f, 1.0f};
    const Farlor::Vector3 emitterStart{0.0f, 0.0f, 0.0f};
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    RayGeometry rayEmitter(emitterStart, emitterDir);

    const Farlor::Vector3 recieverPos{10.0f, 0.0f, 0.0f};
    const Farlor::Vector3 recieverDir{1.0, 0.0f, 0.0f};

    RayGeometry rayReciever(recieverPos, recieverDir);

    const Range arclengthRange = {minTargetArclength, maxTargetArclength};

    GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, arclengthRange, boostrapperSeed);

    std::cout << "Experiment Path Count: " << numPathsToGenerate << std::endl;

    // This is the range we want to meet
    // The range of actual curvature/torsion * ds is below
    Range kdsRange = {0.0f, 2.0f};
    Range tdsRange = {-1.0f, 1.0f};

    ExperimentRunner::ExperimentParameters experimentParams;
    experimentParams.numPathsInExperiment = numPathsToGenerate;
    experimentParams.numPathsToSkip = numPathsToSkip;
    experimentParams.exportGeneratedCurves = true;
    experimentParams.experimentName = experimentName;
    experimentParams.experimentDir = experimentName;
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

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    if (useGpu)
    {
        upExperimentRunner = std::make_unique<GpuFullExperimentRunnerGeneral>(experimentParams, bootstrapper, kdsRange, tdsRange);
    }
    else
    {
        // NOTE: This is tuned to maximize purturb kernel
        upExperimentRunner = std::make_unique<FullExperimentRunner>(experimentParams, bootstrapper, kdsRange, tdsRange);
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

    std::cout << "Paths Generated: " << results.totalPathsGenerated << std::endl;
    std::cout << "Total experiment weight: " << results.experimentWeight << std::endl;
    std::cout << "Avg path weight: " << results.experimentWeight / results.totalPathsGenerated << std::endl;

    // Retrieve Data we want from experiment
    upExperimentRunner->Shutdown();
    upExperimentRunner.reset(nullptr);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    _CrtDumpMemoryLeaks();
#endif

    return 0;
}
