#include "FullExperimentRunner.h"

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>

const float toRadians = TwistyPi / 180.0f;
const float toDegrees = 180.0f / TwistyPi;

using namespace twisty;

int main(int argc, char *argv[])
{
    if (argc < 13)
    {
        std::cout << "Call as: " << argv[0] << " numPathsToGenerate numPathsToSkip experimentName experimentDir numSegments targetArclength minDegree maxDegree numDegreeSteps useGpu bootstrapperSeed perturbSeed" << std::endl;
        return 1;
    }

    const uint32_t numPathsToGenerate = std::stoi(argv[1]);
    const uint32_t numPathsToSkip = std::stoi(argv[2]);
    const std::string experimentName(argv[3]);
    const std::string experimentDirectoryPath(argv[4]);
    const uint32_t numExperimentSegments = std::stoi(argv[5]);
    const float targetArclength = std::stof(argv[6]);

    const float minDegree = std::stof(argv[7]);
    const float maxDegree = std::stof(argv[8]);
    const uint32_t numDegreeSteps = std::stoi(argv[9]);

    const bool useGpu = std::stoi(argv[10]);
    const int boostrapperSeed = std::stoi(argv[11]);
    const int perturbSeed = std::stoi(argv[12]);

    std::cout << "Command line args: " << std::endl;
    std::cout << "\tNum paths to gen: " << numPathsToGenerate << std::endl;
    std::cout << "\tExperiment name: " << experimentName << std::endl;
    std::cout << "\tBootstrapper Seed: " << boostrapperSeed << std::endl;
    std::cout << "\tPerturb Seed: " << perturbSeed << std::endl;

    // Currently this is a hardcoded position and direction
    const Farlor::Vector3 emitterStart{0.0f, 0.0f, 0.0f};
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    RayGeometry rayEmitter(emitterStart, emitterDir);

    const Farlor::Vector3 recieverPos{10.0f, 0.0f, 0.0f};

    // The reciever direction will change based on the parameters passed in
    // We are running a bunch of differnet ones
    const float degreeStepSize = (maxDegree - minDegree) / (numDegreeSteps - 1);
    for (uint32_t degreeIdx = 0; degreeIdx < numDegreeSteps; degreeIdx++)
    {
        const float currentAngleDegree = minDegree + degreeStepSize * degreeIdx;
        const float currentAngleRads = currentAngleDegree * toRadians;

        const Farlor::Vector3 recieverDir{std::cos(currentAngleRads), std::sin(currentAngleRads), 0.0f};

        RayGeometry rayReciever(recieverPos, recieverDir);

        GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, targetArclength, boostrapperSeed);

        std::cout << "Experiment Path Count: " << numPathsToGenerate << std::endl;

        std::stringstream experimentDirSS;
        experimentDirSS << experimentDirectoryPath;
        experimentDirSS << "/Degree_" << degreeIdx;

        ExperimentRunner::ExperimentParameters experimentParams;
        experimentParams.numPathsInExperiment = numPathsToGenerate;
        experimentParams.numPathsToSkip = numPathsToSkip;
        experimentParams.exportGeneratedCurves = true;
        experimentParams.experimentName = experimentName;
        experimentParams.experimentDirPath = experimentDirSS.str();
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

        upExperimentRunner = std::make_unique<FullExperimentRunner>(experimentParams, bootstrapper);

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
#endif

    return 0;
}
