/*

    This experiment executes the monte carlo feynman path integral with a specified number of segments.
    The target number of segments should be small as the specified weight function for this corresponds to the
    simplified weighting function derived by Jerry.

*/

#include "FullExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

const float distanceFromPlane = 10.0f;

const uint32_t numPathsPerInternal = 100000;
const uint32_t numPathsSkipPerInternal = 1000;

// Shoot along the z-axis
int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        std::cout << "Call as: " << argv[0] << " experimentName experimentOutputPath bootstrapperSeed perturbSeed numInitialCurves numPerInitialCurve numSegments" << std::endl;
        return 1;
    }

    const std::string experimentName(argv[1]);
    const std::string experimentOutputPath(argv[2]);
    int bootstrapperSeed = std::stoi(argv[3]);
    int perturbSeed = std::stoi(argv[4]);
    const uint32_t numInitialCurves = std::stoi(argv[5]);
    const uint32_t numPerInitialCurve = std::stoi(argv[6]);
    const uint32_t numSegments = std::stoi(argv[7]);

    std::filesystem::path outputDirectoryPath = std::filesystem::path(experimentOutputPath);
    std::cout << "Output Directory Path: " << outputDirectoryPath << std::endl;
    if (!std::filesystem::exists(outputDirectoryPath))
    {
        std::filesystem::create_directories(outputDirectoryPath);
    }

    if (bootstrapperSeed == 0)
    {
        bootstrapperSeed = time(0);
    }

    if (perturbSeed == 0)
    {
        perturbSeed = time(0);
    }

    // Ok, we want the ray emitter
    const Farlor::Vector3 emitterStart{0.0f, 0.0f, 0.0f};
    const Farlor::Vector3 emitterDir = Farlor::Vector3(0.0f, 0.0f, 1.0f).Normalized();
    twisty::RayGeometry rayEmitter(emitterStart, emitterDir);

    const Farlor::Vector3 recieverPos = Farlor::Vector3(0.0f, 0.0f, distanceFromPlane);
    const Farlor::Vector3 recieverDir = Farlor::Vector3(0.0f, 0.0f, 1.0f).Normalized();
    twisty::RayGeometry rayReciever(recieverPos, recieverDir);

    // Start at somehting close to 1.05 times the minimum arclength and go to 1.5 times the arclength
    float targetArclength = (recieverPos - emitterStart).Magnitude() * 1.1f;
    targetArclength = std::max(targetArclength, 3.0f);

    std::stringstream innerBlockSS;
    innerBlockSS << "<Converged Value>";
    std::cout << innerBlockSS.str() << std::endl;

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
        for (uint32_t perInitialCurveIdx = 0; perInitialCurveIdx < numPerInitialCurve; ++perInitialCurveIdx)
        {
            int perCurveSeed = perCurveGen();
            while (perCurveSeed == 0)
            {
                perCurveSeed = perCurveGen();
            }

            twisty::ExperimentRunner::ExperimentParameters experimentParams;
            experimentParams.numPathsInExperiment = numPathsPerInternal;
            experimentParams.numPathsToSkip = numPathsSkipPerInternal;
            experimentParams.exportGeneratedCurves = false;
            experimentParams.experimentName = experimentName;
            experimentParams.numSegmentsPerCurve = numSegments;
            experimentParams.maximumBootstrapCurveError = 0.5f;
            experimentParams.curvePurturbSeed = perCurveSeed;
            experimentParams.rotateInitialSeedCurveRadians = 0.0f;

            // Use a big mu value
            experimentParams.weightingParameters.mu = 99.0;
            experimentParams.weightingParameters.eps = 0.1;
            experimentParams.weightingParameters.numStepsInt = 2000;
            experimentParams.weightingParameters.minBound = 0.0;
            experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;
            experimentParams.weightingParameters.numCurvatureSteps = 10000;
            experimentParams.weightingParameters.absorbtion = 0.9;
            experimentParams.weightingParameters.scatter = 0.1;
            experimentParams.rotateInitialSeedCurveRadians = 0.0f;

            twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, targetArclength, initialCurveSeed);
            std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);
            bool result = upExperimentRunner->Setup();

            if (!result)
            {
                upExperimentRunner->Shutdown();
                std::cout << "Failed to setup experiment runner." << std::endl;
                return 1;
            }

            twisty::ExperimentRunner::ExperimentResults results = upExperimentRunner->RunExperiment();
            if (results.experimentWeight > maxResult)
            {
                maxResult = results.experimentWeight;
            }
            upExperimentRunner->Shutdown();
        }

        averagedResult += (maxResult * (1.0 / (numInitialCurves)));
    }

    std::cout << "Converged Value: " << averagedResult << std::endl;

    // Export freeze frame pixel data
    {
        std::filesystem::path converedValuesOutputPath = outputDirectoryPath;
        converedValuesOutputPath.append("Converged.dat");

        std::ofstream convergedValuesOutputStream(converedValuesOutputPath.string());
        if (!convergedValuesOutputStream.is_open())
        {
            std::cout << "Failed to create converged values outfile" << std::endl;
            exit(1);
        }

        convergedValuesOutputStream << averagedResult << std::endl;
    }

    return 0;
}
