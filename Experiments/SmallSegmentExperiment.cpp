#define _USE_MATH_DEFINES
#include <cmath>

#include "FullExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

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

    const float targetArclength = 12.0;
    
    std::stringstream innerBlockSS;
    innerBlockSS << "<Converged Value>";
    std::cout << innerBlockSS.str() << std::endl;

    const twisty::Range arclengthRange = {targetArclength, targetArclength};

    // This is the range we want to meet
    // The range of actual curvature/torsion * ds is below
    twisty::Range kdsRange = {0.0f, 2.0f};
    twisty::Range tdsRange = {-1.0f, 1.0f};



    const uint32_t numThetaSteps = 2;
    const double thetaMin = 0.0;
    const double thetaMax = 2.0 * M_PI;
    const double thetaStepSize = (thetaMax - thetaMin) / (numThetaSteps - 1);

    const uint32_t numSSteps = 5;
    const double sMin = -1.0;
    const double sMax = 1.0;
    const double sStepSize = (sMax - sMin) / (numSSteps - 1);

    boost::multiprecision::cpp_dec_float_100 result = 0.0;

    for (uint32_t thetaIdx = 0; thetaIdx < numThetaSteps; thetaIdx++)
    {
        const double theta = thetaMin + thetaIdx * thetaStepSize;
        for (uint32_t sIdx = 0; sIdx < numSSteps; sIdx++)
        {
            const double s = sMin + sIdx * sStepSize;
            const double ss = sqrt(1.0 - s * s);

            Farlor::Vector3 evalVector = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();
            {
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
                        experimentParams.curvePerturbMethod = twisty::ExperimentRunner::CurvePerturbMethod::SimpleGeometry;
                        experimentParams.curvePurturbSeed = perCurveSeed;
                        experimentParams.rotateInitialSeedCurveRadians = 0.0f;

                        // Use a big mu value
                        experimentParams.weightingParameters.mu = 0.5;
                        experimentParams.weightingParameters.eps = 0.1;
                        experimentParams.weightingParameters.numStepsInt = 2000;
                        experimentParams.weightingParameters.minBound = 0.0;
                        experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;
                        experimentParams.weightingParameters.numCurvatureSteps = 10000;
                        experimentParams.weightingParameters.absorbtion = 0.0;
                        experimentParams.weightingParameters.scatter = 0.9;
                        experimentParams.rotateInitialSeedCurveRadians = 0.0f;

                        const float distanceFromPlane = 10.0f;
                        const Farlor::Vector3 recieverPos = Farlor::Vector3(0.0f, 0.0f, distanceFromPlane);
                        twisty::RayGeometry rayReciever(recieverPos, evalVector);

                        twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, arclengthRange, initialCurveSeed);
                        std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper, kdsRange, tdsRange);
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
                result = averagedResult;

                std::cout << "Theta, s: (" << theta << ", " << s << ")" << std::endl;
                std::cout << "\tEnd Dir: " << evalVector << std::endl;
                std::cout << "\tEval: " << result << std::endl;
            }
        }
    }

    //std::cout << "Converged Value: " << averagedResult << std::endl;

    //// Export freeze frame pixel data
    //{
    //    std::filesystem::path converedValuesOutputPath = outputDirectoryPath;
    //    converedValuesOutputPath.append("Converged.dat");

    //    std::ofstream convergedValuesOutputStream(converedValuesOutputPath.string());
    //    if (!convergedValuesOutputStream.is_open())
    //    {
    //        std::cout << "Failed to create converged values outfile" << std::endl;
    //        exit(1);
    //    }

    //    convergedValuesOutputStream << averagedResult << std::endl;
    //}

    return 0;
}
