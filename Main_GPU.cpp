#include "ExperimentRunnerGpu.h"
#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"
#include "TwistyYamlUtils.h"

#include <FMath/Vector3.h>

#include <fmt/format.h>

#include <iostream>
#include <memory>

using namespace twisty;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        fmt::print("Call as: {} numPathsToGenerate experimentName", argv[0]);
        return false;
    }

    // Only set the experiment name once
    std::string experimentName(argv[2]);

    // Bootstrap method
    const Range defaultBounds = { -1.0f, 1.0f };
    const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    RayGeometry rayEmitter(emitterStart, emitterDir);

    const Farlor::Vector3 recieverPos{ 10.0f, 0.0f, 0.0f };
    const float recieverRadius = 1.0f;
    const float recieverFov = TwistyPi;
    SphereGeometry sphereReciever(recieverPos, recieverRadius, recieverFov);

    const Range arclengthRange = { 10.0f, 30.0f };

    GeometryBootstrapper bootstrapper(rayEmitter, sphereReciever, arclengthRange, 0);

    uint32_t numPathsToSkip = 0;
    uint32_t numPathsToGenerate = std::stoi(argv[1]);

    // This is the range we want to meet
    // The range of actual curvature/torsion * ds is below
    Range kdsRange = { 0.0f, 2.0f };
    Range tdsRange = { -1.0f, 1.0f };

    const uint32_t pathBatchSize = 1000;

    ExperimentRunner::ExperimentParameters experimentParams;
    experimentParams.numPathsInExperiment = numPathsToGenerate;
    experimentParams.exportGeneratedCurves = true;
    experimentParams.numSegmentsPerCurve = 200;
    experimentParams.maximumBootstrapCurveError = 0.1f;
    experimentParams.experimentName = experimentName;
    experimentParams.maxPathBatchSize = 100000;

    ExperimentRunnerGpu runner(experimentParams, bootstrapper, pathBatchSize, kdsRange, tdsRange);

    bool result = runner.Setup();
    if (!result)
    {
        runner.Shutdown();
        std::cout << "Failed to setup experiment runner." << std::endl;
        return 1;
    }

    auto experimentResults = runner.RunExperiment();

    // Retrieve Data we want from experiment

    runner.Shutdown();

    return 0;
}