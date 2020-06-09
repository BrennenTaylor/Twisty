#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"
#include "StartEndBootstrapper.h"


#include <FMath/Vector3.h>

#include "fmt/format.h"
#include "fmt/ostream.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>

using namespace twisty;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        fmt::print("Call as: example.exe numCurvesToGenerate numSegmentsPerCurve");
    }

    uint32_t numCurvesToGen = std::stoi(argv[1]);
    uint32_t numSegments = std::stoi(argv[2]);

    // Bootstrap method
    const Range defaultBounds = {-1.0f, 1.0f};
    const Farlor::Vector3 emitterPos{0.0f, 0.0f, 0.0f};
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

    const Farlor::Vector3 recieverPos{10.0f, 0.0f, 0.0f};
    const Farlor::Vector3 recieverDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

    const Range arclengthRange = {10.0f, 30.0f};

    // This is the range we want to meet
    // The range of actual curvature/torsion * ds is below
    Range kdsRange = {0.0f, 2.0f};
    Range tdsRange = {-1.0f, 1.0f};

    std::vector<std::unique_ptr<Curve>> curves;
    for (uint32_t i = 0; i < numCurvesToGen; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        StartEndBootstrapper bootstrapper(emitterPos, emitterDir, recieverPos, recieverDir, arclengthRange, 0);
        std::unique_ptr<Curve> upCurve = bootstrapper.CreateCurve(numSegments);
        auto end = std::chrono::high_resolution_clock::now();
        auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto timeSec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        fmt::print("Curve {} generated, {} ms, {} seconds\n", i, timeMs, timeSec);

        curves.push_back(std::move(upCurve));
    }

    fmt::print("We generated {} curves.\n", curves.size());
    // TODO: Show how we can take the curve batches and generate a final weight.
    twisty::BigFloat totalCurveWeight = 0.0f;
    for (auto& upCurve : curves)
    {
        const double mu = 0.1;
        const uint32_t numStepsInt = 2000;
        const double minBound = 0.0;
        const double maxBound = 100.0;
        const double eps = 0.5f;

        float ds = upCurve->m_arclength / 200.0f;
        float scatter = 0.08f / ds;

        PathSpaceUtils::RegularizedIntegral regIntEvaluator(mu, numStepsInt, minBound, maxBound, eps);
        twisty::BigFloat pathWeight = PathSpaceUtils::WeightPath((*upCurve), [](Farlor::Vector3 pos) -> float { return 0.0f; }, [scatter](Farlor::Vector3 pos) -> float { return scatter; }, regIntEvaluator);
        fmt::print("\tPath Weight: {}\n", pathWeight);
        totalCurveWeight += pathWeight;
    }
    totalCurveWeight /= curves.size();
    fmt::print("Weighted (basic avg) path weights: {}\n", totalCurveWeight);

    return 0;
}