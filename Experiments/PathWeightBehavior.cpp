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
    if (argc < 2)
    {
        fmt::print("Call as: {} numSegmentsPerCurve", argv[0]);
    }

    uint32_t numSegments = std::stoi(argv[1]);

    // Bootstrap method
    const Range defaultBounds = {-1.0f, 1.0f};
    const Farlor::Vector3 emitterPos{0.0f, 0.0f, 0.0f};
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

    const Farlor::Vector3 recieverPos{10.0f, 0.0f, 0.0f};
    const Farlor::Vector3 recieverDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();

    const float arclength = 10.0f;

    // This is the range we want to meet
    // The range of actual curvature/torsion * ds is below
    Range kdsRange = {0.0f, 2.0f};
    Range tdsRange = {-1.0f, 1.0f};

    // Number of segments
    twisty::Curve straightCurve(200);
    straightCurve.m_arclength = arclength;
    straightCurve.m_basePos = emitterPos;
    straightCurve.m_baseTangent = emitterDir;
    straightCurve.m_baseNormal = Farlor::Vector3(0.0f, 1.0f, 0.0f);
    straightCurve.m_baseBinormal = straightCurve.m_baseTangent.Cross(straightCurve.m_baseNormal);
    straightCurve.m_targetPos = recieverPos;
    straightCurve.m_targetTangent = recieverDir;
    for (uint32_t i = 0; i < straightCurve.m_numSegments; ++i)
    {
        straightCurve.m_segments[i].m_curvature = 0.0f;
        straightCurve.m_segments[i].m_torsion = 0.0f;
        straightCurve.m_segments[i].m_length = straightCurve.m_arclength / straightCurve.m_numSegments;
        straightCurve.m_segments[i].UpdateRotation();
    }

    const double mu = 0.1;
    const uint32_t numStepsInt = 2000;
    const double minBound = 0.0;
    const double maxBound = 100.0;
    const double eps = 0.5f;

    float ds = straightCurve.m_arclength / 200.0f;

    PathSpaceUtils::RegularizedIntegral regIntEvaluator(mu, numStepsInt, minBound, maxBound, eps);
    twisty::BigFloat pathWeight = PathSpaceUtils::WeightPath(straightCurve, [](Farlor::Vector3 pos) -> float { return 0.0f; }, [](Farlor::Vector3 pos) -> float { return 0.0f; }, regIntEvaluator);
    fmt::print("\tPath Weight: {}\n", pathWeight);

    return 0;
}