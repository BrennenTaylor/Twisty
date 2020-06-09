#include "MathConsts.h"

#include "StartEndBootstrapper.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>
#include <FMath/Matrix3x3.h>
#include <FMath/Matrix4x4.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <assert.h>
#include <cstdint>

//using namespace Farlor;

int main()
{

    // Unit: cm^-1
    const double a = 0.004;
    // Unit: cm^-1
    const double b = 0.1;
    // Distance between emitter and sensor
    // Unit: cm
    const double radius = 30.0;
    // Max arclength
    // Unit: cm
    const double maxArclength = 100.0;

    // Gaussian Width of phase function
    // Unit: ???
    const double mu = 0.5;

    // Tolerance of ???
    // Unit: ???
    const double eps = 0.075;

    // Number of path segments
    //const uint32_t numSegments = 80;
    //const uint32_t numSegments = 120;
    //const uint32_t numSegments = 160;
    const uint32_t numSegments = 200;

    const uint32_t experimentNumberOfPaths = 100;

    // Normalized max value
    // Pulled from beam spread paper
    const double normalizedMaxValue = 0.025;



    // Emitter parameters
    // Center the world on the emitter
    const float angleDegree = 0.0f;
    assert(angleDegree <= 180.0f && angleDegree >= -180.0f);
    const float angleRad = twisty::RadianFromDegree(angleDegree);
    Farlor::Vector3 emitterLocation(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 emitterDirection(1.0f, 0.0f, 0.0f);
    Farlor::Matrix3x3 emitterRotationMatrix(
        Farlor::Vector3(cos(angleRad), 0.0f, sin(angleRad)),
        Farlor::Vector3(0.0f, 1.0f, 0.0f),
        Farlor::Vector3(-sin(angleRad), 0.0f, cos(angleRad))
    );
    Farlor::Vector3 emitterRotatedDirection = emitterRotationMatrix * emitterDirection;

    // Receiever Parameters
    Farlor::Vector3 recieverLocation = emitterLocation + emitterDirection * radius;
    // For now, we use the vector between the two locations as the normal
    Farlor::Vector3 recieverDirection = emitterDirection;


    // Ok, lets actually run the experiment.
    // Lets do this in parallel
    twisty::Range arclengthRange = {radius, maxArclength};
    uint32_t failures = 0;
    

    const uint32_t numStepsInt = 2000;
    const double minBound = 0.0;
    const double maxBound = 100.0;

    twisty::PathSpaceUtils::RegularizedIntegral regIntEvaluator(mu, numStepsInt, minBound, maxBound, eps);
    
    twisty::BigFloat totalCurveWeight = 0.0f;
    for (uint32_t pathIdx = 0; pathIdx < experimentNumberOfPaths + failures; ++pathIdx)
    {
        twisty::StartEndBootstrapper bezierBootstrapper(emitterLocation, emitterRotatedDirection,
            recieverLocation, recieverDirection, arclengthRange, 0);
        auto upGeneratedCurve = bezierBootstrapper.CreateCurve(numSegments);
        if (!upGeneratedCurve)
        {
            fmt::print("Error generating curve {}", pathIdx);
            failures++;
            continue;
        }

        float ds = upGeneratedCurve->m_arclength / 200.0f;
        float scatter = 0.08f / ds;

        twisty::BigFloat pathWeight = twisty::PathSpaceUtils::WeightPath((*upGeneratedCurve),
            [](Farlor::Vector3 pos) -> float { return 0.0f; },
            [scatter](Farlor::Vector3 pos) -> float { return scatter; },
            regIntEvaluator
        );
        fmt::print("\tPath {} Weight: {}\n", pathIdx, pathWeight);
        totalCurveWeight += pathWeight;
    }
    // How do we normalize this data?
    totalCurveWeight /= experimentNumberOfPaths;
    fmt::print("Weighted (basic avg) path weights: {}\n", totalCurveWeight);
}