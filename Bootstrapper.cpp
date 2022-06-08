#include "Bootstrapper.h"

#include "FMath/Vector3.h"
#include "PerturbUtils.h"

#include <assert.h>
#include <cmath>
#include <ctime>
#include <random>

#define DetailedCurveGen

namespace twisty {
Bootstrapper::RayGeometry::RayGeometry(Farlor::Vector3 start, Farlor::Vector3 dir)
    : m_pos(start)
    , m_dir(dir)
{
}

Bootstrapper::Geometry::SampleRay Bootstrapper::RayGeometry::GetSampleRay() const
{
    return SampleRay { m_pos, m_dir };
}

Bootstrapper::SphereGeometry::SphereGeometry(Farlor::Vector3 pos, float radius, float fov)
    : m_pos { pos }
    , m_radius { radius }
    , m_fov { fov }
{
}

static Farlor::Vector3 SampleUnitSphere(const float rand0, const float rand1)
{
    float theta = 2.0f * M_PI * rand0;
    float phi = std::acos(1.0f - 2.0f * rand1);
    float x = std::sin(phi) * std::cos(theta);
    float y = std::sin(phi) * std::sin(theta);
    float z = std::cos(phi);
    Farlor::Vector3 normal(x, y, z);
    return normal.Normalized();
}

Bootstrapper::Geometry::SampleRay Bootstrapper::SphereGeometry::GetSampleRay() const
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0f, 1.0f);
    float rand0 = static_cast<float>(dist(gen));
    float rand1 = static_cast<float>(dist(gen));
    Farlor::Vector3 sphereSample = SampleUnitSphere(rand0, rand1);
    return SampleRay { m_pos, sphereSample.Normalized() };
}

// Bootstrapper::Bootstrapper()
// {
// }

Bootstrapper::Bootstrapper(const twisty::PerturbUtils::BoundaryConditions &problemGeometry)
    : m_experimentGeometry(problemGeometry)
{
}

Bootstrapper::~Bootstrapper() { }

Farlor::Vector3 Bootstrapper::GetStartPosition() const { return m_experimentGeometry.m_startPos; }
Farlor::Vector3 Bootstrapper::GetStartNormal() const { return m_experimentGeometry.m_startDir; }
Farlor::Vector3 Bootstrapper::GetTargetPosition() const { return m_experimentGeometry.m_endPos; }
Farlor::Vector3 Bootstrapper::GetTargetNormal() const { return m_experimentGeometry.m_endDir; }

float Bootstrapper::CalculateMinimumArclength(
      const uint32_t numSegments, const Farlor::Vector3 &startPos, const Farlor::Vector3 &endPos)
{
    const float d = (endPos - startPos).Magnitude();
    assert(d > 0.0f);
    assert(numSegments > 0);

    return d / (1.0f - (4.0f / numSegments));
}

std::unique_ptr<Curve> Bootstrapper::CreateCurveGeometricSafe(
      uint32_t numSegments, float targetArclength) const
{
    // TODO: Add assertion for minimal number of segments!

    std::cout << "\nBegin generating curve" << std::endl;
    const float arclength = targetArclength;
    const float ds = arclength / numSegments;

    // If we have an odd number of segments, we want to place two segments right
    // at the beginning
    // TODO: The second segment really should be flexible in its placement for
    // allowing the most environment configurations
    //       however, its good enough for now.

    std::cout << "Generating curve: " << std::endl;
    std::cout << "\tStart Pos: " << m_experimentGeometry.m_startPos << std::endl;
    std::cout << "\tStart Dir: " << m_experimentGeometry.m_startDir << std::endl;
    std::cout << "\tEnd Pos: " << m_experimentGeometry.m_endPos << std::endl;
    std::cout << "\tEnd Dir: " << m_experimentGeometry.m_endDir << std::endl;
    std::cout << "\tArclength: " << targetArclength << std::endl;

    bool evenNumberOfSegments = (numSegments % 2) == 0;

    // In the all cases, we place two segments initially
    const Farlor::Vector3 x_sp1
          = m_experimentGeometry.m_startPos + ds * m_experimentGeometry.m_startDir;
    const Farlor::Vector3 x_em1
          = m_experimentGeometry.m_endPos - ds * m_experimentGeometry.m_endDir;
    const Farlor::Vector3 x_s
          = evenNumberOfSegments ? x_sp1 : x_sp1 + (x_em1 - x_sp1).Normalized() * ds;

    const Farlor::Vector3 x_p = (x_s + x_em1) * 0.5;
    const Farlor::Vector3 lineUnitDir = (x_em1 - x_s).Normalized();

    Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
    if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
        otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
    }

    const Farlor::Vector3 normalToLine = lineUnitDir.Cross(otherCrossVec).Normalized();

    // TODO: Add assertion that remaining segments can fill gap

    // Subtract off segments which are already accounted for, 3 if odd and 2 if
    // even
    const int remainingSegmentCount = evenNumberOfSegments ? numSegments - 2 : numSegments - 3;
    // We should have an even number of segments remaining
    assert((remainingSegmentCount % 2) == 0);
    const float hypot = remainingSegmentCount * 0.5f * ds;
    const float D_2 = (x_em1 - x_s).Magnitude() * 0.5f;

    if (D_2 > hypot) {
        std::cout << "Error, we have an invalid environment parameter. No possible "
                     "curve fits constraints"
                  << std::endl;
        return nullptr;
    }

    const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
    const Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

    std::unique_ptr<Curve> upGeneratedCurve = std::make_unique<Curve>(numSegments);
    upGeneratedCurve->m_arclength = arclength;
    upGeneratedCurve->m_numSegments = numSegments;
    upGeneratedCurve->m_segmentLength = ds;
    upGeneratedCurve->m_basePos = GetStartPosition();
    upGeneratedCurve->m_baseTangent = GetStartNormal();
    upGeneratedCurve->m_targetPos = GetTargetPosition();
    upGeneratedCurve->m_targetTangent = GetTargetNormal();

    int remainingSegmentCountDiv2 = remainingSegmentCount / 2;

    // First positions
    int xsPos = 0;
    if (evenNumberOfSegments) {
        upGeneratedCurve->m_positions[0] = m_experimentGeometry.m_startPos;
        upGeneratedCurve->m_positions[1] = x_s;
        xsPos = 1;
    } else {
        upGeneratedCurve->m_positions[0] = m_experimentGeometry.m_startPos;
        upGeneratedCurve->m_positions[1] = x_sp1;
        upGeneratedCurve->m_positions[2] = x_s;
        xsPos = 2;
    }

    // Lock the last segment
    {
        upGeneratedCurve->m_positions[numSegments - 1] = x_em1;
        upGeneratedCurve->m_positions[numSegments] = m_experimentGeometry.m_endPos;
    }

    const Farlor::Vector3 leftDir = (x_t - x_s).Normalized();
    for (int leftIdx = 1; leftIdx <= remainingSegmentCountDiv2; ++leftIdx) {
        // We want leftIdx 0 to be 1 step away, thus the + 1
        upGeneratedCurve->m_positions[leftIdx + xsPos] = x_s + leftIdx * ds * leftDir;
    }

    const Farlor::Vector3 rightDir = (x_em1 - x_t).Normalized();
    for (int rightIdx = 1; rightIdx <= remainingSegmentCountDiv2; ++rightIdx) {
        upGeneratedCurve->m_positions[rightIdx + xsPos + remainingSegmentCountDiv2]
              = x_t + rightIdx * ds * rightDir;
    }

    twisty::PerturbUtils::BoundaryConditions boundaryConditions
          = upGeneratedCurve->GetBoundaryConditions();
    // boundaryConditions.arclength = upGeneratedCurve->m_arclength;
    // boundaryConditions.m_startPos = upGeneratedCurve->m_basePos;
    // boundaryConditions.m_startDir = upGeneratedCurve->m_baseTangent;
    // boundaryConditions.m_endPos = upGeneratedCurve->m_targetPos;
    // boundaryConditions.m_endDir = upGeneratedCurve->m_targetTangent;

    std::cout << "Boundary conditions in bootstrapper: " << std::endl;
    std::cout << "\tStart pos and Dir: " << boundaryConditions.m_startPos << ", "
              << boundaryConditions.m_startDir << std::endl;
    std::cout << "\tEnd pos and Dir: " << boundaryConditions.m_endPos << ", "
              << boundaryConditions.m_endDir << std::endl;

    // No curvature update
    twisty::PerturbUtils::UpdateTangentsFromPos(upGeneratedCurve->m_positions.data(),
          upGeneratedCurve->m_tangents.data(), upGeneratedCurve->m_numSegments, boundaryConditions);

    double totalDistance = 0.0;
    // Calculate the actual length of the segment based curve
    for (uint32_t i = 0; i < upGeneratedCurve->m_numSegments - 1; ++i) {
        totalDistance += (upGeneratedCurve->m_positions[i + 1] - upGeneratedCurve->m_positions[i])
                               .Magnitude();
    }
    // And handle last segment
    {
        totalDistance += (upGeneratedCurve->m_targetPos
              - upGeneratedCurve->m_positions[upGeneratedCurve->m_numSegments - 1])
                               .Magnitude();
    }
    std::cout << "\tSegment Curve Distance: " << totalDistance << std::endl;
    std::cout << "\tTarget Arclength: " << upGeneratedCurve->m_arclength << std::endl;

    return upGeneratedCurve;
}

std::unique_ptr<Curve> Bootstrapper::CreateCurve(
      uint32_t numSegments, float targetArclength, uint32_t generationSeed) const
{
    std::cout << "Error: Dont use this version of the curve creator" << std::endl;
    assert(false);
    std::cout << "\nBegin generating curve" << std::endl;
    const uint32_t bootstrapSeed
          = (generationSeed != 0) ? generationSeed : static_cast<uint32_t>(time(0));
    std::mt19937_64 randomGen(bootstrapSeed);

    std::cout << "\tBoundary conditions: " << std::endl;
    std::cout << "\tStart Pos: " << m_experimentGeometry.m_startPos << std::endl;
    std::cout << "\tStart Dir: " << m_experimentGeometry.m_startDir << std::endl;
    std::cout << "\tEnd Pos: " << m_experimentGeometry.m_endPos << std::endl;
    std::cout << "\tEnd Dir: " << m_experimentGeometry.m_endDir << std::endl;
    std::cout << "\tTarget Arclength: " << targetArclength << std::endl;

    const float requestedArclength = targetArclength;

    const float minL2 = 0.0f;
    const float maxL2 = std::pow(10.0f, 5.0);

    const Farlor::Vector3 x1x0 = m_experimentGeometry.m_endPos - m_experimentGeometry.m_startPos;
    const float length = x1x0.Magnitude();

#if defined(DetailedCurveGen)
    std::cout << "\tDistance btw start, end pts: " << length << std::endl;
#endif

    // Merrsine Twister
    // NOTE: project value of 0.5f is pulled from the disertation
    const float project = 0.5f;
    std::uniform_real_distribution<float> dist(0.0f, project * length);

    float l0 = static_cast<float>(dist(randomGen));
    float l1 = static_cast<float>(dist(randomGen));

#if defined(DetailedCurveGen)
    std::cout << "\t(l0, l1): (" << l0 << ", " << l1 << ")" << std::endl;
#endif

    BezierCurve5 newBezierCurve;
    newBezierCurve.m_controlPts[0] = m_experimentGeometry.m_startPos;
    newBezierCurve.m_controlPts[1]
          = m_experimentGeometry.m_startPos + l0 * m_experimentGeometry.m_startDir;
    newBezierCurve.m_controlPts[2]
          = (m_experimentGeometry.m_startPos + m_experimentGeometry.m_endPos) * 0.5f;
    newBezierCurve.m_controlPts[3]
          = m_experimentGeometry.m_endPos - l1 * m_experimentGeometry.m_endDir;
    newBezierCurve.m_controlPts[4] = m_experimentGeometry.m_endPos;

    // This is the actual smallest arclength that we are able to generate
    const float shortestArclengthPossible = newBezierCurve.CalculateArclength(0.0f, 1.0f);

#if defined(DetailedCurveGen)
    std::cout << "\tShortest Arclength Before Moving cp2: " << shortestArclengthPossible
              << std::endl;
#endif

    std::uniform_real_distribution<float> uniformZeroToOne(0.0f, 1.0f);
    const float e0 = uniformZeroToOne(randomGen);
    const float e1 = uniformZeroToOne(randomGen);
    Farlor::Vector3 n2 = SampleUnitSphere(e0, e1);

#if defined(DetailedCurveGen)
    std::cout << "\tn2: " << n2 << std::endl;
#endif

    // Captures by references, so the actual bezier curve is temporarily modified
    auto TestL2Arclength = [&](float testL2) -> float {
        const Farlor::Vector3 previous = newBezierCurve.m_controlPts[2];
        newBezierCurve.m_controlPts[2] = previous + n2 * testL2;

        const float minVal = 0.0f;
        const float maxVal = 1.0f;
        const float arclength = newBezierCurve.CalculateArclength(minVal, maxVal);
        newBezierCurve.m_controlPts[2] = previous;
        return arclength;
    };

    const float maxL2ArcLength = TestL2Arclength(maxL2);

    const float minArclengthScaleFactor = 1.01f;
    targetArclength = std::max(static_cast<float>(targetArclength), shortestArclengthPossible)
          * minArclengthScaleFactor;
    std::cout << "Selected arclength to target: " << targetArclength << std::endl;

    // Ok, we want to find target_l2 such that
    // TestL2ArcLength(target_l2) - targetArcLength is minimized.
    // target_l2 search space is [minl2, maxL2]
    // Can we guarantee that we have a value on that range.
    const uint32_t maxNumberOfIterations = 300;
    float a = minL2;
    float b = maxL2;
    uint32_t currentIterationCount = 0;

    const float errorThresh = 1e-8f;
    const float distThresh = 1e-10f;
    float minF = TestL2Arclength(a) - targetArclength;
    float maxF = TestL2Arclength(b) - targetArclength;

    if ((minF * maxF) > 0.0f) {
        // Midpoint method will fail
        std::cout << "Error: Cannot perform midpoint method" << std::endl;
        return nullptr;
    }

#if defined(DetailedCurveGen)
    std::cout << "\tPerform Search For Bezier" << std::endl;
#endif

    float guessVal = (a + b) / 2.0f;
    while (currentIterationCount < maxNumberOfIterations) {
        float aVal = TestL2Arclength(a) - targetArclength;
        float bVal = TestL2Arclength(b) - targetArclength;
        float guessArclength = TestL2Arclength(guessVal);
        float guessError = guessArclength - targetArclength;
        if (std::abs(guessError) < errorThresh) {
            // Guess works
            break;
        }

        if (std::abs(a - b) < distThresh) {
            // a and b close enough, we terminate
            break;
        }

        if ((aVal * guessError) < 0.0f) {
            b = guessVal;
        } else {
            a = guessVal;
        }
        guessVal = (a + b) / 2.0f;
        currentIterationCount++;
    }
    // std::cout << "\tCurrent Iteration Count: " << currentIterationCount <<
    // std::endl; printf("\tFinal L2: %.10f\n", guessVal); Use our guess
    newBezierCurve.m_controlPts[2] = newBezierCurve.m_controlPts[2] + n2 * guessVal;
    const float minBound = 0.0f;
    const float maxBound = 1.0f;
    const float finalArclength = newBezierCurve.CalculateArclength(minBound, maxBound);

#if defined(DetailedCurveGen)
    std::cout << "\tFinal arc length: " << finalArclength << std::endl;
    std::cout << "\tTarget arc length: " << targetArclength << std::endl;
    std::cout << "\tFinal bezier error: " << std::abs(finalArclength - targetArclength)
              << std::endl;
#endif

    return ToDiscreteFSCurve(numSegments, newBezierCurve);
}

std::unique_ptr<Curve> Bootstrapper::ToDiscreteFSCurve(
      uint32_t numSegments, BezierCurve5 &bezierCurve) const
{
    const float minT = 0.0f;
    const float maxT = 1.0f;
    const float initialArclength = bezierCurve.CalculateArclength(minT, maxT);
    std::cout << "Initial Curve Length: " << initialArclength << std::endl;
    // ds is going to be in arclength parameterization
    const float ds = initialArclength / numSegments;

    std::unique_ptr<Curve> upGeneratedCurve = std::make_unique<Curve>(numSegments);
    upGeneratedCurve->m_arclength = initialArclength;
    upGeneratedCurve->m_numSegments = numSegments;
    upGeneratedCurve->m_basePos = GetStartPosition();
    upGeneratedCurve->m_baseTangent = GetStartNormal();
    upGeneratedCurve->m_targetPos = GetTargetPosition();
    upGeneratedCurve->m_targetTangent = GetTargetNormal();

    // Initialize base position and frame
    Farlor::Vector3 x_0 = upGeneratedCurve->m_basePos;

    const uint32_t numSteps = 10000;
    // Actually cache the values for the tvalue lookup in the next steps
    bezierCurve.CacheArclength(numSteps);

    const float halfDS = ds * 1.0f;

    static int bezierSegIdx = 0;
    upGeneratedCurve->m_segmentLength = ds;
    // First Position. Seeded by problem
    {
        upGeneratedCurve->m_positions[0] = m_experimentGeometry.m_startPos;
        upGeneratedCurve->m_tangents[0] = m_experimentGeometry.m_startDir;
        bezierSegIdx++;
    }
    // Second position, fixed by construction
    {
        upGeneratedCurve->m_positions[1] = m_experimentGeometry.m_startPos
              + m_experimentGeometry.m_startDir * upGeneratedCurve->m_segmentLength;
    }
    // Last position, defined by problem
    {
        upGeneratedCurve->m_positions[numSegments] = m_experimentGeometry.m_endPos;
    }

    // All other segments than first one
    for (uint32_t i = 2; i < numSegments; ++i) {
        // Target Arclength
        float targetArclength = (i)*ds * 0.9999f;

        const uint32_t maxNumberOfIterations = 1000;

        float a = minT;
        float b = maxT;
        uint32_t currentIterationCount = 0;

        const float errorThresh = 1e-4f;
        const float distThresh = 1e-4f;

        float minF = bezierCurve.CalculateArclengthAlreadyCached(minT, a) - targetArclength;
        float maxF = bezierCurve.CalculateArclengthAlreadyCached(minT, b) - targetArclength;

        if ((minF * maxF) > 0.0f) {
            // Midpoint method will fail
            std::cout << "Error: Cannot perform midpoint method" << std::endl;
            return nullptr;
        }

        float guessVal = (a + b) / 2.0f;
        while (currentIterationCount < maxNumberOfIterations) {
            float aVal = bezierCurve.CalculateArclengthAlreadyCached(minT, a) - targetArclength;
            float bVal = bezierCurve.CalculateArclengthAlreadyCached(minT, b) - targetArclength;
            float guessF
                  = bezierCurve.CalculateArclengthAlreadyCached(minT, guessVal) - targetArclength;
            if (std::abs(guessF) < errorThresh) {
                // Guess works
                break;
            }

            if (std::abs(a - b) < distThresh) {
                // a and b close enough, we terminate
                break;
            }

            if ((aVal * guessF) < 0.0f) {
                b = guessVal;
            } else {
                a = guessVal;
            }
            guessVal = (a + b) / 2.0f;
            currentIterationCount++;
        }

        // Lets get all segment information
        // Sample this from the curve
        Farlor::Vector3 segmentPosition = bezierCurve.GetPosition(guessVal);
        upGeneratedCurve->m_positions[i] = segmentPosition;
        bezierSegIdx++;
    }

    // Caclulate the cached tangents here
    for (uint32_t i = 0; i < upGeneratedCurve->m_numSegments; ++i) {
        Farlor::Vector3 diff
              = (upGeneratedCurve->m_positions[i + 1] - upGeneratedCurve->m_positions[i]);
        upGeneratedCurve->m_tangents[i] = diff.Normalized();
    }

    // Write final tangent here
    {
        upGeneratedCurve->m_tangents[numSegments] = m_experimentGeometry.m_endDir.Normalized();
    }

    // All but the last segment. We do that one manually
    for (uint32_t i = 0; i < numSegments; ++i) {
        auto &tanLeft = upGeneratedCurve->m_tangents[i];
        auto &tanRight = upGeneratedCurve->m_tangents[i + 1];

        {
            float curvature = ((tanRight - tanLeft) * (1.0f / ds)).Magnitude();
            upGeneratedCurve->m_curvatures[i] = curvature;
        }
    }

    float totalDistance = 0.0f;
    // Calculate the actual length of the segment based curve
    for (uint32_t i = 0; i < upGeneratedCurve->m_numSegments - 1; ++i) {
        totalDistance += (upGeneratedCurve->m_positions[i + 1] - upGeneratedCurve->m_positions[i])
                               .Magnitude();
    }
    // And handle last segment
    {
        totalDistance += (upGeneratedCurve->m_targetPos
              - upGeneratedCurve->m_positions[upGeneratedCurve->m_numSegments - 1])
                               .Magnitude();
    }
    std::cout << "\tSegment Curve Distance: " << totalDistance << std::endl;
    std::cout << "\tTarget Arclength: " << upGeneratedCurve->m_arclength << std::endl;

    return upGeneratedCurve;
}
}  // namespace twisty