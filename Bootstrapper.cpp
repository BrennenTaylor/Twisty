#include "Bootstrapper.h"

#include "PerturbUtils.h"
#include "Sample.h"

#include <assert.h>
#include <cmath>
#include <ctime>
#include <random>

#define DetailedCurveGen

namespace twisty
{
    Bootstrapper::Bootstrapper(Range arclengthRange, uint32_t randomSeed)
        : m_startPos(0.0f, 0.0f, 0.0f)
        , m_startDir(0.0f, 0.0f, 0.0f)
        , m_endPos(0.0f, 0.0f, 0.0f)
        , m_endDir(0.0f, 0.0f, 0.0f)
        , m_isCached(false)
        , m_upCachedCurve(nullptr)
        , m_cachedTValues()
        , m_cachedSegmentPositions()
        , m_cachedSegmentFrames()
        , m_upCachedBezierInfo(nullptr)
        , m_upCachedBezier(nullptr)
        , m_arclengthRange(arclengthRange)
        , m_gen()
        , m_bootstrapSeed(randomSeed)
    {
        if (randomSeed != 0)
        {
            m_bootstrapSeed = randomSeed;
        }
        else
        {
            m_bootstrapSeed = static_cast<uint64_t>(time(0));
        }

        m_gen = std::mt19937_64(m_bootstrapSeed);
    }

    Bootstrapper::~Bootstrapper()
    {
    }

    void Bootstrapper::Reset()
    {
        BeginReset();

        m_isCached = false;
        m_upCachedCurve = nullptr;
        m_cachedTValues.clear();
        m_cachedSegmentPositions.clear();
        m_cachedSegmentFrames.clear();
        m_upCachedBezierInfo = nullptr;
        m_upCachedBezier = nullptr;

        EndReset();
    }

    uint32_t Bootstrapper::GetBootstrapSeed() const
    {
        return m_bootstrapSeed;
    }

    Farlor::Vector3 Bootstrapper::GetStartPosition() const
    {
        return m_startPos;
    }
    Farlor::Vector3 Bootstrapper::GetStartNormal() const
    {
        return m_startDir;
    }
    Farlor::Vector3 Bootstrapper::GetTargetPosition() const
    {
        return m_endPos;
    }
    Farlor::Vector3 Bootstrapper::GetTargetNormal() const
    {
        return m_endDir;
    }

    std::unique_ptr<Curve> Bootstrapper::CreateCurveGeometricSafe(uint32_t numSegments)
    {
        std::cout << "\nBegin generating curve" << std::endl;

        const double arclength = m_arclengthRange.m_min;
        const double ds = arclength / numSegments;
        const Farlor::Vector3 x_s = m_startPos + ds * m_startDir;
        const Farlor::Vector3 x_e = m_endPos -= ds * m_endDir;
        const Farlor::Vector3 x_p = (x_s + x_e) * 0.5;
        const Farlor::Vector3 lineUnitDir = (x_e - x_s).Normalized();
        
        Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
        if (lineUnitDir == otherCrossVec)
        {
            otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
        }
        
        const Farlor::Vector3 normalToLine = lineUnitDir.Cross(otherCrossVec).Normalized();
        const double hypot = (numSegments - 2) * 0.5 * ds;
        const double D_2 = (x_e - x_s).Magnitude() / 2.0;
        const double distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
        const Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

        m_upCachedCurve = std::make_unique<Curve>(numSegments);
        m_upCachedCurve->m_arclength = arclength;
        m_upCachedCurve->m_numSegments = numSegments;
        m_upCachedCurve->m_segmentLength = ds;
        m_upCachedCurve->m_basePos = GetStartPosition();
        m_upCachedCurve->m_baseTangent = GetStartNormal();
        m_upCachedCurve->m_targetPos = GetTargetPosition();
        m_upCachedCurve->m_targetTangent = GetTargetNormal();
        
        // First 2 positions
        // Defines first segment
        {
            m_upCachedCurve->m_positions[0] = m_startPos;
            m_upCachedCurve->m_positions[1] = x_s;
        }
        

        // Last 2 positions
        // Defines last segment
        {
            m_upCachedCurve->m_positions[numSegments - 1] = x_e;
            m_upCachedCurve->m_positions[numSegments] = m_endPos;
        }

        // Calculate the left side
        const uint32_t numLeft = (((numSegments + 1) - 4) / 2) + 1;
        const uint32_t numRight = numLeft - 1;
        
        const Farlor::Vector3 leftDir = (x_t - x_s).Normalized();
        for (uint32_t i = 1; i <= numLeft; ++i)
        {
            m_upCachedCurve->m_positions[1 + i] = x_s + i * ds * leftDir;
        }

        const Farlor::Vector3 rightDir = (x_e - x_t).Normalized();
        for (uint32_t i = 1; i <= numRight; ++i)
        {
            m_upCachedCurve->m_positions[numLeft + i] = x_t + i * ds * rightDir;
        }
        
        twisty::PerturbUtils::BoundrayConditions boundaryConditions;
        boundaryConditions.arclength = m_upCachedCurve->m_arclength;
        boundaryConditions.m_startPos = m_upCachedCurve->m_basePos;
        boundaryConditions.m_startDir = m_upCachedCurve->m_baseTangent;
        boundaryConditions.m_endPos = m_upCachedCurve->m_targetPos;
        boundaryConditions.m_endDir = m_upCachedCurve->m_targetTangent;
        twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(m_upCachedCurve->m_positions.data(), m_upCachedCurve->m_tangents.data(),
            m_upCachedCurve->m_curvatures.data(), m_upCachedCurve->m_numSegments, boundaryConditions);


        float totalDistance = 0.0f;
        // Calculate the actual length of the segment based curve
        for (uint32_t i = 0; i < m_upCachedCurve->m_numSegments - 1; ++i)
        {
            totalDistance += (m_upCachedCurve->m_positions[i + 1] - m_upCachedCurve->m_positions[i]).Magnitude();
        }
        // And handle last segment
        {
            totalDistance += (m_upCachedCurve->m_targetPos - m_upCachedCurve->m_positions[m_upCachedCurve->m_numSegments - 1]).Magnitude();
        }
        std::cout << "\tSegment Curve Distance: " << totalDistance << std::endl;
        std::cout << "\tTarget Arclength: " << m_upCachedCurve->m_arclength << std::endl;

        // We need to return a copy
        m_isCached = true;

        auto upReturnCurve = std::make_unique<Curve>(m_upCachedCurve->m_numSegments);
        *upReturnCurve = *m_upCachedCurve;
        return upReturnCurve;
    }

    
    std::unique_ptr<Curve> Bootstrapper::CreateCurve(uint32_t numSegments)
    {
        std::cout << "\nBegin generating curve" << std::endl;

        std::cout << "\tBoundary conditions: " << std::endl;
        std::cout << "\tStart Pos: " << m_startPos << std::endl;
        std::cout << "\tStart Dir: " << m_startDir << std::endl;
        std::cout << "\tEnd Pos: " << m_endPos << std::endl;
        std::cout << "\tEnd Dir: " << m_endDir << std::endl;
        std::cout << "\tMin Arclength Range: " << m_arclengthRange.m_min << std::endl;
        std::cout << "\tMax Arclength Range: " << m_arclengthRange.m_max << std::endl;

        const float requestedArclength = m_arclengthRange.m_min;

        const float minL2 = 0.0f;
        const float maxL2 = pow(10.0f, 5);

        const Farlor::Vector3 x1x0 = m_endPos - m_startPos;
        const float length = x1x0.Magnitude();

#if defined(DetailedCurveGen)
        std::cout << "\tDistance btw start, end pts: " << length << std::endl;
#endif

        // Merrsine Twister
        // NOTE: project value of 0.5f is pulled from the disertation
        const float project = 0.5f;
        std::uniform_real_distribution<float> dist(0.0f, project * length);

        float l0 = static_cast<float>(dist(m_gen));
        float l1 = static_cast<float>(dist(m_gen));

#if defined(DetailedCurveGen)
        std::cout << "\t(l0, l1): (" << l0 << ", " << l1 << ")" << std::endl;
#endif

        m_upCachedBezier = std::make_unique<BezierCurve5>();
        m_upCachedBezier->m_controlPts[0] = m_startPos;
        m_upCachedBezier->m_controlPts[1] = m_startPos + l0 * m_startDir;
        m_upCachedBezier->m_controlPts[2] = (m_startPos + m_endPos) * 0.5f;
        m_upCachedBezier->m_controlPts[3] = m_endPos - l1 * m_endDir;
        m_upCachedBezier->m_controlPts[4] = m_endPos;
        
        // This is the actual smallest arclength that we are able to generate
        const float shortestArclengthPossible = m_upCachedBezier->CalculateArclength(0.0f, 1.0f);


#if defined(DetailedCurveGen)
        std::cout << "\tShortest Arclength Before Moving cp2: " << shortestArclengthPossible << std::endl;
#endif


        std::uniform_real_distribution<float> uniformZeroToOne(0.0f, 1.0f);
        const float e0 = uniformZeroToOne(m_gen);
        const float e1 = uniformZeroToOne(m_gen);
        Farlor::Vector3 n2 = Sample::SampleUnitSphere(e0, e1);
        
#if defined(DetailedCurveGen)
        std::cout << "\tn2: " << n2 << std::endl;
#endif

        // Captures by references, so the actual bezier curve is temporarily modified
        auto TestL2Arclength = [&](float testL2) -> float
        {
            const Farlor::Vector3 previous = m_upCachedBezier->m_controlPts[2];
            m_upCachedBezier->m_controlPts[2] = previous + n2 * testL2;

            const float minVal = 0.0f;
            const float maxVal = 1.0f;
            const float arclength = m_upCachedBezier->CalculateArclength(minVal, maxVal);
            m_upCachedBezier->m_controlPts[2] = previous;
            return arclength;
        };

        // Ok, so we hand an arclength to the bootstrapper. We want to have the closest minimum arclenghth that matches.

        // TODO: Figure out if we need to flip the normal here?
        // We want basically the arc length to never shrink as we move along the normal.
        // If we get this case, we have a problem.

        
        //        float minL2ArcLength = TestL2Arclength(minL2);
//        float maxL2ArcLength = TestL2Arclength(maxL2);
//
//#if defined(DetailedCurveGen)
//        std::cout << "\tminL2ArcLength: " << minL2ArcLength << std::endl;
//        std::cout << "\tmaxL2ArcLength: " << maxL2ArcLength << std::endl;
//#endif
//

        const float maxL2ArcLength = TestL2Arclength(maxL2);

        float minArclength = std::max(static_cast<float>(m_arclengthRange.m_min), shortestArclengthPossible);
        // Require at least a minimum arclength
        float maxArclength = std::max(static_cast<float>(m_arclengthRange.m_max), minArclength);
        maxArclength = std::min(maxArclength, maxL2ArcLength);
//
        assert(minArclength <= maxArclength);
//
        std::uniform_real_distribution<float> arclengthDist(minArclength, maxArclength);
        const float targetArclength = static_cast<float>(arclengthDist(m_gen)) * 1.01;
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

        if ((minF * maxF) > 0.0f)
        {
            // Midpoint method will fail
            std::cout << "Error: Cannot perform midpoint method" << std::endl;
            return nullptr;
        }

#if defined(DetailedCurveGen)
        std::cout << "\tPerform Search For Bezier" << std::endl;
#endif

        float guessVal = (a + b) / 2.0f;
        while (currentIterationCount < maxNumberOfIterations)
        {
            float aVal = TestL2Arclength(a) - targetArclength;
            float bVal = TestL2Arclength(b) - targetArclength;
            float guessArclength = TestL2Arclength(guessVal);
            float guessError = guessArclength - targetArclength;
            //std::cout << "\t\tGuess: " << guessVal << std::endl;
            //std::cout << "\t\tGuess Arclength: " << guessArclength << std::endl;
            //std::cout << "\t\tGuess Error: " << guessError << std::endl;
            if (std::abs(guessError) < errorThresh)
            {
                // Guess works
                break;
            }

            if (std::abs(a - b) < distThresh)
            {
                // a and b close enough, we terminate
                break;
            }

            if ((aVal * guessError) < 0.0f)
            {
                b = guessVal;
            }
            else
            {
                a = guessVal;
            }
            guessVal = (a + b) / 2.0f;
            currentIterationCount++;
        }
        // std::cout << "\tCurrent Iteration Count: " << currentIterationCount << std::endl;
        // printf("\tFinal L2: %.10f\n", guessVal);
        // Use our guess
        m_upCachedBezier->m_controlPts[2] = m_upCachedBezier->m_controlPts[2] + n2 * guessVal;
        const float minBound = 0.0f;
        const float maxBound = 1.0f;
        const float finalArclength = m_upCachedBezier->CalculateArclength(minBound, maxBound);

#if defined(DetailedCurveGen)
        std::cout << "\tFinal arc length: " << finalArclength << std::endl;
        std::cout << "\tTarget arc length: " << targetArclength << std::endl;
        std::cout << "\tFinal bezier error: " << std::abs(finalArclength - targetArclength) << std::endl;
#endif

        // Cached that bezier info for retrieval later
        m_upCachedBezierInfo = std::make_unique<Bootstrapper::BezierInfo>();
        m_upCachedBezierInfo->m_controlPt0 = m_upCachedBezier->m_controlPts[0];
        m_upCachedBezierInfo->m_controlPt1 = m_upCachedBezier->m_controlPts[1];
        m_upCachedBezierInfo->m_controlPt2 = m_upCachedBezier->m_controlPts[2];
        m_upCachedBezierInfo->m_controlPt3 = m_upCachedBezier->m_controlPts[3];
        m_upCachedBezierInfo->m_controlPt4 = m_upCachedBezier->m_controlPts[4];

        return ToDiscreteFSCurve(numSegments, *m_upCachedBezier);
    }
    

    std::unique_ptr<Curve> Bootstrapper::GetCachedCurve()
    {
        if (m_isCached)
        {
            auto upCurve = std::make_unique<Curve>(m_upCachedCurve->m_numSegments);
            *upCurve = *m_upCachedCurve;
            return upCurve;
        }
        return nullptr;
    }

    std::unique_ptr<Curve> Bootstrapper::ToDiscreteFSCurve(uint32_t numSegments, BezierCurve5& bezierCurve)
    {
        // printf("Bezier to Discrete FS Curve info:\n");

        const float minT = 0.0f;
        const float maxT = 1.0f;
        const float initialArclength = bezierCurve.CalculateArclength(minT, maxT);
        std::cout << "Initial Curve Length: " << initialArclength << std::endl;
        // ds is going to be in arclength parameterization
        const float ds = initialArclength / numSegments;

        // printf("\tFull curve arclength: %f\n", fullCurveArcLength);
        // printf("\tds: %f\n", ds);

        m_upCachedCurve = std::make_unique<Curve>(numSegments);
        m_upCachedCurve->m_arclength = initialArclength;
        m_upCachedCurve->m_numSegments = numSegments;
        m_upCachedCurve->m_basePos = GetStartPosition();
        m_upCachedCurve->m_baseTangent = GetStartNormal();
        m_upCachedCurve->m_targetPos = GetTargetPosition();
        m_upCachedCurve->m_targetTangent = GetTargetNormal();

        // Initialize base position and frame
        Farlor::Vector3 x_0 = m_upCachedCurve->m_basePos;

        m_cachedTValues.clear();
        // We always wand the first one
        m_cachedTValues.push_back(0.0f);

        const uint32_t numSteps = 10000;
        // Actually cache the values for the tvalue lookup in the next steps
        bezierCurve.CacheArclength(numSteps);

        const float halfDS = ds * 1.0f;

        static int bezierSegIdx = 0;
        m_upCachedCurve->m_segmentLength = ds;
        // First Position. Seeded by problem
        {
            m_upCachedCurve->m_positions[0] = m_startPos;
            m_upCachedCurve->m_tangents[0] = m_startDir;
            bezierSegIdx++;
        }
        // Second position, fixed by construction
        {
            m_upCachedCurve->m_positions[1] = m_startPos + m_startDir * m_upCachedCurve->m_segmentLength;
        }
        // Last position, defined by problem
        {
            m_upCachedCurve->m_positions[numSegments] = m_endPos;
        }

        // All other segments than first one
        for (uint32_t i = 2; i < numSegments; ++i)
        {
            // Target Arclength
            float targetArclength = (i) * ds * 0.9999f;

            const uint32_t maxNumberOfIterations = 1000;

            float a = minT;
            float b = maxT;
            uint32_t currentIterationCount = 0;

            const float errorThresh = 1e-4f;
            const float distThresh = 1e-4f;

            float minF = bezierCurve.CalculateArclengthAlreadyCached(minT, a) - targetArclength;
            float maxF = bezierCurve.CalculateArclengthAlreadyCached(minT, b) - targetArclength;

            if ((minF * maxF) > 0.0f)
            {
                // Midpoint method will fail
                std::cout << "Error: Cannot perform midpoint method" << std::endl;
                return nullptr;
            }

            float guessVal = (a + b) / 2.0f;
            while (currentIterationCount < maxNumberOfIterations)
            {
                float aVal = bezierCurve.CalculateArclengthAlreadyCached(minT, a) - targetArclength;
                float bVal = bezierCurve.CalculateArclengthAlreadyCached(minT, b) - targetArclength;
                float guessF = bezierCurve.CalculateArclengthAlreadyCached(minT, guessVal) - targetArclength;
                if (std::abs(guessF) < errorThresh)
                {
                    // Guess works
                    break;
                }

                if (std::abs(a - b) < distThresh)
                {
                    // a and b close enough, we terminate
                    break;
                }

                if ((aVal * guessF) < 0.0f)
                {
                    b = guessVal;
                }
                else
                {
                    a = guessVal;
                }
                guessVal = (a + b) / 2.0f;
                currentIterationCount++;
            }

            m_cachedTValues.push_back(guessVal);

            // Lets get all segment information
            // Sample this from the curve

            Farlor::Vector3 segmentPosition = bezierCurve.GetPosition(guessVal);
            m_upCachedCurve->m_positions[i] = segmentPosition;
            bezierSegIdx++;
        }

        // Caclulate the cached tangents here
        for (uint32_t i = 0; i < m_upCachedCurve->m_numSegments; ++i)
        {
            Farlor::Vector3 diff = (m_upCachedCurve->m_positions[i + 1] - m_upCachedCurve->m_positions[i]);
            m_upCachedCurve->m_tangents[i] = diff.Normalized();
        }

        // Write final tangent here
        {
            m_upCachedCurve->m_tangents[numSegments] = m_endDir.Normalized();
        }

        // All but the last segment. We do that one manually
        for (uint32_t i = 0; i < numSegments; ++i)
        {
            auto& tanLeft = m_upCachedCurve->m_tangents[i];
            auto& tanRight = m_upCachedCurve->m_tangents[i + 1];

            {
                float curvature = ((tanRight - tanLeft) * (1.0f / ds)).Magnitude();
                m_upCachedCurve->m_curvatures[i] = curvature;
            }
        }

        float totalDistance = 0.0f;
        // Calculate the actual length of the segment based curve
        for (uint32_t i = 0; i < m_upCachedCurve->m_numSegments - 1; ++i)
        {
            totalDistance += (m_upCachedCurve->m_positions[i + 1] - m_upCachedCurve->m_positions[i]).Magnitude();
        }
        // And handle last segment
        {
            totalDistance += (m_upCachedCurve->m_targetPos - m_upCachedCurve->m_positions[m_upCachedCurve->m_numSegments - 1]).Magnitude();
        }
        std::cout << "\tSegment Curve Distance: " << totalDistance << std::endl;
        std::cout << "\tTarget Arclength: " << m_upCachedCurve->m_arclength << std::endl;

        // We need to return a copy
        m_isCached = true;

        auto upReturnCurve = std::make_unique<Curve>(m_upCachedCurve->m_numSegments);
        *upReturnCurve = *m_upCachedCurve;
        return upReturnCurve;
    }

    std::unique_ptr<Bootstrapper::BezierInfo> Bootstrapper::GetBezierInfo() const
    {
        if (m_isCached)
        {
            auto upBezierInfo = std::make_unique<Bootstrapper::BezierInfo>();
            *upBezierInfo = *m_upCachedBezierInfo;
            return upBezierInfo;
        }
        return nullptr;
    }

    std::vector<Farlor::Vector3> Bootstrapper::GetTValuePositions() const
    {
        std::vector<Farlor::Vector3> positions;
        if (m_isCached)
        {
            for (auto value : m_cachedTValues)
            {
                positions.push_back(m_upCachedBezier->GetPosition(value));
            }
        }
        return positions;
    }

    std::vector<Farlor::Vector3> Bootstrapper::GetTValueFrames() const
    {
        std::vector<Farlor::Vector3> tangents;
        if (m_isCached)
        {
            for (auto value : m_cachedTValues)
            {
                Farlor::Vector3 tangent;
                tangent = m_upCachedBezier->Tangent(value).Normalized();
                tangents.push_back(tangent);
            }
        }
        return tangents;
    }
}