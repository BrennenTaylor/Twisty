#include "BezierCurve.h"

#include <cmath>

#include <assert.h>

// This bezier curve exists in default parameter space, t.
// Control pt 0 is at t = 0.
// Control pt N is at t = 1.
namespace twisty
{
    // -------------- Bezier Curve 5 ----------------------------
    const uint32_t BezierCurve5::s_NumControlPts = 5;

    BezierCurve5::BezierCurve5()
        : m_controlPts(s_NumControlPts)
        , m_cachedTLocations()
    {
        // Initialize control points to base
        for (uint32_t i = 0; i < s_NumControlPts; ++i)
        {
            m_controlPts[i] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        }
    }

    // Pass in t, [0.0f, 1.0f]
    // Calculate using interpolation method
    Farlor::Vector3 BezierCurve5::GetPosition(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);

        float mt = (1.0f - t);

        // TODO: We should use de casteljau's algorithm
        Farlor::Vector3 val(0.0f, 0.0f, 0.0f);
        val += 1.0f * std::pow(t, 4)              * m_controlPts[4];
        val += 4.0f * std::pow(t, 3) * std::pow(mt, 1) * m_controlPts[3];
        val += 6.0f * std::pow(t, 2) * std::pow(mt, 2) * m_controlPts[2];
        val += 4.0f * std::pow(t, 1) * std::pow(mt, 3) * m_controlPts[1];
        val += 1.0f             * std::pow(mt, 4) * m_controlPts[0];
        return val;
    }

    BezierCurve4 BezierCurve5::GetDerivativeCurve()
    {
        BezierCurve4 dCurve;
        dCurve.m_controlPts[0] = m_controlPts[1] - m_controlPts[0];
        dCurve.m_controlPts[1] = m_controlPts[2] - m_controlPts[1];
        dCurve.m_controlPts[2] = m_controlPts[3] - m_controlPts[2];
        dCurve.m_controlPts[3] = m_controlPts[4] - m_controlPts[3];
        return dCurve;
    }

    // Perform gradient decent derivative at point t
    Farlor::Vector3 BezierCurve5::FirstDerivative(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);

        BezierCurve4 dCurve = GetDerivativeCurve();
        return dCurve.GetPosition(t);
    }

    Farlor::Vector3 BezierCurve5::SecondDerivative(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);

        BezierCurve4 dCurve = GetDerivativeCurve();
        BezierCurve3 ddCurve = dCurve.GetDerivativeCurve();
        return ddCurve.GetPosition(t);
    }

    Farlor::Vector3 BezierCurve5::Tangent(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);

        return FirstDerivative(t).Normalized();
    }

    Farlor::Vector3 BezierCurve5::Normal(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);

        return SecondDerivative(t).Normalized();
    }

    Farlor::Vector3 BezierCurve5::Binormal(float t)
    {
        auto tangent = Tangent(t);
        auto normal = Normal(t);
        return tangent.Cross(normal).Normalized();
    }

    // Verified with the formula at: http://tutorial.math.lamar.edu/Classes/CalcIII/VectorArcLength.aspx
    float BezierCurve5::CalculateArclength(float minVal, float maxVal)
    {
        assert(minVal >= 0.0f);
        assert(minVal <= 1.0f);
        assert(maxVal >= 0.0f);
        assert(maxVal <= 1.0f);
        assert(minVal <= maxVal);

        // This is a euler approximation to the integral
        const uint32_t numSteps = 10000;
        float stepSize = (maxVal - minVal) / numSteps;
        float arclength = 0.0f;
        Farlor::Vector3 prevPos = GetPosition(minVal);
        for (uint32_t i = 1; i <= numSteps; ++i)
        {
            Farlor::Vector3 currentPos = GetPosition(minVal + stepSize * i);
            arclength += (currentPos - prevPos).Magnitude();
            prevPos = currentPos;
        }
        return arclength;
    }

    void BezierCurve5::CacheArclength(uint32_t numCachedValues)
    {
        m_cachedTLocations.clear();
        const float minVal = 0.0f;
        const float maxVal = 1.0f;
        float stepSize = (maxVal - minVal) / numCachedValues;
        float arclength = 0.0f;
        Farlor::Vector3 prevPos = GetPosition(minVal);
        for (uint32_t i = 0; i <= numCachedValues; ++i)
        {
            Farlor::Vector3 currentPos = GetPosition(minVal + stepSize * i);
            arclength += (currentPos - prevPos).Magnitude();
            float tValue = stepSize * i;
            auto pair = std::make_pair(tValue, arclength);
            m_cachedTLocations.push_back(pair);
            prevPos = currentPos;
        }

        // for (auto& elem : m_cachedTLocations)
        // {
        //     printf ("(%f, %f)\n", elem.first, elem.second);
        // }
    }

    float BezierCurve5::CalculateArclengthAlreadyCached(float minVal, float maxVal)
    {
        // printf("Searching for arclength of %f to %f\n", minVal, maxVal);
        if (m_cachedTLocations.size() == 0)
        {
            std::cout << "Need to cache values before we can use them" << std::endl;
            assert(false);
        }
        if (minVal == maxVal)
        {
            return 0.0f;
        }
        assert(minVal < maxVal);
        if (minVal < m_cachedTLocations.front().first)
        {
            std::cout << "Bad min val, too low for cached values" << std::endl;
            assert(false);
        }
        if (minVal > m_cachedTLocations.back().first)
        {
            std::cout << "Bad min val, too high for cached values" << std::endl;
            assert(false);
        }
        if (maxVal < m_cachedTLocations.front().first)
        {
            std::cout << "Bad max val, too low for cached values" << std::endl;
            assert(false);
        }
        if (maxVal > m_cachedTLocations.back().first)
        {
            std::cout << "Bad max val, too high for cached values" << std::endl;
            assert(false);
        }

        float leftArcLength = 0.0f;
        float rightArcLength = 0.0f;

        {
            uint32_t minLeft = 0;
            uint32_t minRight = 1;
            // Find arclength of minVal
            for (uint32_t i = 1; i < m_cachedTLocations.size() - 1; ++i)
            {
                if (m_cachedTLocations[i].first <= minVal && m_cachedTLocations[i + 1].first >= minVal)
                {
                    minLeft = i;
                    minRight = i + 1;
                }
            }

            float minDistLeft = minVal - m_cachedTLocations[minLeft].first;
            float minTotalDist =  m_cachedTLocations[minRight].first - m_cachedTLocations[minLeft].first;
            float minLeftWeight = 1.0f - (minDistLeft / minTotalDist);
            float minRightWeight = 1.0f - minLeftWeight;

            // printf("Selected left and right (%f, %f), (%f, %f)\n", m_cachedTLocations[minLeft].first, m_cachedTLocations[minLeft].second,
            //     m_cachedTLocations[minRight].first, m_cachedTLocations[minRight].second);

            leftArcLength = minLeftWeight * m_cachedTLocations[minLeft].second + minRightWeight * m_cachedTLocations[minRight].second;
        }

        {
            uint32_t maxLeft = 0;
            uint32_t maxRight = 1;
            // Find arclength of minVal
            for (uint32_t i = 1; i < m_cachedTLocations.size() - 1; ++i)
            {
                if (m_cachedTLocations[i].first <= maxVal && m_cachedTLocations[i + 1].first >= maxVal)
                {
                    maxLeft = i;
                    maxRight = i + 1;
                }
            }

            float maxDistLeft = maxVal - m_cachedTLocations[maxLeft].first;
            float maxTotalDist = m_cachedTLocations[maxRight].first - m_cachedTLocations[maxLeft].first;
            float maxLeftWeight = 1.0f - (maxDistLeft / maxTotalDist);
            float maxRightWeight = 1.0f - maxLeftWeight;

            rightArcLength = maxLeftWeight * m_cachedTLocations[maxLeft].second + maxRightWeight * m_cachedTLocations[maxRight].second;
        }

        // printf("Left arc length: %f\n", leftArcLength);
        // printf("Right arc length: %f\n", rightArcLength);

        return rightArcLength - leftArcLength;
    }

    void BezierCurve5::PrintControlPts()
    {
        std::cout << "Control Points: " << std::endl;
        for (uint32_t i = 0; i < m_controlPts.size(); ++i)
        {
            std::cout << "\tCPT " << i << ": " << m_controlPts[i] << std::endl;
        }
    }

    // -------------- Bezier Curve 4 ----------------------------
    const uint32_t BezierCurve4::s_NumControlPts = 4;

    BezierCurve4::BezierCurve4()
        : m_controlPts(s_NumControlPts)
    {
        // Initialize control points to base
        for (uint32_t i = 0; i < s_NumControlPts; ++i)
        {
            m_controlPts[i] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        }
    }

    // Pass in t, [0.0f, 1.0f]
    // Calculate using interpolation method
    Farlor::Vector3 BezierCurve4::GetPosition(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);

        float mt = (1.0f - t);

        // TODO: We could use de casteljau's algorithm
        Farlor::Vector3 val(0.0f, 0.0f, 0.0f);
        val += 1.0f * pow(t, 3)              * m_controlPts[3];
        val += 3.0f * pow(t, 2) * pow(mt, 1) * m_controlPts[2];
        val += 3.0f * pow(t, 1) * pow(mt, 2) * m_controlPts[1];
        val += 1.0f             * pow(mt, 3) * m_controlPts[0];
        return val;
    }

    BezierCurve3 BezierCurve4::GetDerivativeCurve()
    {
        BezierCurve3 dCurve;
        dCurve.m_controlPts[0] = m_controlPts[1] - m_controlPts[0];
        dCurve.m_controlPts[1] = m_controlPts[2] - m_controlPts[1];
        dCurve.m_controlPts[2] = m_controlPts[3] - m_controlPts[2];
        return dCurve;
    }

    // Perform gradient decent derivative at point t
    Farlor::Vector3 BezierCurve4::FirstDerivative(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);
        BezierCurve3 dCurve;
        return dCurve.GetPosition(t);
    }

    void BezierCurve4::PrintControlPts()
    {
        std::cout << "Control Points: " << std::endl;
        for (uint32_t i = 0; i < m_controlPts.size(); ++i)
        {
            std::cout << "\tCPT " << i << ": " << m_controlPts[i] << std::endl;
        }
    }


    // -------------- Bezier Curve 3 ----------------------------
    const uint32_t BezierCurve3::s_NumControlPts = 3;

    BezierCurve3::BezierCurve3()
        : m_controlPts(s_NumControlPts)
    {
        // Initialize control points to base
        for (uint32_t i = 0; i < s_NumControlPts; ++i)
        {
            m_controlPts[i] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        }
    }

    // Pass in t, [0.0f, 1.0f]
    // Calculate using interpolation method
    Farlor::Vector3 BezierCurve3::GetPosition(float t)
    {
        assert(t >= 0.0f);
        assert(t <= 1.0f);

        float mt = (1.0f - t);

        // TODO: We could use de casteljau's algorithm
        Farlor::Vector3 val(0.0f, 0.0f, 0.0f);
        val += 1.0f * pow(t, 2)              * m_controlPts[2];
        val += 2.0f * pow(t, 1) * pow(mt, 1) * m_controlPts[1];
        val += 1.0f             * pow(mt, 2) * m_controlPts[0];
        return val;
    }

    void BezierCurve3::PrintControlPts()
    {
        std::cout << "Control Points: " << std::endl;
        for (uint32_t i = 0; i < m_controlPts.size(); ++i)
        {
            std::cout << "\tCPT " << i << ": " << m_controlPts[i] << std::endl;
        }
    }
}