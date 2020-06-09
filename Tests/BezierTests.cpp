#include <BezierCurve.h>

#include <assert.h>

bool ThresholdFloatEquality(float first, float second, float epsilon)
{
    return abs(first - second) <= epsilon;
}

int main()
{
    {
        twisty::BezierCurve5 b5;
        b5.m_controlPts[0] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        b5.m_controlPts[1] = Farlor::Vector3(3.0f, 4.0f, 0.0f);
        b5.m_controlPts[2] = Farlor::Vector3(5.0f, 4.0f, 5.0f);
        b5.m_controlPts[3] = Farlor::Vector3(7.0f, 3.0f, 0.0f);
        b5.m_controlPts[4] = Farlor::Vector3(10.0f, 0.0f, 0.0f);

        if (b5.m_controlPts[0] != b5.GetPosition(0.0f))
        {
            std::cout << "Failed t0" << std::endl;
        }
        if (b5.m_controlPts[4] != b5.GetPosition(1.0f))
        {
            std::cout << "Failed t1" << std::endl;
        }
    }

    // Test arclength
    {
        twisty::BezierCurve5 b5;
        b5.m_controlPts[0] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        b5.m_controlPts[1] = Farlor::Vector3(3.0f, 0.0f, 0.0f);
        b5.m_controlPts[2] = Farlor::Vector3(5.0f, 0.0f, 0.0f);
        b5.m_controlPts[3] = Farlor::Vector3(7.0f, 0.0f, 0.0f);
        b5.m_controlPts[4] = Farlor::Vector3(10.0f, 0.0f, 0.0f);

        const float targetArclength = 10.0f;
        std::cout << "Calculated arclength: " << b5.CalculateArclength(0, 1) << std::endl;
        if(!ThresholdFloatEquality(targetArclength, b5.CalculateArclength(0.0f, 1.0f), 0.1f))
        {
            std::cout << "Failed t2" << std::endl;
            std::cout << "Calculated arclength: " << b5.CalculateArclength(0, 1) << std::endl;
        }
    }

    // Test Tangent
    {
        twisty::BezierCurve5 b5;
        b5.m_controlPts[0] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        b5.m_controlPts[1] = Farlor::Vector3(3.0f, 0.0f, 0.0f);
        b5.m_controlPts[2] = Farlor::Vector3(5.0f, 0.0f, 0.0f);
        b5.m_controlPts[3] = Farlor::Vector3(7.0f, 0.0f, 0.0f);
        b5.m_controlPts[4] = Farlor::Vector3(10.0f, 0.0f, 0.0f);

        {
            const Farlor::Vector3 target(1.0f, 0.0f, 0.0f);
            if (target != b5.Tangent(0.0f))
            {
                std::cout << "Failed t3" << std::endl;
                std::cout << "Calculated tangent: " << b5.Tangent(0.0f) << std::endl;
            }
        }
        {
            const Farlor::Vector3 target(1.0f, 0.0f, 0.0f);
            if (target != b5.Tangent(1.0f))
            {
                std::cout << "Failed t4" << std::endl;
                std::cout << "Calculated tangent: " << b5.Tangent(1.0f) << std::endl;
            }
        }
    }

    return 0;
}