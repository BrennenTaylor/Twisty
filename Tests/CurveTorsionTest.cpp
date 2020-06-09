
#include "Curve.h"
#include "CurveUtils.h"
#include "GeometryBootstrapper.h"
#include "Geometry.h"
#include "Range.h"
#include "MathConsts.h"

using namespace twisty;

int main()
{
    const uint32_t numSegmentsPerPath = 200;

    Curve customCurve(numSegmentsPerPath);
    customCurve.m_arclength = 10.0f;
    customCurve.m_basePos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
    customCurve.m_baseTangent = Farlor::Vector3(0.0f, 1.0f, 0.0f);
    customCurve.m_baseNormal = Farlor::Vector3(1.0f, 0.0f, 0.0f);
    customCurve.m_baseBinormal = Farlor::Vector3(0.0f, 0.0f, 1.0f);

    customCurve.m_targetPos = Farlor::Vector3(0.0f, 10.0f, 0.0f);
    customCurve.m_targetTangent = Farlor::Vector3(0.0f, 1.0f, 0.0f);

    for (auto& segment : customCurve.m_segments)
    {
        segment.m_curvature = 0.0f;
        segment.m_torsion = 1.0f;
        segment.m_length = customCurve.m_arclength / customCurve.m_numSegments;
        segment.UpdateRotation();
    }

    // Without manipulating the state, and thus this should be a "current error", we want to report the error
    float curveMeasure = CurveUtils::CalculateCurveMeasure(customCurve);
    std::cout << "Curve Measure: " << curveMeasure << std::endl;
    return 0;
}