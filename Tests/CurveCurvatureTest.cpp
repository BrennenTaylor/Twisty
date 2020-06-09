
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
    customCurve.m_baseTangent = Farlor::Vector3(1.0f, 0.0f, 0.0f);
    customCurve.m_baseNormal = Farlor::Vector3(0.0f, 1.0f, 0.0f);
    customCurve.m_baseBinormal = Farlor::Vector3(0.0f, 0.0f, 1.0f);

    customCurve.m_targetPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
    customCurve.m_targetTangent = Farlor::Vector3(1.0f, 0.0f, 0.0f);

    float segmentLength = customCurve.m_arclength / customCurve.m_numSegments;
    float segmentCurvature = ((2.0f * TwistyPi) / customCurve.m_numSegments) / segmentLength;

    for (auto& segment : customCurve.m_segments)
    {
        segment.m_curvature = segmentCurvature;
        segment.m_torsion = 0.0f;
        segment.m_length = segmentLength;
        segment.UpdateRotation();
        std::cout << "Rotation Matrix: " << segment.m_rotationMatrix << std::endl;
    }

    Farlor::Vector3 x_i(0.0f, 0.0f, 0.0f);
    Farlor::Matrix3x3 f_i(Farlor::Vector3(1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 1.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));
    // Initial condition
    Farlor::Matrix3x3 f_j = f_i;
    Farlor::Vector3 x_j = x_i;

    for (auto& segment : customCurve.m_segments)
    {
        x_j = x_j + f_j.m_rows[0] * segmentLength;
        f_j = segment.m_rotationMatrix * f_j;
    }

    std::cout << "x_i: " << x_i << std::endl;
    std::cout << "x_j: " << x_j << std::endl;

    std::cout << "f_i: " << f_i << std::endl;
    std::cout << "f_j: " << f_j << std::endl;

    // Without manipulating the state, and thus this should be a "current error", we want to report the error
    float curveMeasure = CurveUtils::CalculateCurveMeasure(customCurve);
    std::cout << "Curve Measure: " << curveMeasure << std::endl;

    /*Farlor::Matrix3x3 f_i(Farlor::Vector3(1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 1.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));
    Farlor::Matrix3x3 u_i(Farlor::Vector3(-1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, -1.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));
    Farlor::Matrix3x3 f_j = u_i * f_i;

    std::cout << "f_i: " << f_i << std::endl;
    std::cout << "u_i: " << u_i << std::endl;
    std::cout << "f_j: " << f_j << std::endl;

*/
    //{
    //    // Assume curve of length 10.0f
    //    // and m=200 segments
    //    float segmentLength = 10.0f / 200.0f;
    //    float targetSegmentAngle = (2.0f * TwistyPi) / 200.0f;

    //    Segment segment;
    //    segment.m_length = segmentLength;
    //    segment.m_curvature = targetSegmentAngle / segmentLength;
    //    segment.m_torsion = 0.0f;
    //    segment.UpdateRotation();
    //    std::cout << "Segment rotation: " << segment.m_rotationMatrix << std::endl;

    //
    //    Farlor::Vector3 x_i(0.0f, 0.0f, 0.0f);
    //    Farlor::Matrix3x3 f_i(Farlor::Vector3(1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 1.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));
    //    Farlor::Matrix3x3 u_i = segment.m_rotationMatrix;
    //    // Initial condition
    //    Farlor::Matrix3x3 f_j = f_i;
    //    Farlor::Vector3 x_j = x_i;

    //    for (uint32_t i = 0; i < 200; ++i)
    //    {
    //        x_j = x_j + f_j.m_rows[0] * segmentLength;
    //        f_j = u_i * f_j;
    //    }

    //    std::cout << "x_i: " << x_i << std::endl;
    //    std::cout << "x_j: " << x_j << std::endl;

    //    std::cout << "f_i: " << f_i << std::endl;
    //    std::cout << "u_i: " << u_i << std::endl;
    //    std::cout << "f_j: " << f_j << std::endl;
    //}
    return 0;
}