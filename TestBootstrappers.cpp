#include "TestBootstrappers.h"

#include "Sample.h"
#include "MathConsts.h"

#include <algorithm>
#include <assert.h>
#include <math.h>
#include <random>
#include <fstream>
#include <filesystem>

namespace twisty
{
    namespace testing
    {
        LinearBootstrapper::LinearBootstrapper()
            : Bootstrapper(0)
        {
            m_startPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
            m_startDir = Farlor::Vector3(1.0f, 1.0f, 1.0f).Normalized();

            m_endPos = Farlor::Vector3(10.0f, 10.0f, 10.0f);
            m_endDir = Farlor::Vector3(1.0f, 1.0f, 1.0f).Normalized();
        }

        std::unique_ptr<Curve> LinearBootstrapper::CreateCurve(uint32_t numSegments)
        {
            std::cout << "Generating a bezier curve." << std::endl;
            std::cout << "Desired Constraints: " << std::endl;
            std::cout << "\tX0: " << m_startPos << std::endl;
            std::cout << "\tN0: " << m_startDir << std::endl;
            std::cout << "\tX1: " << m_endPos << std::endl;
            std::cout << "\tN1: " << m_endDir << std::endl;

            m_upCachedBezier = std::make_unique<BezierCurve5>();

            // Even
            //{
            //    const Farlor::Vector3 diff = m_endPos - m_startPos;
            //    m_upCachedBezier->m_controlPts[0] = m_startPos + diff * 0.0f;
            //    m_upCachedBezier->m_controlPts[1] = m_startPos + diff * 0.25f;
            //    m_upCachedBezier->m_controlPts[2] = m_startPos + diff * 0.5f;
            //    m_upCachedBezier->m_controlPts[3] = m_startPos + diff * 0.75f;
            //    m_upCachedBezier->m_controlPts[4] = m_startPos + diff * 1.0f;
            //    m_upCachedBezier->PrintControlPts();
            //}

            // Clumped to back
            {
                const Farlor::Vector3 diff = m_endPos - m_startPos;
                m_upCachedBezier->m_controlPts[0] = m_startPos + diff * 0.0f;
                m_upCachedBezier->m_controlPts[1] = m_startPos + diff * 0.75f;
                m_upCachedBezier->m_controlPts[2] = m_startPos + diff * 0.85f;
                m_upCachedBezier->m_controlPts[3] = m_startPos + diff * 0.95f;
                m_upCachedBezier->m_controlPts[4] = m_startPos + diff * 1.0f;
                m_upCachedBezier->PrintControlPts();
            }

            // Clumped to front
            //{
            //    const Farlor::Vector3 diff = m_endPos - m_startPos;
            //    m_upCachedBezier->m_controlPts[0] = m_startPos + diff * 0.0f;
            //    m_upCachedBezier->m_controlPts[1] = m_startPos + diff * 0.05f;
            //    m_upCachedBezier->m_controlPts[2] = m_startPos + diff * 0.15f;
            //    m_upCachedBezier->m_controlPts[3] = m_startPos + diff * 0.25f;
            //    m_upCachedBezier->m_controlPts[4] = m_startPos + diff * 1.0f;
            //    m_upCachedBezier->PrintControlPts();
            //}

            // Cached that bezier info for retrieval later
            m_upCachedBezierInfo = std::make_unique<Bootstrapper::BezierInfo>();
            m_upCachedBezierInfo->m_controlPt0 = m_upCachedBezier->m_controlPts[0];
            m_upCachedBezierInfo->m_controlPt1 = m_upCachedBezier->m_controlPts[1];
            m_upCachedBezierInfo->m_controlPt2 = m_upCachedBezier->m_controlPts[2];
            m_upCachedBezierInfo->m_controlPt3 = m_upCachedBezier->m_controlPts[3];
            m_upCachedBezierInfo->m_controlPt4 = m_upCachedBezier->m_controlPts[4];

            return ToDiscreteFSCurve(numSegments, *m_upCachedBezier);
        }

        void LinearBootstrapper::BeginReset()
        {
        }

        void LinearBootstrapper::EndReset()
        {
        }


        // Quadratic Bootstrapper ------------
        QuadraticBootstrapper::QuadraticBootstrapper()
            : Bootstrapper(0)
        {
            m_startPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
            m_startDir = Farlor::Vector3(0.0f, 1.0f, 0.0f).Normalized();

            m_endPos = Farlor::Vector3(10.0f, 10.0f, 0.0f);
            m_endDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
        }

        std::unique_ptr<Curve> QuadraticBootstrapper::CreateCurve(uint32_t numSegments)
        {
            std::cout << "Generating a bezier curve." << std::endl;
            std::cout << "Desired Constraints: " << std::endl;
            std::cout << "\tX0: " << m_startPos << std::endl;
            std::cout << "\tN0: " << m_startDir << std::endl;
            std::cout << "\tX1: " << m_endPos << std::endl;
            std::cout << "\tN1: " << m_endDir << std::endl;

            m_upCachedBezier = std::make_unique<BezierCurve5>();

            // Quadratic curve
            {
                // We want the main points to form a right angle
                // Degree 2 control pts
                const Farlor::Vector3 d2_0(0.0f, 0.0f, 0.0f);
                const Farlor::Vector3 d2_1(0.0f, 10.0f, 0.0f);
                const Farlor::Vector3 d2_2(10.0f, 10.0f, 0.0f);

                // Degree 3 control points
                const Farlor::Vector3 d3_0 = d2_0;
                const Farlor::Vector3 d3_1 = ((1.0f) / (2 + 1)) * d2_0 + (1.0f - ((1.0f) / (2 + 1))) * d2_1;
                const Farlor::Vector3 d3_2 = ((2.0f) / (2 + 1)) * d2_1 + (1.0f - ((2.0f) / (2 + 1))) * d2_2;
                const Farlor::Vector3 d3_3 = d2_2;

                const Farlor::Vector3 d4_0 = d3_0;
                const Farlor::Vector3 d4_1 = ((1.0f) / (3 + 1)) * d3_0 + (1.0f - ((1.0f) / (3 + 1))) * d3_1;
                const Farlor::Vector3 d4_2 = ((2.0f) / (3 + 1)) * d3_1 + (1.0f - ((2.0f) / (3 + 1))) * d3_2;
                const Farlor::Vector3 d4_3 = ((3.0f) / (3 + 1)) * d3_2 + (1.0f - ((3.0f) / (3 + 1))) * d3_2;
                const Farlor::Vector3 d4_4 = d3_3;

                m_upCachedBezier->m_controlPts[0] = d4_0;
                m_upCachedBezier->m_controlPts[1] = d4_1;
                m_upCachedBezier->m_controlPts[2] = d4_2;
                m_upCachedBezier->m_controlPts[3] = d4_3;
                m_upCachedBezier->m_controlPts[4] = d4_4;
                m_upCachedBezier->PrintControlPts();
            }

            // Cached that bezier info for retrieval later
            m_upCachedBezierInfo = std::make_unique<Bootstrapper::BezierInfo>();
            m_upCachedBezierInfo->m_controlPt0 = m_upCachedBezier->m_controlPts[0];
            m_upCachedBezierInfo->m_controlPt1 = m_upCachedBezier->m_controlPts[1];
            m_upCachedBezierInfo->m_controlPt2 = m_upCachedBezier->m_controlPts[2];
            m_upCachedBezierInfo->m_controlPt3 = m_upCachedBezier->m_controlPts[3];
            m_upCachedBezierInfo->m_controlPt4 = m_upCachedBezier->m_controlPts[4];

            return ToDiscreteFSCurve(numSegments, *m_upCachedBezier);
        }

        void QuadraticBootstrapper::BeginReset()
        {
        }

        void QuadraticBootstrapper::EndReset()
        {
        }


        // CircleBootstrapper------------
        CircleBootstrapper::CircleBootstrapper()
            : Bootstrapper(0)
            , m_radius{0.0f}
        {
            // We set the circle
            m_startPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
            m_startDir = Farlor::Vector3(0.0f, 1.0f, 0.0f).Normalized();

            m_endPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
            m_endDir = Farlor::Vector3(0.0f, 1.0f, 0.0f).Normalized();

            m_radius = 10.0f;
        }

        std::unique_ptr<Curve> CircleBootstrapper::CreateCurve(uint32_t numSegments)
        {
            assert(false);
            return nullptr;
            //if (m_isCached)
            //{
            //    Reset();
            //    m_isCached = false;
            //}

            //std::cout << "Generating a bezier curve." << std::endl;
            //std::cout << "Desired Constraints: " << std::endl;
            //std::cout << "\tX0: " << m_startPos << std::endl;
            //std::cout << "\tN0: " << m_startDir << std::endl;
            //std::cout << "\tX1: " << m_endPos << std::endl;
            //std::cout << "\tN1: " << m_endDir << std::endl;

            //const float fullCurveArcLength = 2.0f * m_radius * twisty::TwistyPi;

            //const Farlor::Vector3 center = m_startPos + Farlor::Vector3(0.0f, m_radius, 0.0f);

            //// ds is going to be in arclength parameterization
            //const float ds = fullCurveArcLength / numSegments;

            //printf("\tFull curve arclength: %f\n", fullCurveArcLength);
            //printf("\tds: %f\n", ds);

            //m_upCachedCurve = std::make_unique<Curve>(numSegments);
            //m_upCachedCurve->m_arclength = fullCurveArcLength;
            //m_upCachedCurve->m_numSegments = numSegments;
            //m_upCachedCurve->m_basePos = m_startPos;
            //m_upCachedCurve->m_baseTangent = m_startDir;
            //m_upCachedCurve->m_baseNormal = Farlor::Vector3(0.0f, 1.0f, 0.0f);
            //m_upCachedCurve->m_baseBinormal = m_upCachedCurve->m_baseTangent.Cross(m_upCachedCurve->m_baseNormal).Normalized();
            //m_upCachedCurve->m_baseNormal = m_upCachedCurve->m_baseBinormal.Cross(m_upCachedCurve->m_baseTangent).Normalized();
            //m_upCachedCurve->m_targetPos = GetTargetPosition();
            //m_upCachedCurve->m_targetTangent = GetTargetNormal();

            //// Base Frame
            //Farlor::Vector3 x_0 = m_startPos;
            //Farlor::Matrix3x3 f_0(m_upCachedCurve->m_baseTangent, m_upCachedCurve->m_baseNormal, m_upCachedCurve->m_baseBinormal);

            //m_cachedSegmentPositions.push_back(x_0);
            //m_cachedSegmentFrames.push_back(f_0);


            //for (uint32_t i = 0; i < numSegments; ++i)
            //{
            //    float angleOfRotation = ((twisty::TwistyPi * 2.0f) / numSegments) * (i + 1);

            //    Farlor::Matrix3x3 rotationMat(Farlor::Vector3(cos(angleOfRotation), -sin(angleOfRotation), 0.0f), Farlor::Vector3(sin(angleOfRotation), cos(angleOfRotation), 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));

            //    Farlor::Vector3 positionOffset(0.0f, m_radius, 0.0f);
            //    Farlor::Vector3 rotatedPositionOffset = rotationMat * positionOffset;

            //    Farlor::Vector3 segPos = center + rotatedPositionOffset;
            //    Farlor::Matrix3x3 segFrame = rotationMat * f_0;

            //    // Sample this from the curve
            //    Farlor::Vector3 tangent = segFrame.m_rows[0].Normalized();
            //    Farlor::Vector3 normal = segFrame.m_rows[1].Normalized();
            //    Farlor::Vector3 binormal = segFrame.m_rows[2].Normalized();
            //    normal = binormal.Cross(tangent).Normalized();

            //    m_cachedSegmentPositions.push_back(segPos);
            //    m_cachedSegmentFrames.push_back(segFrame);

            //    Segment segment;
            //    segment.m_length = ds;
            //    m_upCachedCurve->m_segments[i] = segment;
            //}

            //// Update all segment curvatures

            //for (uint32_t i = 0; i < numSegments; ++i)
            //{
            //    auto& f0 = m_cachedSegmentFrames[i];
            //    auto& f1 = m_cachedSegmentFrames[i + 1];

            //    {
            //        float curvature = ((f1.m_rows[0] - f0.m_rows[0]) * (1.0f / ds)).Magnitude();
            //        m_upCachedCurve->m_segments[i].m_curvature = curvature;
            //    }

            //    {
            //        auto torsionLeft = -1.0f * f0.m_rows[1];
            //        auto torsionRight = (f1.m_rows[2] - f0.m_rows[2]) * (1.0f / ds);
            //        float torsion = torsionLeft.Dot(torsionRight);
            //        m_upCachedCurve->m_segments[i].m_torsion = torsion;
            //    }

            //    // We can update rotation now that we've set the curvature and torsion
            //    m_upCachedCurve->m_segments[i].UpdateRotation();
            //}

            //// We need to return a copy
            //m_isCached = true;

            //auto upReturnCurve = std::make_unique<Curve>(m_upCachedCurve->m_numSegments);
            //*upReturnCurve = *m_upCachedCurve;
            //return upReturnCurve;
        }

        void CircleBootstrapper::BeginReset()
        {
        }

        void CircleBootstrapper::EndReset()
        {
        }
    }
}