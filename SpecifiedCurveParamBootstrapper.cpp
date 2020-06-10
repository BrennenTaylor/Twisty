#include "SpecifiedCurveParamBootstrapper.h"

#include <assert.h>

namespace twisty
{
    SpecifiedCurveParamBootstrapper::SpecifiedCurveParamBootstrapper(float initialCurvature, float initialTorsion, const Range& arclengthRange, uint32_t randomSeed)
        : Bootstrapper(arclengthRange, randomSeed)
        , m_initialCurvature(initialCurvature)
        , m_initialTorsion(initialTorsion)
    {
        m_startPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        m_startDir = Farlor::Vector3(1.0f, 0.0f, 0.0f);
    }

    std::unique_ptr<Curve> SpecifiedCurveParamBootstrapper::CreateCurve(uint32_t numSegments)
    {
        assert(false);
        return nullptr;
        //m_upCachedCurve = std::make_unique<Curve>(numSegments);
        //// Min/max should be the same values, just pick one.
        //m_upCachedCurve->m_arclength = m_arclengthRange.m_min;
        //m_upCachedCurve->m_numSegments = numSegments;
        //m_upCachedCurve->m_basePos = m_startPos;
        //m_upCachedCurve->m_baseTangent = m_startDir;
        //m_upCachedCurve->m_baseNormal = Farlor::Vector3(0.0f, 1.0f, 0.0f);
        //m_upCachedCurve->m_baseBinormal = m_upCachedCurve->m_baseTangent.Cross(m_upCachedCurve->m_baseNormal).Normalized();
        //m_upCachedCurve->m_baseNormal = m_upCachedCurve->m_baseBinormal.Cross(m_upCachedCurve->m_baseTangent).Normalized();
        //// These are set later
        //// Default to invalid values
        //m_upCachedCurve->m_targetPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        //m_upCachedCurve->m_targetTangent = Farlor::Vector3(0.0f, 0.0f, 0.0f);

        //// Initialize base position and frame
        //Farlor::Vector3 x_0 = m_upCachedCurve->m_basePos;
        //Farlor::Matrix3x3 f_0 = Farlor::Matrix3x3(m_upCachedCurve->m_baseTangent, m_upCachedCurve->m_baseNormal, m_upCachedCurve->m_baseBinormal);

        //Farlor::Vector3 x_j = x_0;
        //Farlor::Matrix3x3 f_j = f_0;

        //m_cachedTValues.clear();
        //// We always wand the first one
        //m_cachedTValues.push_back(0.0f);

        //const float ds = m_upCachedCurve->m_arclength / m_upCachedCurve->m_numSegments;

        //std::vector<Farlor::Matrix3x3> frameList;
        //frameList.push_back(f_0);

        //// Update each segment
        //for (uint32_t i = 0; i < numSegments; ++i)
        //{
        //    Segment segment;
        //    segment.m_length = ds;
        //    segment.m_curvature = m_initialCurvature;

        //    m_upCachedCurve->m_segments[i] = segment;
        //}

        //// Finally, lets build up the calculated end values
        //// At this point, we should have a valid position and tangent in the end
        //x_j = x_0;
        //f_j = f_0;

        //for (auto &segment : m_upCachedCurve->m_segments)
        //{
        //    x_j = x_j + f_j.m_rows[0] * segment.m_length;
        //    f_j = segment.m_rotationMatrix * f_j;
        //}

        //m_endPos = x_j;
        //m_endDir = f_j.m_rows[0];

        //m_upCachedCurve->m_targetPos = m_endPos;
        //m_upCachedCurve->m_targetTangent = m_endDir;

        //// We need to return a copy
        //m_isCached = true;

        //auto upReturnCurve = std::make_unique<Curve>(m_upCachedCurve->m_numSegments);
        //*upReturnCurve = *m_upCachedCurve;
        //return upReturnCurve;
    }

    void SpecifiedCurveParamBootstrapper::BeginReset()
    {
    }

    void SpecifiedCurveParamBootstrapper::EndReset()
    {
    }
}