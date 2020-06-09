#include "CurveUtils.h"

#include <algorithm>

namespace twisty
{
    float CurveUtils::CalculateCurveError(const Curve& curve)
    {
        Farlor::Vector3 finalPos(0.0f, 0.0f, 0.0f);
        Farlor::Vector3 finalDir(0.0f, 0.0f, 0.0f);
        curve.CalculateFinalPosAndTangent(finalPos, finalDir);

        // Weights to tweak
        const float posWeight = 1.0f;
        const float tanWeight = 1.0f;

        float dx = abs(finalPos.x - curve.m_targetPos.x);
        float dy = abs(finalPos.y - curve.m_targetPos.y);
        float dz = abs(finalPos.z - curve.m_targetPos.z);

        // L1 norm of tangents
        float l1 = abs(finalDir.x - curve.m_targetTangent.x) + abs(finalDir.y - curve.m_targetTangent.y) + abs(finalDir.z - curve.m_targetTangent.z);
        // L2 norm of tangents
        float l2 = (finalDir - curve.m_targetTangent).Magnitude();

        float maxError = dx;
        maxError = std::max(maxError, dy);
        maxError = std::max(maxError, dz);
        maxError = std::max(maxError, l1);
        maxError = std::max(maxError, l2);

        return maxError;
    }

    float CurveUtils::CalculateCurveMeasure(const Curve& curve)
    {
        float curveError = CalculateCurveError(curve);
        //std::cout << "Curve Error: " << curveError << std::endl;
        return exp(-1.0f * curveError);
    }
}