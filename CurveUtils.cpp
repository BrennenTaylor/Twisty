#include "CurveUtils.h"

#include <algorithm>
#include <cmath>

namespace twisty {
// float CurveUtils::CalculateCurveError(const Curve& curve)
// {
//     Farlor::Vector3 finalPos(0.0f, 0.0f, 0.0f);
//     Farlor::Vector3 finalDir(0.0f, 0.0f, 0.0f);
//     curve.CalculateFinalPosAndTangent(finalPos, finalDir);

//     // Weights to tweak
//     const float posWeight = 1.0f;
//     const float tanWeight = 1.0f;

//     float dx = std::abs(finalPos.x - curve.m_targetPos.x);
//     float dy = std::abs(finalPos.y - curve.m_targetPos.y);
//     float dz = std::abs(finalPos.z - curve.m_targetPos.z);

//     // L1 norm of tangents
//     float l1 = std::abs(finalDir.x - curve.m_targetTangent.x) + std::abs(finalDir.y - curve.m_targetTangent.y)
//         + std::abs(finalDir.z - curve.m_targetTangent.z);
//     // L2 norm of tangents
//     float l2 = (finalDir - curve.m_targetTangent).Magnitude();

//     float maxError = dx;
//     maxError = std::max(maxError, dy);
//     maxError = std::max(maxError, dz);
//     maxError = std::max(maxError, l1);
//     maxError = std::max(maxError, l2);

//     return maxError;
// }

// float CurveUtils::CalculateCurveMeasure(const Curve& curve)
// {
//     float curveError = CalculateCurveError(curve);
//     //std::cout << "Curve Error: " << curveError << std::endl;
//     return std::exp(-1.0f * curveError);
// }
}