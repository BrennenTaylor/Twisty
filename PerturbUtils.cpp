#include "PerturbUtils.h"

//#define USE_BETTER_CURVATURE

namespace twisty
{
    namespace PerturbUtils
    {
#ifndef USE_BETTER_CURVATURE
        // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
        void RecalculateTangentsCurvaturesFromPos(Farlor::Vector3* pPositions, Farlor::Vector3* pTangents,
            float* pCurvatures, const uint32_t numSegments, const BoundaryConditions& boundaryConditions)
        {
            const float ds = boundaryConditions.arclength / numSegments;

            // Set initial and final positions
            pPositions[0] = boundaryConditions.m_startPos;
            pPositions[1] = pPositions[0] + ds * boundaryConditions.m_startDir.Normalized();
            pPositions[numSegments] = boundaryConditions.m_endPos;

            // Update tangents
            for (uint32_t i = 0; i < numSegments; ++i)
            {
                Farlor::Vector3 diff = (pPositions[i + 1] - pPositions[i]);
                pTangents[i] = diff.Normalized();
            }
            pTangents[numSegments] = boundaryConditions.m_endDir;

            // Update segments
            for (uint32_t i = 0; i < numSegments; ++i)
            {
                auto& tanLeft = pTangents[i];
                auto& tanRight = pTangents[i + 1];
                {
                    const float curvature = tanRight.Normalized().Dot(tanLeft.Normalized());
                    //const float curvature = ((tanRight - tanLeft) * (1.0f / ds)).Magnitude();
                    pCurvatures[i] = curvature;
                }
            }
        }
#else
        // Note: This version uses the symetric difference of tangents for curvature
        // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
        void RecalculateTangentsCurvaturesFromPos(Farlor::Vector3* pPositions, Farlor::Vector3* pTangents,
            float* pCurvatures, const uint32_t numSegments, const BoundaryConditions& boundaryConditions)
        {
            const float ds = boundaryConditions.arclength / numSegments;

            // Set initial and final positions
            pPositions[0] = boundaryConditions.m_startPos;
            pPositions[1] = pPositions[0] + ds * boundaryConditions.m_startDir.Normalized();
            pPositions[numSegments] = boundaryConditions.m_endPos;

            // Update tangents
            // Set first tangent directly, defined by boundary conditions
            pTangents[0] = boundaryConditions.m_startDir;

            // Set others via finite difference
            for (uint32_t i = 1; i < numSegments; ++i)
            {
                Farlor::Vector3 diff = (pPositions[i + 1] - pPositions[i - 1]);
                pTangents[i] = diff.Normalized();
            }
            pTangents[numSegments] = boundaryConditions.m_endDir;

            // Calculate curvature
            // First, we calcualte only the first using the old method
            {
                auto& tanLeft = pTangents[0];
                auto& tanRight = pTangents[1];
                {
                    const float curvature = ((tanRight - tanLeft) * (1.0f / ds)).Magnitude();
                    pCurvatures[0] = curvature;
                }
            }

            // All the rest we calculate using the new method
            for (uint32_t i = 1; i < numSegments; ++i)
            {
                // First, grab tangent
                auto& tan = pTangents[i];

                // Second, calculate dp2ds2
                Farlor::Vector3 dp2ds2 = pPositions[i + 1] + pPositions[i - 1] - 2.0f * pPositions[i];
                dp2ds2 = dp2ds2 * (1.0f / (ds * ds));

                // Third, calculate dTds
                Farlor::Vector3 dTds = dp2ds2 - tan * (tan.Dot(dp2ds2));
                pCurvatures[i] = dTds.Magnitude();
            }
        }
#endif
    }
}