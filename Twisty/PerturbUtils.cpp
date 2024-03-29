// #include "PerturbUtils.h"

// #include "PathWeightUtils.h"

// namespace twisty
// {
//     namespace PerturbUtils
//     {
//         // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
//         void UpdateTangentsFromPos(Farlor::Vector3* pPositions, Farlor::Vector3* pTangents,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions)
//         {
//             UpdateTangentsFromPos((float*)pPositions->m_data(), (float*)pTangents->m_data(), numSegments, boundaryConditions);
//         }

//         // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
//         void UpdateCurvaturesFromTangents(Farlor::Vector3* pTangents, float* pCurvatures,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions, const twisty::WeightingParameters& wp)
//         {
//             UpdateCurvaturesFromTangents((float*)pTangents->m_data(), pCurvatures, numSegments, boundaryConditions, wp);
//         }

//         // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
//         void UpdateTangentsFromPos(float* pPositions, float* pTangents,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions)
//         {
//             const float ds = boundaryConditions.arclength / numSegments;

//             // Set initial and final positions
//             pPositions[0] = boundaryConditions.m_startPos;
//             pPositions[1] = pPositions[0] + ds * boundaryConditions.m_startDir.Normalized();
//             pPositions[numSegments] = boundaryConditions.m_endPos;

//             // Update tangents
//             for (uint32_t i = 0; i < numSegments; ++i)
//             {
//                 Farlor::Vector3 diff = (pPositions[i + 1] - pPositions[i]);
//                 pTangents[i] = diff.Normalized();
//             }
//             pTangents[numSegments] = boundaryConditions.m_endDir;
//         }

//         // This function assumes that the initial and end positions and tangents are set already to the constraints defined by the problem
//         void UpdateCurvaturesFromTangents(float* pTangents, float* pCurvatures,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions, const twisty::WeightingParameters& wp)
//         {
//             const float ds = boundaryConditions.arclength / numSegments;

//             switch (wp.weightingMethod)
//             {
//                 case twisty::WeightingMethod::RadiativeTransfer:
//                 {
//                     // Update segments
//                     for (uint32_t i = 0; i < numSegments; ++i)
//                     {
//                         auto& tanLeft = pTangents[i];
//                         auto& tanRight = pTangents[i + 1];
//                         {   
//                             const float curvature = ((tanRight - tanLeft) * (1.0f / ds)).Magnitude();
//                             pCurvatures[i] = curvature;
//                         }
//                     }
//                 } break;
//                 case twisty::WeightingMethod::SimplifiedModel:
//                 {
//                     // Update segments
//                     for (uint32_t i = 0; i < numSegments; ++i)
//                     {
//                         auto& tanLeft = pTangents[i];
//                         auto& tanRight = pTangents[i + 1];
//                         {   
//                             const float curvature = tanRight.Normalized().Dot(tanLeft.Normalized());
//                             pCurvatures[i] = -curvature; // Negate curvature so that an increase in curvature leads to higher weight values
//                         }
//                     }
//                 } break;
//                 default:
//                 {
//                     std::cout << "Unsupported weighting function, unknown curvature definition" << std::endl;
//                 } break;
//             };
//         }
//     }
// }