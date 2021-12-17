// #pragma once

// #include <FMath/FMath.h>

// namespace twisty
// {
//     struct WeightingParameters;
// }

// namespace twisty
// {
//     namespace PerturbUtils
//     {
//         struct BoundaryConditions
//         {
//             Farlor::Vector3 m_startPos = Farlor::Vector3(0.0, 0.0, 0.0);
//             Farlor::Vector3 m_startDir = Farlor::Vector3(1.0, 0.0, 0.0);
//             Farlor::Vector3 m_endPos = Farlor::Vector3(0.0, 0.0, 0.0);
//             Farlor::Vector3 m_endDir = Farlor::Vector3(1.0, 0.0, 0.0);
//             float arclength = 0.0f;
//         };

//         void UpdateTangentsFromPos(Farlor::Vector3* pPositions, Farlor::Vector3* pTangents,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions);

//         void UpdateCurvaturesFromTangents(Farlor::Vector3* pTangents, float* pCurvatures,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions, const twisty::WeightingParameters& wp);
//     }
// }