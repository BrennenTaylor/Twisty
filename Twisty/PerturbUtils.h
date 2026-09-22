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
//             glm::vec3 m_startPos = glm::vec3(0.0, 0.0, 0.0);
//             glm::vec3 m_startDir = glm::vec3(1.0, 0.0, 0.0);
//             glm::vec3 m_endPos = glm::vec3(0.0, 0.0, 0.0);
//             glm::vec3 m_endDir = glm::vec3(1.0, 0.0, 0.0);
//             float arclength = 0.0f;
//         };

//         void UpdateTangentsFromPos(glm::vec3* pPositions, glm::vec3* pTangents,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions);

//         void UpdateCurvaturesFromTangents(glm::vec3* pTangents, float* pCurvatures,
//             const uint32_t numSegments, const BoundaryConditions& boundaryConditions, const twisty::WeightingParameters& wp);
//     }
// }