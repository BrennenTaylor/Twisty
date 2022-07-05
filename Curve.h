/**
 * @file Curve.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-03-13
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include <FMath/Matrix3x3.h>
#include <FMath/Vector3.h>

#include "CurvePerturbUtils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief Contains all functionality of the twisty project
 *
 */
namespace twisty {
/**
 * @brief Object which represents a discrete fs curve.
 *
 */
class Curve {
   public:
    Curve(uint32_t numSegments = 0);
    static std::unique_ptr<Curve> LoadCurveFromStream(std::ifstream &ifstream);
    static void WriteCurveToStream(std::ofstream &outputStream, const twisty::Curve &seedCurve);

    twisty::PerturbUtils::BoundaryConditions GetBoundaryConditions() const;
    twisty::PerturbUtils::BoundaryConditions_CudaSafe GetBoundaryConditionsCudaSafe() const;

   public:
    uint32_t m_numSegments = 0;
    float m_ds = 0.0f;
    PerturbUtils::BoundaryConditions m_boundaryConditions;

    std::vector<float> m_curvatures;
    std::vector<Farlor::Vector3> m_positions;
    std::vector<Farlor::Vector3> m_tangents;
};
}  // namespace twisty