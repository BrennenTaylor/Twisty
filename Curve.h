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

#include <FMath/Vector3.h>
#include <FMath/Matrix3x3.h>

#include "CurvePerturbUtils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// namespace twisty
// {
//     namespace PerturbUtils
//     {
//         struct BoundaryConditions;
//     }
// }

/**
 * @brief Contains all functionality of the twisty project
 *
 */
namespace twisty
{
    /**
     * @brief Object which represents a discrete fs curve.
     *
     */
    class Curve
    {
    public:
        /**
         * @brief Construct a new Curve object
         *
         * @param numSegments Number of discrete segments in the curve
         */
        explicit Curve(uint32_t numSegments = 0);

        /**
         * @brief Write the curve data out to a file specified by the filename
         *
         * @param filename NAme of file to write out to without extension
         */
        //void ExportCurve(std::string filename);

        /**
         * @brief Itegrates over the curve using a first order method, obtaining position and frame values
         * for each segment.
         *
         * @param positions Out-parameter of positions of each integrated segment
         * @param frames Out-parameter of frames of each integrated segment
         */
        void ReconstructCurvePositionsAndFramesFirstOrder(std::vector<Farlor::Vector3>& positions, std::vector<Farlor::Vector3>& tangents) const;

        /**
         * @brief Itegrates over the curve using a first order method, obtaining position and frame values
         * for each segment.
         *
         * @param positions Out-parameter of positions of each integrated segment
         * @param frames Out-parameter of frames of each integrated segment
         */
        void ReconstructCurvePositionsFirstOrder(std::vector<Farlor::Vector3>& positions) const;

        void CalculateFinalPosAndTangent(Farlor::Vector3& finalPos, Farlor::Vector3& finalDir) const;

        // Should be inverse functions
        static std::unique_ptr<Curve> LoadCurveFromStream(std::ifstream& ifstream);
        static void WriteCurveToStream(std::ofstream& outputStream, const twisty::Curve& seedCurve);

        // enum class CurveMode
        // {
        //     ReconstructFrames = 0,
        //     StoredTangents = 1
        // };

        twisty::PerturbUtils::BoundaryConditions GetBoundaryConditions() const;
        twisty::PerturbUtils::BoundaryConditions_CudaSafe GetBoundaryConditionsCudaSafe() const;

    public:
        uint32_t m_numSegments = 0;
        float m_arclength = 0.0f;
        float m_segmentLength = 0.0f;

        Farlor::Vector3 m_basePos;
        Farlor::Vector3 m_baseTangent;
        // Farlor::Vector3 m_baseNormal;
        // Farlor::Vector3 m_baseBinormal;
        Farlor::Vector3 m_targetPos;
        Farlor::Vector3 m_targetTangent;

        /*
            With M segments, we store:
            M curvature values
            M + 1 position values
                The first position is always the start position
                The second position is always the start position + start direction * ds
                The final position is always the target position
            M + 1 direction values
                The first direction is always the start direction
                The final direction is always the target direction
           NOTE: All valid curves must have these properties hold
        */
        std::vector<float> m_curvatures;
        std::vector<Farlor::Vector3> m_positions;
        std::vector<Farlor::Vector3> m_tangents;

        // CurveMode m_mode = CurveMode::StoredTangents;
    };
}