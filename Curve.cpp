#include "Curve.h"

#include <assert.h>
#include <filesystem>
#include <fstream>

namespace twisty
{
    Curve::Curve(uint32_t numSegments)
        : m_numSegments{ numSegments }
        , m_arclength{ 0.0f }
        , m_segmentLength(0.0f)
        , m_basePos{ 0.0f, 0.0f, 0.0f }
        , m_baseTangent{ 0.0f, 0.0f, 0.0f }
        , m_targetPos{0.0f, 0.0f, 0.0f}
        , m_targetTangent{0.0f, 0.0f, 0.0f}
        , m_curvatures(m_numSegments)
        , m_positions(m_numSegments + 1)
        , m_tangents(m_numSegments + 1)
    {
    }

    void Curve::CalculateFinalPosAndTangent(Farlor::Vector3& finalPos, Farlor::Vector3& finalDir) const
    {
        finalPos = m_targetPos;
        finalDir = m_targetTangent;
    }

    // Returns the polyline, as a vector of points, from the starting to end point with every segment point along the way
    // The starting point and S_0's point are the same.
    // Assumes that all the segments have up to date curvature, torsion and U matricies
    void Curve::ReconstructCurvePositionsFirstOrder(std::vector<Farlor::Vector3>& positions) const
    {
        positions.clear();

        for (uint32_t segIdx = 0; segIdx < m_numSegments; ++segIdx)
        {
            // Grab the segment and cache for easy access
            positions.push_back(m_positions[segIdx]);
        }
    }

    void Curve::ReconstructCurvePositionsAndFramesFirstOrder(std::vector<Farlor::Vector3>& positions, std::vector<Farlor::Vector3>& tangents) const
    {
        // Ensure the vectors we want to fill are empty.
        positions.clear();
        tangents.clear();

        for (uint32_t segIdx = 0; segIdx < m_numSegments; ++segIdx)
        {
            // Grab the segment and cache for easy access
            positions.push_back(m_positions[segIdx]);
            tangents.push_back(m_tangents[segIdx]);
        }
    }

    // Must be an open, binary ofstream
    void Curve::WriteCurveToStream(std::ofstream& outputStream, const twisty::Curve& seedCurve)
    {
        outputStream.write((char*)&seedCurve.m_numSegments, sizeof(uint32_t));
        outputStream.write((char*)&seedCurve.m_arclength, sizeof(float));

        outputStream.write((char*)&seedCurve.m_basePos, sizeof(float) * 3);
        outputStream.write((char*)&seedCurve.m_baseTangent, sizeof(float) * 3);
        outputStream.write((char*)&seedCurve.m_targetPos, sizeof(float) * 3);
        outputStream.write((char*)&seedCurve.m_targetTangent, sizeof(float) * 3);

        outputStream.write((char*)&seedCurve.m_curvatures[0], sizeof(float) * seedCurve.m_numSegments);
        outputStream.write((char*)&seedCurve.m_positions[0], sizeof(Farlor::Vector3) * (seedCurve.m_numSegments + 1));
        outputStream.write((char*)&seedCurve.m_tangents[0], sizeof(Farlor::Vector3) * (seedCurve.m_numSegments + 1));

        // TODO: Depricate this: Expects number of paths to be written but not useful
        uint32_t tempVal = 0;
        outputStream.write((char*)&tempVal, sizeof(uint32_t));
    }

    std::unique_ptr<Curve> Curve::LoadCurveFromStream(std::ifstream& inputStream)
    {
        uint32_t numSegments = 0;
        inputStream.read((char*)&numSegments, sizeof(uint32_t));

        std::unique_ptr<Curve> upInitialCurve = std::make_unique<Curve>(numSegments);

        inputStream.read((char*)&upInitialCurve->m_arclength, sizeof(float));
        inputStream.read((char*)&upInitialCurve->m_basePos, sizeof(Farlor::Vector3));
        inputStream.read((char*)&upInitialCurve->m_baseTangent, sizeof(Farlor::Vector3));
        inputStream.read((char*)&upInitialCurve->m_targetPos, sizeof(Farlor::Vector3));
        inputStream.read((char*)&upInitialCurve->m_targetTangent, sizeof(Farlor::Vector3));

        inputStream.read((char*)&upInitialCurve->m_curvatures[0], sizeof(float) * upInitialCurve->m_numSegments);
        inputStream.read((char*)&upInitialCurve->m_positions[0], sizeof(Farlor::Vector3) * (upInitialCurve->m_numSegments + 1));
        inputStream.read((char*)&upInitialCurve->m_tangents[0], sizeof(Farlor::Vector3) * (upInitialCurve->m_numSegments + 1));

        return upInitialCurve;
    }

    twisty::PerturbUtils::BoundaryConditions Curve::GetBoundaryConditions() const
    {
        twisty::PerturbUtils::BoundaryConditions bc;
        bc.arclength = m_arclength;
        bc.m_startPos = m_basePos;
        bc.m_startDir = m_baseTangent;
        bc.m_endPos = m_targetPos;
        bc.m_endDir = m_targetTangent;
        return bc;
    }
}