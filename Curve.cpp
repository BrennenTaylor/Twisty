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

    //void Curve::ExportCurve(std::string name)
    //{
    //    // For now, lets output the path batches
    //    std::ofstream outfile;
    //    std::string filename;
    //    filename = name;
    //    filename += std::string(".csv");

    //    std::string totalPath = std::experimental::filesystem::current_path().string();
    //    totalPath += std::string("/") + filename;
    //    outfile.open(totalPath.c_str());

    //    // Do the curve head information
    //    outfile << "Curve Head\n";
    //    outfile << "Num Segments," << m_numSegments << "\n";
    //    outfile << "Arclength," << m_arclength << "\n";

    //    outfile << "Base Position and Frame" << std::endl;
    //    outfile << "Base Position," << m_basePos.x << ", " << m_basePos.y << ", " << m_basePos.z << "\n";
    //    outfile << "Base Tangent," << m_baseTangent.x << ", " << m_baseTangent.y << ", " << m_baseTangent.z << "\n";

    //    outfile << "Segments\n";
    //    for (uint32_t i = 0; i < m_numSegments; ++i)
    //    {
    //        outfile << "\nSegment " << i << "\n";
    //        outfile << "Length," << m_segments[i].m_length << "\n";
    //        outfile << "Position," << m_segments[i].m_position.x << ", " << m_segments[i].m_position.y << ", " << m_segments[i].m_position.z << "\n";
    //        outfile << "Tangent," << m_segments[i].m_tangent.x << ", " << m_segments[i].m_tangent.y << ", " << m_segments[i].m_tangent.z << "\n";
    //    }

    //    outfile << std::endl;
    //    outfile.close();
    //}
}