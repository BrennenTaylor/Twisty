#include "Curve.h"

#include "CurvePerturbUtils.h"

#include <assert.h>
#include <filesystem>
#include <fstream>

namespace twisty {
Curve::Curve(uint32_t numSegments)
    : m_numSegments { numSegments }
    , m_ds(0.0f)
    , m_boundaryConditions()
    , m_curvatures(m_numSegments - 1)
    , m_positions(m_numSegments + 1)
    , m_tangents(m_numSegments)
{
}

void Curve::SetBoundaryConditions(
      const twisty::PerturbUtils::BoundaryConditions &boundaryConditions)
{
    m_boundaryConditions = boundaryConditions;
    m_ds = m_boundaryConditions.arclength / m_numSegments;
    m_positions[0] = boundaryConditions.m_startPos;
    m_positions[1] = boundaryConditions.m_startPos + boundaryConditions.m_startDir * m_ds;
    m_positions[m_numSegments - 1]
          = boundaryConditions.m_endPos - boundaryConditions.m_endDir * m_ds;
    m_positions[m_numSegments] = boundaryConditions.m_endPos;
}

// Must be an open, binary ofstream
void Curve::WriteCurveToStream(std::ofstream &outputStream, const twisty::Curve &seedCurve)
{
    outputStream.write((char *)&seedCurve.m_numSegments, sizeof(uint32_t));

    outputStream.write((char *)&seedCurve.m_boundaryConditions.m_startPos, sizeof(float) * 3);
    outputStream.write((char *)&seedCurve.m_boundaryConditions.m_startDir, sizeof(float) * 3);
    outputStream.write((char *)&seedCurve.m_boundaryConditions.m_endPos, sizeof(float) * 3);
    outputStream.write((char *)&seedCurve.m_boundaryConditions.m_endDir, sizeof(float) * 3);
    outputStream.write((char *)&seedCurve.m_boundaryConditions.arclength, sizeof(float));

    outputStream.write((char *)&seedCurve.m_positions[0],
          sizeof(Farlor::Vector3) * (seedCurve.m_numSegments + 1));
}

std::unique_ptr<Curve> Curve::LoadCurveFromStream(std::ifstream &inputStream)
{
    uint32_t numSegments = 0;
    inputStream.read((char *)&numSegments, sizeof(uint32_t));

    std::unique_ptr<Curve> upInitialCurve = std::make_unique<Curve>(numSegments);

    inputStream.read(
          (char *)&upInitialCurve->m_boundaryConditions.m_startPos, sizeof(Farlor::Vector3));
    inputStream.read(
          (char *)&upInitialCurve->m_boundaryConditions.m_startDir, sizeof(Farlor::Vector3));
    inputStream.read(
          (char *)&upInitialCurve->m_boundaryConditions.m_endPos, sizeof(Farlor::Vector3));
    inputStream.read(
          (char *)&upInitialCurve->m_boundaryConditions.m_endDir, sizeof(Farlor::Vector3));
    inputStream.read((char *)&upInitialCurve->m_boundaryConditions.arclength, sizeof(float));

    upInitialCurve->m_ds = upInitialCurve->m_boundaryConditions.arclength / numSegments;

    inputStream.read((char *)&upInitialCurve->m_positions[0],
          sizeof(Farlor::Vector3) * (upInitialCurve->m_numSegments + 1));

    PerturbUtils::UpdateTangentsFromPos(upInitialCurve->m_positions.data(),
          upInitialCurve->m_tangents.data(),
          numSegments,
          upInitialCurve->m_boundaryConditions);
    PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(upInitialCurve->m_tangents.data(),
          upInitialCurve->m_curvatures.data(),
          numSegments,
          upInitialCurve->m_boundaryConditions);

    return upInitialCurve;
}

twisty::PerturbUtils::BoundaryConditions Curve::GetBoundaryConditions() const
{
    return m_boundaryConditions;
}

twisty::PerturbUtils::BoundaryConditions_CudaSafe Curve::GetBoundaryConditionsCudaSafe() const
{
    twisty::PerturbUtils::BoundaryConditions_CudaSafe bc;
    bc.arclength = m_boundaryConditions.arclength;
    bc.m_startPos[0] = m_boundaryConditions.m_startPos.m_data[0];
    bc.m_startPos[1] = m_boundaryConditions.m_startPos.m_data[1];
    bc.m_startPos[2] = m_boundaryConditions.m_startPos.m_data[2];

    bc.m_startDir[0] = m_boundaryConditions.m_startDir.m_data[0];
    bc.m_startDir[1] = m_boundaryConditions.m_startDir.m_data[1];
    bc.m_startDir[2] = m_boundaryConditions.m_startDir.m_data[2];

    bc.m_endPos[0] = m_boundaryConditions.m_endPos.m_data[0];
    bc.m_endPos[1] = m_boundaryConditions.m_endPos.m_data[1];
    bc.m_endPos[2] = m_boundaryConditions.m_endPos.m_data[2];

    bc.m_endDir[0] = m_boundaryConditions.m_endDir.m_data[0];
    bc.m_endDir[1] = m_boundaryConditions.m_endDir.m_data[1];
    bc.m_endDir[2] = m_boundaryConditions.m_endDir.m_data[2];
    return bc;
}
}  // namespace twisty