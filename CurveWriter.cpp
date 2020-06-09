#include "CurveWriter.h"

#include <iostream>

namespace twisty
{
    const uint32_t CurveWriter::CurveIdValue = 1 << 0;
    const uint32_t CurveWriter::BezierInfoIdValue = 1 << 1;
    const uint32_t CurveWriter::GtPositionsIdValue = 1 << 2;
    const uint32_t CurveWriter::GtFramesIdValue = 1 << 3;
    const std::string CurveWriter::CurveDataFileExt = ".tcd";

    CurveWriter::CurveWriter()
    {
    }

    void CurveWriter::EndSettingValuesAndWrite(std::string fullFilename)
    {
        if (filename.empty())
        {
            std::cout << "Failed to get filename, try again." << std::endl;
            return;
        }

        std::cout << "Attempt to save to: " << fullFilename << std::endl;

        // Need to get id to read in values
        uint32_t contentId = 0;
        if (m_upCurve)
        {
            contentId |= CurveViewer::CurveIdValue;
        }
        if (m_upBezierInfo)
        {
            contentId |= CurveViewer::BezierInfoIdValue;
        }
        if (m_gtPositions.size() > 0)
        {
            contentId |= CurveViewer::GtPositionsIdValue;
        }
        if (m_gtFrames.size() > 0)
        {
            contentId |= CurveViewer::GtFramesIdValue;
        }

        QString qFullFilename = QString::fromStdString(fullFilename);
        QFile fileToSave(qFullFilename);
        if (!fileToSave.open(QIODevice::ReadWrite))
        {
            std::cout << "Failed to open file: " << fullFilename << std::endl;
            std::cout << "Aborting..." << std::endl;
            return;
        }

        QTextStream stream(&fileToSave);

        // First, we write out the content code
        stream << contentId << "\n";

        // Write curve stuff
        if (contentId & CurveViewer::CurveIdValue)
        {
            stream << m_upCurve->m_numSegments << "\n";
            stream << m_upCurve->m_arclength << "\n";
            stream << m_upCurve->m_basePos.x << " " << m_upCurve->m_basePos.y << " " << m_upCurve->m_basePos.z << "\n";
            stream << m_upCurve->m_baseTangent.x << " " << m_upCurve->m_baseTangent.y << " " << m_upCurve->m_baseTangent.z << "\n";
            stream << m_upCurve->m_baseNormal.x << " " << m_upCurve->m_baseNormal.y << " " << m_upCurve->m_baseNormal.z << "\n";
            stream << m_upCurve->m_baseBinormal.x << " " << m_upCurve->m_baseBinormal.y << " " << m_upCurve->m_baseBinormal.z << "\n";

            stream << m_upCurve->m_targetPos.x << " " << m_upCurve->m_targetPos.y << " " << m_upCurve->m_targetPos.z << "\n";
            stream << m_upCurve->m_targetTangent.x << " " << m_upCurve->m_targetTangent.y << " " << m_upCurve->m_targetTangent.z << "\n";

            // Now we want to write out the segments
            for (uint32_t i = 0; i < m_upCurve->m_segments.size(); ++i)
            {
                stream << m_upCurve->m_segments[i].m_curvature << " " << m_upCurve->m_segments[i].m_torsion << "\n";
            }
        }

        // Write bezier info stuff
        if (contentId & CurveViewer::BezierInfoIdValue)
        {
            stream << m_upBezierInfo->m_controlPt0.x << " " << m_upBezierInfo->m_controlPt0.y << " " << m_upBezierInfo->m_controlPt0.z << "\n";
            stream << m_upBezierInfo->m_controlPt1.x << " " << m_upBezierInfo->m_controlPt1.y << " " << m_upBezierInfo->m_controlPt1.z << "\n";
            stream << m_upBezierInfo->m_controlPt2.x << " " << m_upBezierInfo->m_controlPt2.y << " " << m_upBezierInfo->m_controlPt2.z << "\n";
            stream << m_upBezierInfo->m_controlPt3.x << " " << m_upBezierInfo->m_controlPt3.y << " " << m_upBezierInfo->m_controlPt3.z << "\n";
            stream << m_upBezierInfo->m_controlPt4.x << " " << m_upBezierInfo->m_controlPt4.y << " " << m_upBezierInfo->m_controlPt4.z << "\n";
        }

        if (contentId & CurveViewer::GtPositionsIdValue)
        {
            stream << m_gtPositions.size() << "\n";
            for (uint32_t i = 0; i < m_gtPositions.size(); ++i)
            {
                stream << m_gtPositions[i].x << " " << m_gtPositions[i].y << " " << m_gtPositions[i].z << "\n";
            }
        }

        if (contentId & CurveViewer::GtFramesIdValue)
        {
            stream << m_gtFrames.size() << "\n";
            for (uint32_t i = 0; i < m_gtFrames.size(); ++i)
            {
                stream << m_gtFrames[i].m_rows[0].x << " " << m_gtFrames[i].m_rows[0].y << " " << m_gtFrames[i].m_rows[0].z << "\n";
                stream << m_gtFrames[i].m_rows[1].x << " " << m_gtFrames[i].m_rows[1].y << " " << m_gtFrames[i].m_rows[1].z << "\n";
                stream << m_gtFrames[i].m_rows[2].x << " " << m_gtFrames[i].m_rows[2].y << " " << m_gtFrames[i].m_rows[2].z << "\n";
            }
        }
        std::cout << "Done saving!" << std::endl;
    }
}