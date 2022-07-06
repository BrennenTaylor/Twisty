#pragma once

#include <QBasicTimer>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include "Bootstrapper.h"
#include "Curve.h"
#include "ExperimentRunner.h"
#include "PathBatchViewerWidget.h"

#include <filesystem>
#include <vector>

class QTimer;

class CurveViewer
    : public QOpenGLWidget
    , public QOpenGLFunctions {
    Q_OBJECT

   public:
    CurveViewer(QWidget *pParent = nullptr, bool parentDrivesUpdate = false);

    void initializeGL() override;
    void paintGL() override;
    //void resizeGL(int w, int h) override;

    void SetInitialCurve(twisty::Curve &curve);

    void mousePressEvent(QMouseEvent *pEvent) override;
    void mouseReleaseEvent(QMouseEvent *pEvent) override;
    void mouseMoveEvent(QMouseEvent *pEvent) override;
    void wheelEvent(QWheelEvent *pEvent) override;

    void keyPressEvent(QKeyEvent *pEvent) override;
    void keyReleaseEvent(QKeyEvent *pEvent) override;

    twisty::Curve &GetInitialCurve() { return m_initialCurve; }

    void ForceUpdate() { update(); }

    void SetPathDrawData(
          float *pPathData, float *pInterpData, uint32_t numPaths, uint32_t numPointsPerPath)
    {
        m_pPathData = pPathData;
        m_pInterpData = pInterpData;
        m_numPaths = numPaths;
        m_numPointsPerPath = numPointsPerPath;
        m_animatedPathIdx = 0;
    }

    void ResetPathDrawData()
    {
        m_pPathData = nullptr;
        m_numPaths = 0;
        m_numPointsPerPath = 0;
        m_animatedPathIdx = 0;
    }

    void EnablePathAnimation()
    {
        m_isAnimatedPathPlayback = true;
        m_animatedPathIdx = 0;
    }

    void DisablePathAnimation() { m_isAnimatedPathPlayback = false; }

    uint32_t GetCurrentAnimatedIdx() { return m_animatedPathIdx; }


    // Signals
   signals:
    void CurveReset();
    void AnimatedCurveIdxChanged(uint32_t idx);

   private:
    void RenderPolyline(float *pData, uint32_t numPoints, const Farlor::Vector3 &color);
    void RenderCurve(const twisty::Curve &curve);
    void RenderPath(
          const twisty::Curve &curve, const Farlor::Vector3 &color, bool renderSegmentFrames);
    void RenderGrid();

    void ResetView();

    void PrepareCurveReset();

    void ClampRotation();

    void DoAnimationUpdate();

   private:
    QTimer *m_pRenderTimer = nullptr;
    QTimer *m_pAnimationTimer = nullptr;

    bool m_parentDrivesUpdate = false;
    twisty::Curve m_initialCurve;

    QBasicTimer m_timer;

    bool m_drawGrid;

    Farlor::Vector3 m_lookAt;
    float m_rotateX;
    float m_rotateY;
    float m_zoom;
    int32_t m_cachedX;
    int32_t m_cachedY;

    float *m_pPathData = nullptr;
    float *m_pInterpData = nullptr;
    uint32_t m_numPaths = 0;
    uint32_t m_numPointsPerPath = 0;

    bool m_isAnimatedPathPlayback = false;
    uint32_t m_animatedPathIdx = 0;
};