#pragma once

#include "PathWeightUtils.h"

#include <QBasicTimer>

#include <QOpenGLWidget>
#include <QOpenGLFunctions>

#include "Bootstrapper.h"
#include "Curve.h"
#include "ExperimentRunner.h"
#include "PathPerturbViewerWidget.h"

#include <filesystem>
#include <vector>

class RunningCurveViewer : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT

public:
    RunningCurveViewer(PathPerturbViewerWidget& pathPerturbViewerWidget, QWidget* pParent = nullptr);

    void InitializeExperiment();

    void initializeGL() override;
    void paintGL() override;

    //void resizeGL(int w, int h) override;

    void SetInitialCurve(twisty::Curve& curve);
    void SetBezierInfo(twisty::Bootstrapper::BezierInfo& bezierInfo);

    void SetCurveCacheSize(uint32_t newSize);

    void SetGtPositions(std::vector<Farlor::Vector3>& gtPositions);
    void SetGtFrames(std::vector<Farlor::Matrix3x3>& gtFrames);

    void mousePressEvent(QMouseEvent* pEvent) override;
    void mouseReleaseEvent(QMouseEvent* pEvent) override;
    void mouseMoveEvent(QMouseEvent* pEvent) override;
    void wheelEvent(QWheelEvent* pEvent) override;

    void keyPressEvent(QKeyEvent* pEvent) override;
    void keyReleaseEvent(QKeyEvent* pEvent) override;

    twisty::Curve& GetInitialCurve()
    {
        return *m_upInitialCurve;
    }

// Signals
signals:
    void CurveReset();

private:
    void RenderCurve(const twisty::Curve& curve);
    void RenderPath(const twisty::Curve& curve, const Farlor::Vector3& color, bool renderSegmentFrames, float transparency = 1.0f);
    void RenderBezierInfo(const twisty::Bootstrapper::BezierInfo& bezierInfo);
    void RenderBezierPositions();
    void RenderBezierFrames();
    void RenderGrid();

    void ResetView();

    void PrepareCurveReset();

    void ClampRotation();

    void UpdateWorkaround() { update(); }

    std::unique_ptr<twisty::Curve> SimpleGeometryCurvePerturb(const twisty::Curve& curve, uint32_t& flag);
    std::unique_ptr<twisty::Curve> PurturbCurve(const twisty::Curve& curve, uint32_t& flag);

private:
    PathPerturbViewerWidget& m_pathPerturbViewerWidget;

    std::unique_ptr<twisty::Curve> m_upInitialCurve;
    std::vector<twisty::Curve> m_curveCache;
    std::vector<uint32_t> m_curveAgeCache;
    uint32_t m_curveCacheSize;
    uint32_t m_curveCacheIdx = 0;

    std::unique_ptr<twisty::Bootstrapper::BezierInfo> m_upBezierInfo;

    std::vector<Farlor::Vector3> m_gtPositions;
    std::vector<Farlor::Matrix3x3> m_gtFrames;

    QBasicTimer m_timer;

    bool m_drawGrid;
    bool m_scaledCurvature;
    
    bool m_drawPoints;

    Farlor::Vector3 m_lookAt;


    float m_rotateX;
    float m_rotateY;
    float m_zoom;

    int m_cachedX;
    int m_cachedY;

    bool m_isInitialized;


    twisty::Curve m_curveToBend;

    std::mt19937 m_rng;
};