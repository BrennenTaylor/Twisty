#include "CurveViewer.h"

#include <QTimer>

#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QInputDialog>
#include <QMouseEvent>
#include <QOpenGLPaintDevice>
#include <QTextStream>

#include <MathConsts.h>

#include <gl/GLU.h>

#include <assert.h>

CurveViewer::CurveViewer(QWidget *pParent, bool parentDrivesUpdate)
    : QOpenGLWidget(pParent)
    , m_parentDrivesUpdate(parentDrivesUpdate)
    , m_initialCurve(1)
    , m_timer()
    , m_drawGrid(true)
    , m_lookAt(0.0f, 0.0f, 0.0f)
    , m_rotateX(0.0f)
    , m_rotateY(0.0f)
    , m_zoom(0.0f)
    , m_cachedX(0)
    , m_cachedY(0)
{
    QSurfaceFormat format;
    format.setSamples(16);
    setFormat(format);

    m_pRenderTimer = new QTimer(this);
    connect(m_pRenderTimer, &QTimer::timeout, this, &CurveViewer::ForceUpdate);
    if (!m_parentDrivesUpdate) {
        m_pRenderTimer->start(16.0);
    }

    m_pAnimationTimer = new QTimer(this);
    connect(m_pAnimationTimer, &QTimer::timeout, this, &CurveViewer::DoAnimationUpdate);
    m_pAnimationTimer->start(100.0);

    setFocusPolicy(Qt::StrongFocus);

    ResetView();
}

void CurveViewer::initializeGL()
{
    makeCurrent();

    initializeOpenGLFunctions();

    // alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_MULTISAMPLE);

    // depth buffer
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);

    // fancy rendering
    glLineWidth(2.0);
    glShadeModel(GL_SMOOTH);
}

void CurveViewer::paintGL()
{
    makeCurrent();

    float windWidth = width();
    float windHeight = height();

    glViewport(0, 0, windWidth, windHeight);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    const float zNear = 0.01f;
    const float zFar = 100.0f;
    gluPerspective(m_zoom, width() / height(), zNear, zFar);

    const Farlor::Vector3 eyePos(0.0f, 0.0f, 30.0f);
    const Farlor::Vector3 worldUp(0.0f, 1.0f, 0.0f);

    gluLookAt(eyePos.x, eyePos.y, eyePos.z, m_lookAt.x, m_lookAt.y, m_lookAt.z, worldUp.x,
          worldUp.y, worldUp.z);


    ClampRotation();

    glMatrixMode(GL_MODELVIEW);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef(m_lookAt.x, m_lookAt.y, m_lookAt.z);
    glRotatef(m_rotateX, 0.0f, 1.0f, 0.0f);
    glRotatef(m_rotateY, 1.0f, 0.0f, 0.0f);
    glTranslatef(-m_lookAt.x, -m_lookAt.y, -m_lookAt.z);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    if (m_drawGrid) {
        RenderGrid();
    }


    RenderPath(m_initialCurve, Farlor::Vector3(0.0f, 1.0f, 1.0f), false);

    // Render look at position
    {
        const float radius = 0.07f;
        glPushMatrix();
        glColor3f(0.0f, 1.0f, 1.0f);
        glTranslatef(m_lookAt.x, m_lookAt.y, m_lookAt.z);
        gluSphere(gluNewQuadric(), radius, 20, 20);
        glPopMatrix();
    }

    // Render Start position
    {
        const float radius = 0.1f;
        glPushMatrix();
        glColor3f(0.0f, 1.0f, 0.0f);
        glTranslatef(m_initialCurve.m_boundaryConditions.m_startPos.x,
              m_initialCurve.m_boundaryConditions.m_startPos.y,
              m_initialCurve.m_boundaryConditions.m_startPos.z);
        gluSphere(gluNewQuadric(), radius, 20, 20);
        glPopMatrix();
    }

    // Render End position
    {
        const float radius = 0.1f;
        glPushMatrix();
        glColor3f(0.0f, 0.0f, 1.0f);
        glTranslatef(m_initialCurve.m_boundaryConditions.m_endPos.x,
              m_initialCurve.m_boundaryConditions.m_endPos.y,
              m_initialCurve.m_boundaryConditions.m_endPos.z);
        gluSphere(gluNewQuadric(), radius, 20, 20);
        glPopMatrix();
    }

    Farlor::Vector3 LowColor(0.0f, 1.0f, 0.0f);
    Farlor::Vector3 HighColor(1.0f, 0.0f, 0.0f);

    // Render custom paths
    if (m_numPaths) {
        const uint32_t numFloatsPerPath = 3 * m_numPointsPerPath;

        // If animated, only draw the one
        if (m_isAnimatedPathPlayback) {
            float interpVal = *(m_pInterpData + m_animatedPathIdx);
            Farlor::Vector3 interpColor = (1.0f - interpVal) * HighColor + interpVal * LowColor;

            float *pPolyStart = m_pPathData + numFloatsPerPath * m_animatedPathIdx;
            RenderPolyline(pPolyStart, m_numPointsPerPath, interpColor);
        } else {
            // We arent animating, draw all paths
            for (uint32_t pathIdx = 0; pathIdx < m_numPaths; pathIdx++) {
                float interpVal = *(m_pInterpData + pathIdx);
                Farlor::Vector3 interpColor = (1.0f - interpVal) * HighColor + interpVal * LowColor;

                float *pPolyStart = m_pPathData + numFloatsPerPath * pathIdx;
                RenderPolyline(pPolyStart, m_numPointsPerPath, interpColor);
            }
        }
    }
}

void CurveViewer::SetInitialCurve(twisty::Curve &curve)
{
    m_initialCurve = curve;
    m_lookAt = (m_initialCurve.m_boundaryConditions.m_endPos
                     + m_initialCurve.m_boundaryConditions.m_startPos)
          * 0.5;
}

void CurveViewer::RenderPolyline(float *pData, uint32_t numPoints, const Farlor::Vector3 &color)
{
    // User should pass in nullptr if no data
    if (!pData) {
        return;
    }

    // If we are here, we assume that the pointer is valid and contains the number of points passed in
    makeCurrent();

    {
        glBegin(GL_LINE_STRIP);

        for (uint32_t pointIdx = 0; pointIdx < numPoints; ++pointIdx) {
            uint32_t leftIdx = pointIdx * 3;

            {
                glColor3f(color.x, color.y, color.z);

                Farlor::Vector3 point;
                point.x = pData[leftIdx + 0];
                point.y = pData[leftIdx + 1];
                point.z = pData[leftIdx + 2];

                glVertex3f(point.x, point.y, point.z);
            }
        }

        glEnd();
    }
}

void CurveViewer::RenderCurve(const twisty::Curve &curve)
{
    makeCurrent();

    std::vector<float> scales;
    const float segmentLength = curve.m_ds;

    // Draw based on those positions and frames
    {
        glBegin(GL_LINES);

        for (uint32_t i = 0; i < curve.m_positions.size() - 1; ++i) {
            auto &x_j = curve.m_positions[i];
            auto &x_j_1 = curve.m_positions[i + 1];

            {
                glColor3f(1.0f, 0.0f, 0.0f);
                glVertex3f(x_j.x, x_j.y, x_j.z);
                glVertex3f(x_j_1.x, x_j_1.y, x_j_1.z);
            }
        }

        glEnd();
    }
}

void CurveViewer::RenderPath(
      const twisty::Curve &curve, const Farlor::Vector3 &color, bool renderSegmentFrames)
{
    makeCurrent();

    std::vector<float> scales;
    const float segmentLength = curve.m_ds;
    {
        glBegin(GL_LINES);

        for (uint32_t i = 0; i < curve.m_positions.size() - 1; ++i) {
            auto &x_j = curve.m_positions[i];
            auto &x_j_1 = curve.m_positions[i + 1];

            if (renderSegmentFrames) {
                glColor3f(1.0f, 0.0f, 0.0f);
            } else {
                glColor3f(color.x, color.y, color.z);
            }
            glVertex3f(x_j.x, x_j.y, x_j.z);
            glVertex3f(x_j_1.x, x_j_1.y, x_j_1.z);
        }
        glEnd();
    }

    //std::cin.get();
}

void CurveViewer::RenderGrid()
{
    makeCurrent();

    const int32_t m = 10;
    const int32_t n = 10;

    int x0 = int(-m / 2);
    int z0 = int(-n / 2);
    for (int32_t x = 0; x < m; x++) {
        for (int32_t z = 0; z < n; z++) {
            glBegin(GL_QUADS);
            glVertex3f(x0 + x, 0, z0 + z);
            glVertex3f(x0 + x + 1, 0, z0 + z);
            glVertex3f(x0 + x + 1, 0, z0 + z + 1);
            glVertex3f(x0 + x, 0, z0 + z + 1);
            glEnd();
        }
    }
}

void CurveViewer::ResetView()
{
    makeCurrent();

    m_rotateX = 0.0f;
    m_rotateY = 0.0f;
    m_zoom = 50.0f;
}

void CurveViewer::mousePressEvent(QMouseEvent *pEvent)
{
    makeCurrent();

    m_cachedX = pEvent->x();
    m_cachedY = pEvent->y();
}

void CurveViewer::mouseReleaseEvent(QMouseEvent *pEvent) { }

void CurveViewer::mouseMoveEvent(QMouseEvent *pEvent)
{
    makeCurrent();

    int newX = pEvent->pos().x();
    int newY = pEvent->pos().y();

    int dx = newX - m_cachedX;
    int dy = newY - m_cachedY;

    m_cachedX = newX;
    m_cachedY = newY;

    if (pEvent->buttons() & Qt::MouseButton::LeftButton) {
        float rotateSpeed = 1.0f;
        m_rotateX += dx * rotateSpeed;
        m_rotateY += dy * rotateSpeed;
        ClampRotation();
    }
    // We want to update rotation based on dx and dy
    update();
}

void CurveViewer::wheelEvent(QWheelEvent *pEvent)
{
    makeCurrent();

    const float zoomMin = 0.1f;
    const float zoomMax = 1000.0f;
    float zoomFactor = pEvent->delta() / 120.0f;
    float zoomSpeed = 1.0f;
    // We want to flip to maintain that scrolling up is a zoom in and vice versa
    m_zoom += zoomFactor * zoomSpeed * -1.0f;
    m_zoom = std::max(m_zoom, zoomMin);
    m_zoom = std::min(m_zoom, zoomMax);
}

void CurveViewer::keyPressEvent(QKeyEvent *pEvent)
{
    makeCurrent();

    const float lookAtMoveSpeed = 0.1f;

    switch (pEvent->key()) {
        case Qt::Key_G: {
            emit CurveReset();
        } break;

        case Qt::Key_J: {
            m_lookAt.x -= lookAtMoveSpeed;
        } break;

        case Qt::Key_L: {
            m_lookAt.x += lookAtMoveSpeed;
        } break;

        case Qt::Key_K: {
            m_lookAt.y -= lookAtMoveSpeed;
        } break;

        case Qt::Key_I: {
            m_lookAt.y += lookAtMoveSpeed;
        } break;

        case Qt::Key_U: {
            m_lookAt.z -= lookAtMoveSpeed;
        } break;

        case Qt::Key_O: {
            m_lookAt.z += lookAtMoveSpeed;
        } break;

        default: {
            return QWidget::keyPressEvent(pEvent);
        } break;
    }
}

void CurveViewer::keyReleaseEvent(QKeyEvent *pEvent) { }

void CurveViewer::ClampRotation()
{
    while (m_rotateX > 360.0f) {
        m_rotateX -= 360.0f;
    }

    while (m_rotateX < 0.0f) {
        m_rotateX += 360.0f;
    }

    while (m_rotateY > 360.0f) {
        m_rotateY -= 360.0f;
    }

    while (m_rotateY < 0.0f) {
        m_rotateY += 360.0f;
    }
}

void CurveViewer::DoAnimationUpdate()
{
    m_animatedPathIdx++;
    if (m_animatedPathIdx >= m_numPaths) {
        m_animatedPathIdx = 0;
    }
    emit AnimatedCurveIdxChanged(m_animatedPathIdx);
}