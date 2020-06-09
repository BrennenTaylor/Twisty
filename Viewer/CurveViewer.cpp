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

CurveViewer::CurveViewer(QWidget* pParent, bool parentDrivesUpdate)
    : QOpenGLWidget(pParent)
    , m_parentDrivesUpdate(parentDrivesUpdate)
    , m_upInitialCurve(nullptr)
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
    if (!m_parentDrivesUpdate)
    {
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

    gluLookAt(
        eyePos.x, eyePos.y, eyePos.z,
        m_lookAt.x, m_lookAt.y, m_lookAt.z,
        worldUp.x, worldUp.y, worldUp.z
    );


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

    if (m_drawGrid)
    {
        RenderGrid();
    }

    if (m_upInitialCurve)
    {
        RenderPath(*m_upInitialCurve, Farlor::Vector3(0.0f, 1.0f, 0.0f), false);
    }

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
        glTranslatef(m_upInitialCurve->m_basePos.x, m_upInitialCurve->m_basePos.y, m_upInitialCurve->m_basePos.z);
        gluSphere(gluNewQuadric(), radius, 20, 20);
        glPopMatrix();
    }

    // Render End position
    {
        const float radius = 0.1f;
        glPushMatrix();
        glColor3f(0.0f, 0.0f, 1.0f);
        glTranslatef(m_upInitialCurve->m_targetPos.x, m_upInitialCurve->m_targetPos.y, m_upInitialCurve->m_basePos.z);
        gluSphere(gluNewQuadric(), radius, 20, 20);
        glPopMatrix();
    }

    // Render custom paths
    if (m_numPaths)
    {
        const uint32_t numFloatsPerPath = 3 * m_numPointsPerPath;

        // If animated, only draw the one
        if (m_isAnimatedPathPlayback)
        {
            float* pPolyStart = m_pPathData + numFloatsPerPath * m_animatedPathIdx;
            RenderPolyline(pPolyStart, m_numPointsPerPath);
        }
        else
        {
            // We arent animating, draw all paths
            for (uint32_t pathIdx = 0; pathIdx < m_numPaths; pathIdx++)
            {
                float* pPolyStart = m_pPathData + numFloatsPerPath * pathIdx;
                RenderPolyline(pPolyStart, m_numPointsPerPath);
            }
        }
    }
}

void CurveViewer::SetInitialCurve(twisty::Curve& curve)
{
    m_upInitialCurve = std::make_unique<twisty::Curve>(curve);
    m_lookAt = (m_upInitialCurve->m_targetPos + m_upInitialCurve->m_basePos) * 0.5;
}

void CurveViewer::RenderPolyline(float* pData, uint32_t numPoints)
{
    // User should pass in nullptr if no data
    if (!pData)
    {
        return;
    }

    // If we are here, we assume that the pointer is valid and contains the number of points passed in
    makeCurrent();

    {
        glBegin(GL_LINE_STRIP);

        for (uint32_t pointIdx = 0; pointIdx < numPoints; ++pointIdx)
        {
            uint32_t leftIdx = pointIdx * 3;
            //uint32_t rightIdx = (pointIdx + 1) * 3;

            {
                glColor3f(1.0f, 0.0f, 0.0f);

                Farlor::Vector3 leftPt;
                leftPt.x = pData[leftIdx + 0];
                leftPt.y = pData[leftIdx + 1];
                leftPt.z = pData[leftIdx + 2];

                /*Farlor::Vector3 rightPt;
                rightPt.x = pData[rightIdx + 0];
                rightPt.y = pData[rightIdx + 1];
                rightPt.z = pData[rightIdx + 2];*/

                glVertex3f(leftPt.x, leftPt.y, leftPt.z);
                //glVertex3f(rightPt.x, rightPt.y, rightPt.z);
            }
        }

        glEnd();
    }
}

void CurveViewer::RenderCurve(const twisty::Curve& curve)
{
    makeCurrent();

    std::vector<float> scales;
    const float segmentLength = curve.m_arclength / curve.m_numSegments;


    std::vector<Farlor::Vector3> positions;
    std::vector<Farlor::Vector3> tangents;

    curve.ReconstructCurvePositionsAndFramesFirstOrder(positions, tangents);
    positions.push_back(m_upInitialCurve->m_targetPos);

    // Single push back for first segment
    scales.push_back(1.0f);
    for (uint32_t i = 0; i < curve.m_numSegments; ++i)
    {
        // Handle calculation of draw scale
        float curvatureScale = 1.0f;

        scales.push_back(curvatureScale);
    }

    // Draw based on those positions and frames
    {
        glBegin(GL_LINES);

        for (uint32_t i = 0; i < positions.size() - 1; ++i)
        {
            auto& x_j = positions[i];
            auto& x_j_1 = positions[i+1];
            
            float scale = scales[i];
            {
                glColor3f(1.0f, 0.0f, 0.0f);
                glVertex3f(x_j.x, x_j.y, x_j.z);
                //Farlor::Vector3 tan = x_j + t_j * segmentLength;
                glVertex3f(x_j_1.x, x_j_1.y, x_j_1.z);
            }
        }

        glEnd();
    }

}

void CurveViewer::RenderPath(const twisty::Curve& curve, const Farlor::Vector3& color, bool renderSegmentFrames)
{
    makeCurrent();

    std::vector<float> scales;
    const float segmentLength = curve.m_arclength / curve.m_numSegments;

    std::vector<Farlor::Vector3> positions;
    std::vector<Farlor::Vector3> tangents;

    curve.ReconstructCurvePositionsAndFramesFirstOrder(positions, tangents);
    positions.push_back(m_upInitialCurve->m_targetPos);
    {
        glBegin(GL_LINES);

        for (uint32_t i = 0; i < positions.size() - 1; ++i)
        {
            auto& x_j = positions[i];
            auto& x_j_1 = positions[i + 1];

            //std::cout << "Position: " << x_j << std::endl;
            //std::cout << "Tangent: " << t_j << std::endl;

            if (renderSegmentFrames)
            {
                glColor3f(1.0f, 0.0f, 0.0f);
            }
            else
            {
                glColor3f(color.x, color.y, color.z);
            }
            glVertex3f(x_j.x, x_j.y, x_j.z);
            //Farlor::Vector3 tan = x_j + t_j * segmentLength;
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
    for (int32_t x = 0; x < m; x++)
    {
        for (int32_t z = 0; z < n; z++)
        {
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

void CurveViewer::mousePressEvent(QMouseEvent* pEvent)
{
    makeCurrent();

    m_cachedX = pEvent->x();
    m_cachedY = pEvent->y();
}

void CurveViewer::mouseReleaseEvent(QMouseEvent* pEvent)
{
}

void CurveViewer::mouseMoveEvent(QMouseEvent* pEvent)
{
    makeCurrent();

    int newX = pEvent->pos().x();
    int newY = pEvent->pos().y();

    int dx = newX - m_cachedX;
    int dy = newY - m_cachedY;

    m_cachedX = newX;
    m_cachedY = newY;

    if (pEvent->buttons() & Qt::MouseButton::LeftButton)
    {
        float rotateSpeed = 1.0f;
        m_rotateX += dx * rotateSpeed;
        m_rotateY += dy * rotateSpeed;
        ClampRotation();
    }
    // We want to update rotation based on dx and dy
    update();
}

void CurveViewer::wheelEvent(QWheelEvent* pEvent)
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

void CurveViewer::keyPressEvent(QKeyEvent* pEvent)
{
    makeCurrent();

    const float lookAtMoveSpeed = 0.1f;

    switch (pEvent->key())
    {
    case Qt::Key_G:
    {
        PrepareCurveReset();
        emit CurveReset();
    } break;

    case Qt::Key_J:
    {
        m_lookAt.x -= lookAtMoveSpeed;
    } break;

    case Qt::Key_L:
    {
        m_lookAt.x += lookAtMoveSpeed;
    } break;

    case Qt::Key_K:
    {
        m_lookAt.y -= lookAtMoveSpeed;
    } break;

    case Qt::Key_I:
    {
        m_lookAt.y += lookAtMoveSpeed;
    } break;

    case Qt::Key_U:
    {
        m_lookAt.z -= lookAtMoveSpeed;
    } break;

    case Qt::Key_O:
    {
        m_lookAt.z += lookAtMoveSpeed;
    } break;

    default:
    {
        return QWidget::keyPressEvent(pEvent);
    } break;
    }
}

void CurveViewer::keyReleaseEvent(QKeyEvent* pEvent)
{
}

void CurveViewer::PrepareCurveReset()
{
    m_upInitialCurve = nullptr;
}

void CurveViewer::ClampRotation()
{
    while (m_rotateX > 360.0f)
    {
        m_rotateX -= 360.0f;
    }

    while (m_rotateX < 0.0f)
    {
        m_rotateX += 360.0f;
    }

    while (m_rotateY > 360.0f)
    {
        m_rotateY -= 360.0f;
    }

    while (m_rotateY < 0.0f)
    {
        m_rotateY += 360.0f;
    }
}

void CurveViewer::DoAnimationUpdate()
{
    m_animatedPathIdx++;
    if (m_animatedPathIdx >= m_numPaths)
    {
        m_animatedPathIdx = 0;
    }
    emit AnimatedCurveIdxChanged(m_animatedPathIdx);
}