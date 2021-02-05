#include "RunningCurveViewer.h"

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

RunningCurveViewer::RunningCurveViewer(PathPerturbViewerWidget& pathPerturbViewerWidget, QWidget* pParent)
    : QOpenGLWidget(pParent)
    , m_pathPerturbViewerWidget(pathPerturbViewerWidget)
    , m_upInitialCurve(nullptr)
    , m_curveCache(1)
    , m_curveAgeCache(1)
    , m_curveCacheSize(1)
    , m_curveCacheIdx(0)
    , m_upBezierInfo(nullptr)
    , m_gtPositions()
    , m_gtFrames()
    , m_timer()
    , m_drawGrid(true)
    , m_scaledCurvature(false)
    , m_drawPoints(false)
    , m_lookAt(0.0f, 0.0f, 0.0f)
    , m_rotateX(0.0f)
    , m_rotateY(0.0f)
    , m_zoom(0.0f)
    , m_cachedX(0)
    , m_cachedY(0)
    , m_isInitialized(false)
    , m_curveToBend()
    , m_rng()
{
    QTimer* pTimer = new QTimer(this);
    connect(pTimer, &QTimer::timeout, this, &RunningCurveViewer::UpdateWorkaround);
    pTimer->start(16.0);

    setFocusPolicy(Qt::StrongFocus);

    ResetView();

    uint32_t seed = 0;
    if (seed == 0)
    {
        seed = time(0);
    }
    std::cout << "Purturb seed used: " << seed << std::endl;
    m_rng = std::mt19937(seed);
}

void RunningCurveViewer::initializeGL()
{
    makeCurrent();

    initializeOpenGLFunctions();

    // alpha blending
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // depth buffer
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);

    // fancy rendering
    glLineWidth(2.0);
    glShadeModel(GL_SMOOTH);
}

const bool DetailedPerturb = false;

Farlor::Matrix3x3  RotationMatrixAroundAxis(float angle, Farlor::Vector3 axis)
{
    // Ensure its normalized
    axis.Normalize();

    Farlor::Matrix3x3 rotation(
        Farlor::Vector3(
            cos(angle) + axis.x * axis.x * (1.0f - cos(angle)),
            axis.x * axis.y * (1.0f - cos(angle)) - axis.z * sin(angle),
            axis.x * axis.z * (1.0f - cos(angle)) + axis.y * sin(angle)
        ),
        Farlor::Vector3(
            axis.y * axis.x * (1.0f - cos(angle)) + axis.z * sin(angle),
            cos(angle) + axis.y * axis.y * (1 - cos(angle)),
            axis.y * axis.z * (1 - cos(angle)) - axis.x * sin(angle)
        ),
        Farlor::Vector3(
            axis.z * axis.x * (1 - cos(angle)) - axis.y * sin(angle),
            axis.z * axis.y * (1 - cos(angle)) + axis.x * sin(angle),
            cos(angle) + axis.z * axis.z * (1 - cos(angle))
        )
    );
    return rotation;
}

std::pair<float, float> CurvatureAndTorsionBetweenTwoFrames(const Farlor::Matrix3x3& startFrame, const Farlor::Matrix3x3& endFrame, float segmentLength)
{
    std::pair<float, float> curvatureAndTorsion = { 0.0f, 0.0f };
    {
        float curvature = ((endFrame.m_rows[0] - startFrame.m_rows[0]) * (1.0f / segmentLength)).Magnitude();
        curvatureAndTorsion.first = curvature;
    }

    {
        auto torsionLeft = -1.0f * startFrame.m_rows[1];
        auto torsionRight = (endFrame.m_rows[2] - startFrame.m_rows[2]) * (1.0f / segmentLength);
        float torsion = torsionLeft.Dot(torsionRight);
        curvatureAndTorsion.second = torsion;
    }
    return curvatureAndTorsion;
}


std::unique_ptr<twisty::Curve> RunningCurveViewer::SimpleGeometryCurvePerturb(const twisty::Curve& curve, uint32_t& flag)
{
    if (DetailedPerturb)
    {
        std::cout << "Begin Purturb --------" << std::endl;
    }

    std::unique_ptr<twisty::Curve> upNewCurve = std::make_unique<twisty::Curve>(curve);

    // We bound on left by one as we dont want to rotate the first segment at all
    // Left bound by m-2 as we at least want there to be one point between the left and right points selected so an actual perturbation occurs
    std::uniform_int_distribution<int> leftPointIndexUniformDist(1, upNewCurve->m_numSegments - 3); // uniform, unbiased
    int32_t leftPointIndex = leftPointIndexUniformDist(m_rng);
    std::uniform_int_distribution<int> rightPointIndexUniformDist(leftPointIndex + 2, upNewCurve->m_numSegments - 1); // uniform, unbiased
    int32_t rightPointIndex = rightPointIndexUniformDist(m_rng);
    /*
            leftPointIndex = 1;
            rightPointIndex = 9;*/

    assert(leftPointIndex < rightPointIndex);
    assert((rightPointIndex - leftPointIndex) >= 2);

    if (DetailedPerturb)
    {
        std::cout << "\tLeft Index: " << leftPointIndex << std::endl;
        std::cout << "\tRight Index: " << rightPointIndex << std::endl;
    }

    // 0 - 2 PI uniform distribution
    //std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * 0.01f, TwistyPi * 0.01f);
    //std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * 0.1f, TwistyPi * 0.1f);
    std::uniform_real_distribution<float> zeroToTwoPiUniformDist( -TwistyPi,  TwistyPi);

    // End targets of purturbation
    Farlor::Vector3 targetN = m_upInitialCurve->m_targetTangent;
    Farlor::Vector3 targetP = m_upInitialCurve->m_targetPos;


    /** This is where the fun begins **/
    std::vector<Farlor::Vector3> points;
    // All but the last point
    upNewCurve->ReconstructCurvePositionsFirstOrder(points);

    // We need two frames for each segment to get the new curvature and torsion.
    // we need the frame left of the segment, as well as the frame right of the segment.

    // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
    Farlor::Vector3 leftPoint = points[leftPointIndex];
    Farlor::Vector3 rightPoint = points[rightPointIndex];
    Farlor::Vector3 axisOfRotation = (rightPoint - leftPoint).Normalized();

    float randomAngle = zeroToTwoPiUniformDist(m_rng);
    //randomAngle = 0.0f;
    Farlor::Matrix3x3 rotationMatrix = RotationMatrixAroundAxis(randomAngle, axisOfRotation);

    if (DetailedPerturb)
    {
        std::cout << "\tRotation Info: " << std::endl;
        std::cout << "\t\tRandom angle: " << randomAngle << std::endl;
        std::cout << "\t\tRotationMatrix: " << rotationMatrix << std::endl;
    }

    // Lets build up the new poly line.
    std::vector<Farlor::Vector3> updatedPolyline;

    // First, lets add in the points before the rotation. These experience no rotation.
    for (uint32_t pointIdx = 0; pointIdx <= leftPointIndex; pointIdx++)
    {
        updatedPolyline.push_back(points[pointIdx]);
    }

    // Now, we do the rotated points
    for (uint32_t pointIdx = leftPointIndex + 1; pointIdx < rightPointIndex; pointIdx++)
    {
        Farlor::Vector3 pointToRotate = points[pointIdx];
        Farlor::Vector3 shiftedPoint = pointToRotate - leftPoint;
        Farlor::Vector3 rotatedPoint = rotationMatrix * shiftedPoint;
        Farlor::Vector3 finalPoint = rotatedPoint + leftPoint;
        updatedPolyline.push_back(finalPoint);
    }

    // Finally, we get those at the end after the rotation occures
    for (uint32_t pointIdx = rightPointIndex; pointIdx < points.size(); pointIdx++)
    {
        updatedPolyline.push_back(points[pointIdx]);
    }
    updatedPolyline.push_back(m_upInitialCurve->m_targetPos);

    assert(points.size() + 1 == updatedPolyline.size());

    // Now that we have the polyline, we want to construct the reference frames.
    // Now, we build up reference frames.
    std::vector<Farlor::Vector3> tangents;

    // For now, simply compute the difference in positions.
    // We can do a different approach later.
    for (uint32_t posIdx = 0; posIdx < updatedPolyline.size() - 1; ++posIdx)
    {
        Farlor::Vector3 tangent = updatedPolyline[posIdx + 1] - updatedPolyline[posIdx];
        tangent = tangent.Normalized();
        tangents.push_back(tangent);
    }

    // End Frame
    // We only really need the tangent for it
    {
        tangents.push_back(m_upInitialCurve->m_targetTangent);
    }

    assert(tangents.size() == upNewCurve->m_numSegments + 1);

    for (uint32_t i = 0; i < upNewCurve->m_numSegments; ++i)
    {
        float curvature = ((tangents[i + 1] - tangents[i]) * (1.0f / upNewCurve->m_segmentLength)).Magnitude();
        upNewCurve->m_curvatures[i] = curvature;
        upNewCurve->m_positions[i] = updatedPolyline[i];
        upNewCurve->m_tangents[i] = tangents[i];
    }

    return upNewCurve;
}

std::unique_ptr<twisty::Curve> RunningCurveViewer::PurturbCurve(const twisty::Curve& curve, uint32_t& flag)
{
    if (DetailedPerturb)
    {
        std::cout << "Performing purturb" << std::endl;
    }

    // Actually do the purturbation
    std::unique_ptr<twisty::Curve> upNewCurve = nullptr;
    upNewCurve = SimpleGeometryCurvePerturb(curve, flag);
    //else if (m_experimentParams.curvePerturbMethod == CurvePerturbMethod::ComplexGeometry)
    //{
    //    upNewCurve = ComplexGeometryCurvePerturb(curve, flag);
    //}
    //else if (m_experimentParams.curvePerturbMethod == CurvePerturbMethod::RootSolve)
    //{
    //    upNewCurve = RootSolveCurvePerturb(curve, flag);
    //}

    return upNewCurve;
}

void RunningCurveViewer::paintGL()
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

    glRotatef(m_rotateX, 0.0f, 1.0f, 0.0f);
    glRotatef(m_rotateY, 1.0f, 0.0f, 0.0f);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

    if (m_drawGrid)
    {
        RenderGrid();
    }

    const Farlor::Vector3 newColor(0.0f, 1.0f, 0.0f);
    const Farlor::Vector3 oldColor(1.0f, 0.0f, 0.0f);
    for (uint32_t curveIdx = 0; curveIdx < m_curveCache.size(); curveIdx++)
    {
        float percentAged = static_cast<float>(m_curveAgeCache[curveIdx]) / static_cast<float>(m_curveCacheSize);
        Farlor::Vector3 curveColor = (1.0f - percentAged) * newColor + percentAged * oldColor;

        RenderPath(m_curveCache[curveIdx], curveColor, false, (1.0f - percentAged));

        m_curveAgeCache[curveIdx]++;
    }

    if (m_upInitialCurve)
    {
        //RenderCurve(*m_upInitialCurve);
        RenderPath(*m_upInitialCurve, Farlor::Vector3(0.0f, 1.0f, 1.0f), false);
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
        glColor3f(1.0f, 0.0f, 0.0f);
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

    // Generate next curve
    if (m_isInitialized)
    {
        twisty::Curve seedCurve = m_curveToBend;

        uint32_t flag = 0;
        std::unique_ptr<twisty::Curve> upNewCurve = PurturbCurve(m_curveToBend, flag);

        if (!upNewCurve)
        {
            std::cout << "Error: We really should have a curve. Skipping this curve" << std::endl;
        }

        m_curveToBend = *upNewCurve;
        m_curveCache[m_curveCacheIdx] = m_curveToBend;
        m_curveAgeCache[m_curveCacheIdx] = 0;

        float ds = m_upInitialCurve->m_arclength / 200.0f;
        float scatter = 0.08f / ds;

        // Final increments
        m_curveCacheIdx = (m_curveCacheIdx + 1) % m_curveCacheSize;
    }
}

// Do initialization stuff
void RunningCurveViewer::InitializeExperiment()
{
    uint32_t numFailures = 0;
    uint32_t totalFailures = 0;
    uint32_t totalSuccess = 0;

    m_isInitialized = true;
}

void RunningCurveViewer::SetCurveCacheSize(uint32_t newSize)
{
    m_curveCacheSize = newSize;
    m_curveCache.resize(m_curveCacheSize);
    m_curveAgeCache.resize(m_curveCacheSize);
}

void RunningCurveViewer::SetInitialCurve(twisty::Curve& curve)
{
    m_upInitialCurve = std::make_unique<twisty::Curve>(curve);
    m_curveToBend = *m_upInitialCurve;
}

void RunningCurveViewer::SetBezierInfo(twisty::Bootstrapper::BezierInfo& bezierInfo)
{
    m_upBezierInfo = std::make_unique<twisty::Bootstrapper::BezierInfo>(bezierInfo);
}

void RunningCurveViewer::SetGtPositions(std::vector<Farlor::Vector3>& gtPositions)
{
    m_gtPositions = gtPositions;
}

void RunningCurveViewer::SetGtFrames(std::vector<Farlor::Matrix3x3>& gtFrames)
{
     m_gtFrames = gtFrames;
}

void RunningCurveViewer::RenderCurve(const twisty::Curve& curve)
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
        if (m_scaledCurvature)
        {
            curvatureScale *= curve.m_curvatures[i];
        }

        scales.push_back(curvatureScale);
    }

    // Draw based on those positions and frames

    if (m_drawPoints)
    {
        glPointSize(3.0f);
        glBegin(GL_POINTS);
        glColor3f(1.0f, 1.0f, 1.0f);
        for (uint32_t i = 0; i < positions.size(); ++i)
        {
            glVertex3f(positions[i].x, positions[i].y, positions[i].z);
        }
        glEnd();
    }

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

void RunningCurveViewer::RenderPath(const twisty::Curve& curve, const Farlor::Vector3& color, bool renderSegmentFrames, float transparency)
{
    makeCurrent();

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

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
                glColor4f(1.0f, 0.0f, 0.0f, transparency);
            }
            else
            {
                glColor4f(color.x, color.y, color.z, transparency);
            }
            glVertex3f(x_j.x, x_j.y, x_j.z);
            //Farlor::Vector3 tan = x_j + t_j * segmentLength;
            glVertex3f(x_j_1.x, x_j_1.y, x_j_1.z);
        }
        glEnd();
    }


    glDisable(GL_BLEND);
    //std::cin.get();

}

void RunningCurveViewer::RenderBezierPositions()
{
    makeCurrent();

    glBegin(GL_LINES);

    for (uint32_t i = 0; i < m_gtPositions.size() - 1; ++i)
    {
        auto& first = m_gtPositions[i];
        auto& second = m_gtPositions[i + 1];

        glColor3f(1.0f, 0.0f, 1.0f);
        glVertex3f(first.x, first.y, first.z);
        glVertex3f(second.x, second.y, second.z);
    }

    glEnd();
}

void RunningCurveViewer::RenderBezierFrames()
{
    makeCurrent();

    glBegin(GL_LINES);

    assert(m_gtPositions.size() == m_gtFrames.size());

    for (uint32_t i = 0; i < m_gtFrames.size() - 1; ++i)
    {
        float curvatureScale = 1.0f;
        if (m_scaledCurvature)
        {
            curvatureScale = m_upInitialCurve->m_curvatures[i];
        }

        auto& position = m_gtPositions[i];
        auto& frame = m_gtFrames[i];
        const float ds = m_upInitialCurve->m_segmentLength;

        // Draw the frame on the point
        {
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(position.x, position.y, position.z);
            Farlor::Vector3 tan = position + frame.m_rows[0] * ds;
            glVertex3f(tan.x, tan.y, tan.z);
        }
    }

    glEnd();
}

void RunningCurveViewer::RenderBezierInfo(const twisty::Bootstrapper::BezierInfo& bezierInfo)
{
    makeCurrent();

    const float radius = 0.07f;
    glPushMatrix();
    glTranslatef(bezierInfo.m_controlPt0.x, bezierInfo.m_controlPt0.y, bezierInfo.m_controlPt0.z);
    gluSphere(gluNewQuadric(), radius, 20, 20);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(bezierInfo.m_controlPt1.x, bezierInfo.m_controlPt1.y, bezierInfo.m_controlPt1.z);
    gluSphere(gluNewQuadric(), radius, 20, 20);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(bezierInfo.m_controlPt2.x, bezierInfo.m_controlPt2.y, bezierInfo.m_controlPt2.z);
    gluSphere(gluNewQuadric(), radius, 20, 20);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(bezierInfo.m_controlPt3.x, bezierInfo.m_controlPt3.y, bezierInfo.m_controlPt3.z);
    gluSphere(gluNewQuadric(), radius, 20, 20);
    glPopMatrix();

    glPushMatrix();
    glTranslatef(bezierInfo.m_controlPt4.x, bezierInfo.m_controlPt4.y, bezierInfo.m_controlPt4.z);
    gluSphere(gluNewQuadric(), radius, 20, 20);
    glPopMatrix();
}

void RunningCurveViewer::RenderGrid()
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

void RunningCurveViewer::ResetView()
{
    makeCurrent();

    m_rotateX = 0.0f;
    m_rotateY = 0.0f;
    m_zoom = 50.0f;
}

void RunningCurveViewer::mousePressEvent(QMouseEvent* pEvent)
{
    makeCurrent();

    m_cachedX = pEvent->x();
    m_cachedY = pEvent->y();
}

void RunningCurveViewer::mouseReleaseEvent(QMouseEvent* pEvent)
{
}

void RunningCurveViewer::mouseMoveEvent(QMouseEvent* pEvent)
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

void RunningCurveViewer::wheelEvent(QWheelEvent* pEvent)
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

void RunningCurveViewer::keyPressEvent(QKeyEvent* pEvent)
{
    makeCurrent();

    const float lookAtMoveSpeed = 0.1f;

    switch (pEvent->key())
    {
    case Qt::Key_T:
    {
        m_scaledCurvature = !m_scaledCurvature;
    } break;
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

    case Qt::Key_P:
    {
        m_drawPoints = !m_drawPoints;
    } break;

    default:
    {
        return QWidget::keyPressEvent(pEvent);
    } break;
    }
}

void RunningCurveViewer::keyReleaseEvent(QKeyEvent* pEvent)
{
}

void RunningCurveViewer::PrepareCurveReset()
{
    m_upInitialCurve = nullptr;
    m_upBezierInfo = nullptr;
    m_gtPositions.clear();
    m_gtFrames.clear();
}

void RunningCurveViewer::ClampRotation()
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