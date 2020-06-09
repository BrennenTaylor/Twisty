#include "OpenGLWidget.h"

#include <QOpenGLPaintDevice>
#include <QPainter>
#include <QKeyEvent>

OpenGLWidget::OpenGLWidget(QWidget* pParent)
    : QWindow(pParent)
    , m_pContext(nullptr)
    , m_pDevice(nullptr)
{
    setSurfaceType(QWindow::OpenGLSurface);
}

OpenGLWidget::~OpenGLWidget()
{
}

void OpenGLWidget::render(QPainter* pPainter)
{
}

void OpenGLWidget::render()
{
    if (!m_pDevice)
    {
        m_pDevice = new QOpenGLPaintDevice();
    }

    m_pDevice->setSize(size());

    QPainter painter(m_pDevice);
    render(&painter);
}

void OpenGLWidget::initialize()
{
}

void OpenGLWidget::renderLater()
{
    requestUpdate();
}

void OpenGLWidget::renderNow()
{
    if (!isExposed())
    {
        return;
    }

    bool needsInitialize = false;

    if (!m_pContext) {
        m_pContext = new QOpenGLContext(this);
        m_pContext->setFormat(requestedFormat());
        m_pContext->create();

        needsInitialize = true;
    }

    m_pContext->makeCurrent(this);

    if (needsInitialize) {
        initializeOpenGLFunctions();
        initialize();
    }

    render();

    m_pContext->swapBuffers(this);
}


bool OpenGLWidget::event(QEvent* pEvent)
{
    switch (pEvent->type())
    {
    case QEvent::UpdateRequest:
    {
        renderNow();
        return true;
    } break;

    default:
        return QWidget::event(pEvent);
    }
}