#include "OpenGLWindow.h"

#include <QOpenGLPaintDevice>
#include <QPainter>
#include <QKeyEvent>

OpenGLWindow::OpenGLWindow(QWindow* pParent)
    : QWindow(pParent)
    , m_pContext(nullptr)
    , m_pDevice(nullptr)
{
    setSurfaceType(QWindow::OpenGLSurface);
}

OpenGLWindow::~OpenGLWindow()
{
}

void OpenGLWindow::render(QPainter* pPainter)
{
}

void OpenGLWindow::render()
{
    if (!m_pDevice)
    {
        m_pDevice = new QOpenGLPaintDevice();
    }

    m_pDevice->setSize(size());

    QPainter painter(m_pDevice);
    render(&painter);
}

void OpenGLWindow::initialize()
{
}

void OpenGLWindow::renderLater()
{
    requestUpdate();
}

void OpenGLWindow::renderNow()
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


bool OpenGLWindow::event(QEvent* pEvent)
{
    switch (pEvent->type())
    {
    case QEvent::UpdateRequest:
    {
        renderNow();
        return true;
    } break;

    default:
        return QWindow::event(pEvent);
    }
}

void OpenGLWindow::exposeEvent(QExposeEvent* pEvent)
{
    if (isExposed())
    {
        renderNow();
    }
}