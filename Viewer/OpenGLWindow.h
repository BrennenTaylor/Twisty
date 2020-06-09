#pragma once

#include <QWindow>
#include <QOpenGLFunctions>

class OpenGLWindow : public QWindow, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit OpenGLWindow(QWindow* pParent = nullptr);
    ~OpenGLWindow();

    virtual void render(QPainter* pPainter);
    virtual void render();

    virtual void initialize();

public slots:
    void renderLater();
    void renderNow();

protected:
    bool event(QEvent* pEvent) override;
    void exposeEvent(QExposeEvent* pEvent) override;

protected:
    QOpenGLContext* m_pContext;
    QOpenGLPaintDevice* m_pDevice;
};