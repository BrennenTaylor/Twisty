#pragma once

#include <QOpenGLWidget>
#include <QOpenGLFunctions>

class OpenGLWidget : public QOpenGLWidget, public QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit OpenGLWidget(QWidget* pParent = nullptr);
    ~OpenGLWidget();

    virtual void render(QPainter* pPainter);
    virtual void render();

    virtual void initialize();

public slots:
    void renderLater();
    void renderNow();

protected:
    bool event(QEvent* pEvent) override;

protected:
};