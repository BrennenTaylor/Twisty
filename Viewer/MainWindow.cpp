#include "MainWindow.h"

#include <QDesktopWidget>
#include <QMouseEvent>

MainWindow::MainWindow(QWidget* pParent)
    : QMainWindow(pParent)
{
    resize(QDesktopWidget().availableGeometry(this).size() * 0.7);
}

MainWindow::~MainWindow()
{
}

void MainWindow::keyPressEvent(QKeyEvent* pEvent)
{
    switch (pEvent->key())
    {
        case Qt::Key_Escape:
        {
            this->close();
        } break;

        default:
        {
            return QMainWindow::keyPressEvent(pEvent);
        } break;
    }
}