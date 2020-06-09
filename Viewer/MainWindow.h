#pragma once

#include <QMainWindow>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* pParent = nullptr);
    ~MainWindow();

    void keyPressEvent(QKeyEvent* pEvent);

private:
};