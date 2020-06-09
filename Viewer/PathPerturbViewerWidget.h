#pragma once

#include "Bootstrapper.h"
#include "Curve.h"
#include "ExperimentRunner.h"

#include <QWidget>

#include <filesystem>
#include <map>
#include <unordered_map>

class RunningCurveViewer;

class QButtonGroup;
class QHBoxLayout;
class QVBoxLayout;

class PathPerturbViewerWidget : public QWidget
{
    Q_OBJECT

public:

public:
    explicit PathPerturbViewerWidget(QWidget *pParent = nullptr);
    ~PathPerturbViewerWidget();

    RunningCurveViewer *GetRunningCurveViewerWidget();

private:
    QHBoxLayout *m_pMainLayout;
    RunningCurveViewer *m_pRunningCurveViewer;

    QWidget *m_pCheckboxContainer;
    QVBoxLayout *m_pCheckboxLayout;

    QButtonGroup *m_pButtonGroup;
};