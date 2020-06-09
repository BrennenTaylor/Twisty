#include "PathPerturbViewerWidget.h"
#include "RunningCurveViewer.h"

#include <fmt/format.h>

#include <QButtonGroup>
#include <QCheckBox>
#include <QLayout>
#include <QSplitter>

PathPerturbViewerWidget::PathPerturbViewerWidget(QWidget* pParent)
    : QWidget(pParent)
    , m_pMainLayout(nullptr)
    , m_pRunningCurveViewer(nullptr)
    , m_pCheckboxContainer(nullptr)
    , m_pCheckboxLayout(nullptr)
    , m_pButtonGroup(nullptr)
{
    setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);

    m_pMainLayout = new QHBoxLayout();

    setLayout(m_pMainLayout);

    QSplitter* pSplitter = new QSplitter(this);
    m_pMainLayout->addWidget(pSplitter);

    m_pRunningCurveViewer = new RunningCurveViewer(*this, this);

    m_pCheckboxContainer = new QWidget(this);

    pSplitter->addWidget(m_pRunningCurveViewer);
    pSplitter->addWidget(m_pCheckboxContainer);

    m_pCheckboxLayout = new QVBoxLayout();
    m_pCheckboxContainer->setLayout(m_pCheckboxLayout);
}

PathPerturbViewerWidget::~PathPerturbViewerWidget()
{
}

RunningCurveViewer* PathPerturbViewerWidget::GetRunningCurveViewerWidget()
{
    return m_pRunningCurveViewer;
}