#include "PathBatchViewerWidget.h"
#include "ExperimentRunner.h"

#include "CurveViewer.h"

#include <QPushButton>
#include <QButtonGroup>
#include <QCheckBox>
#include <QLabel>
#include <QLayout>
#include <QLineEdit>
#include <QSplitter>

#include <sstream>

PathBatchViewerWidget::PathBatchViewerWidget(QWidget *pParent)
    : QWidget(pParent)
    , m_pMainLayout(nullptr)
    , m_pCurveViewer(nullptr)
    , m_pPathControlContainer(nullptr)
    , m_pPathControlLayout(nullptr)
    , m_pButtonGroup(nullptr)
    , m_loadedCurvePoints()
    , m_numLoadedCurves(0)
    , m_drawIdx(0)
{
    setSizePolicy(QSizePolicy::Policy::Preferred, QSizePolicy::Policy::Preferred);

    m_pMainLayout = new QHBoxLayout();
    setLayout(m_pMainLayout);

    QSplitter *pSplitter = new QSplitter(this);
    m_pMainLayout->addWidget(pSplitter);

    m_pCurveViewer = new CurveViewer(this);
    m_pPathControlContainer = new QWidget(this);
    m_pPathControlContainer->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    m_pPathControlLayout = new QVBoxLayout();
    m_pPathControlContainer->setLayout(m_pPathControlLayout);

    // Add widgets to splitter
    pSplitter->addWidget(m_pCurveViewer);
    pSplitter->addWidget(m_pPathControlContainer);

    // The path control section
    m_pStartPathIdxEdit = new QLineEdit(this);
    m_pEndPathIdxEdit = new QLineEdit(this);

    m_pPathControlLayout->addWidget(m_pStartPathIdxEdit);
    m_pPathControlLayout->addWidget(m_pEndPathIdxEdit);

    m_pButtonGroup = new QButtonGroup();
    m_pSetPathDataButton = new QPushButton("Set", this);
    m_pResetPathDataButton = new QPushButton("Reset", this);

    connect(m_pSetPathDataButton, &QPushButton::pressed, this,
          &PathBatchViewerWidget::SetButtonCallback);
    connect(m_pResetPathDataButton, &QPushButton::pressed, this,
          &PathBatchViewerWidget::ResetButtonCallback);

    m_pButtonGroup->addButton(m_pSetPathDataButton);
    m_pButtonGroup->addButton(m_pResetPathDataButton);

    m_pPathControlLayout->addWidget(m_pSetPathDataButton);
    m_pPathControlLayout->addWidget(m_pResetPathDataButton);

    // The animation control section
    m_pAnimationContainer = new QWidget(m_pPathControlContainer);
    m_pPathControlLayout->addWidget(m_pAnimationContainer);

    m_pAnimationLayout = new QHBoxLayout();
    m_pAnimationContainer->setLayout(m_pAnimationLayout);

    m_pAnimatePathsCB = new QCheckBox("Animate Viewed Paths", m_pAnimationContainer);
    connect(m_pAnimatePathsCB, &QCheckBox::toggled, this,
          &PathBatchViewerWidget::AnimatePathsCallback);
    m_pAnimationLayout->addWidget(m_pAnimatePathsCB);

    m_pCurrentAnimationIdxLabel = new QLineEdit(
          QString::number(m_pCurveViewer->GetCurrentAnimatedIdx()), m_pAnimationContainer);
    m_pCurrentAnimationIdxLabel->setReadOnly(true);
    m_pCurrentAnimationIdxLabel->hide();
    m_pAnimationLayout->addWidget(m_pCurrentAnimationIdxLabel);

    m_pCurrentAnimationCurveWeightLable = new QLineEdit("", m_pAnimationContainer);
    m_pCurrentAnimationCurveWeightLable->setReadOnly(true);
    m_pCurrentAnimationCurveWeightLable->hide();
    m_pAnimationLayout->addWidget(m_pCurrentAnimationCurveWeightLable);

    connect(m_pCurveViewer, &CurveViewer::AnimatedCurveIdxChanged, this, [this](uint32_t idx) {
        m_pCurrentAnimationIdxLabel->setReadOnly(false);
        m_pCurrentAnimationIdxLabel->setText(QString::number(idx));
        m_pCurrentAnimationIdxLabel->setReadOnly(true);
    });

    connect(m_pCurveViewer, &CurveViewer::AnimatedCurveIdxChanged, this, [this](uint32_t idx) {
        const int idxOffset = idx - m_pStartPathIdxEdit->text().toInt();

        if (m_loadedCurveWeights.size() <= idxOffset) {
            m_pCurrentAnimationCurveWeightLable->setReadOnly(false);
            m_pCurrentAnimationCurveWeightLable->setText("");
            m_pCurrentAnimationCurveWeightLable->setReadOnly(true);
            return;
        }


        std::ostringstream weightSS;
        weightSS << m_loadedCurveWeights[idxOffset];

        m_pCurrentAnimationCurveWeightLable->setReadOnly(false);
        m_pCurrentAnimationCurveWeightLable->setText(QString::fromStdString(weightSS.str()));
        m_pCurrentAnimationCurveWeightLable->setReadOnly(true);
    });

    ResetButtonCallback();
}

PathBatchViewerWidget::~PathBatchViewerWidget() { }

CurveViewer &PathBatchViewerWidget::GetCurveViewerWidget() { return (*m_pCurveViewer); }

void PathBatchViewerWidget::RegisterRawPathDataFile(
      std::filesystem::path rawPathsFullpath, std::filesystem::path rawWeightsFullpath)
{
    m_pathToRawPaths = rawPathsFullpath;
    m_pathToRawWeights = rawWeightsFullpath;
}

void PathBatchViewerWidget::SetButtonCallback()
{
    const uint32_t startIdx = m_pStartPathIdxEdit->text().toInt();
    const uint32_t endIdx = m_pEndPathIdxEdit->text().toInt() + 1;
    const uint32_t numPointsPerCurve = m_pCurveViewer->GetInitialCurve().m_numSegments + 1;

    // Here, we want to make present the data from the file
    MakeDataPresent(startIdx, endIdx, numPointsPerCurve);

    m_pCurveViewer->SetPathDrawData(m_loadedCurvePoints.data(), m_curveInterpolationValues.data(),
          endIdx - startIdx, numPointsPerCurve);
}

void PathBatchViewerWidget::ResetButtonCallback()
{
    m_pCurveViewer->ResetPathDrawData();
    m_pStartPathIdxEdit->setText("0");
    m_pEndPathIdxEdit->setText("0");
}

void PathBatchViewerWidget::AnimatePathsCallback(bool checked)
{
    if (checked) {
        m_pCurveViewer->EnablePathAnimation();
        m_pCurrentAnimationIdxLabel->show();
        m_pCurrentAnimationCurveWeightLable->show();
    } else {
        m_pCurveViewer->DisablePathAnimation();
        m_pCurrentAnimationIdxLabel->hide();
        m_pCurrentAnimationCurveWeightLable->hide();
    }
}

void PathBatchViewerWidget::MakeDataPresent(
      uint32_t startIdx, uint32_t endIdx, uint32_t numPointsPerCurve)
{
    {
        std::ifstream rawDataFile(m_pathToRawPaths, std::ios::binary | std::ios::in);
        if (!rawDataFile.is_open()) {
            std::cout << "Couldnt open raw file for some reason: " << m_pathToRawPaths << std::endl;
        }

        uint64_t numPathsToReadIn = endIdx - startIdx;
        uint64_t numFloatsNeeded = sizeof(Farlor::Vector3) * numPointsPerCurve * numPathsToReadIn;
        const uint64_t bytesPerCurve = sizeof(Farlor::Vector3) * (numPointsPerCurve);
        const uint64_t numBytesToRead = bytesPerCurve * numPathsToReadIn;

        // Always resize to the required larger size
        if (m_loadedCurvePoints.size() < numFloatsNeeded) {
            m_loadedCurvePoints.resize(numFloatsNeeded);
        }

        uint64_t startSeekIdx = (uint64_t)(bytesPerCurve) * (uint64_t)(startIdx);
        rawDataFile.seekg(startSeekIdx, std::ios::beg);
        rawDataFile.read((char *)m_loadedCurvePoints.data(), numBytesToRead);
    }
    // Path weights
    {
        const boost::multiprecision::cpp_dec_float_100 minPossibleValue = 2.75308e-16;
        const boost::multiprecision::cpp_dec_float_100 maxPossibleValue = 8.88178e-14;

        std::ifstream curveWeightsFile(m_pathToRawWeights, std::ios::binary | std::ios::in);
        if (!curveWeightsFile.is_open()) {
            std::cout << "Couldnt open raw weight file for some reason: " << m_pathToRawWeights
                      << std::endl;
        }

        uint64_t numPathsToReadIn = endIdx - startIdx;
        uint64_t numFloatsNeeded = numPathsToReadIn;
        const uint64_t bytesPerCurve = sizeof(double);
        const uint64_t numBytesToRead = bytesPerCurve * numPathsToReadIn;

        // Always resize to the required larger size
        if (m_loadedCurveWeights.size() < numFloatsNeeded) {
            m_loadedCurveWeights.resize(numFloatsNeeded);
            m_curveInterpolationValues.resize(numFloatsNeeded);
        }

        std::vector<double> loadedCompressedWeights(m_loadedCurveWeights.size());

        uint64_t startSeekIdx = (uint64_t)(bytesPerCurve) * (uint64_t)(startIdx);
        curveWeightsFile.seekg(startSeekIdx, std::ios::beg);
        curveWeightsFile.read((char *)loadedCompressedWeights.data(), numBytesToRead);

        for (int i = 0; i < loadedCompressedWeights.size(); i++) {
            m_loadedCurveWeights[i] = loadedCompressedWeights[i];
            m_loadedCurveWeights[i] = boost::multiprecision::pow(10.0, m_loadedCurveWeights[i]);

            m_curveInterpolationValues[i] = (float)((m_loadedCurveWeights[i] - minPossibleValue)
                  / (maxPossibleValue - minPossibleValue));
            m_loadedCurveWeights[i] = m_curveInterpolationValues[i];
        }
    }
}