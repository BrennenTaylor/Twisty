#pragma once

#include "Bootstrapper.h"
#include "Curve.h"
#include "ExperimentRunner.h"

#include <QWidget>

#include <filesystem>
#include <map>
#include <unordered_map>

class CurveViewer;

class QPushButton;
class QButtonGroup;
class QCheckBox;
class QHBoxLayout;
class QVBoxLayout;
class QLineEdit;
class QLabel;

class PathBatchViewerWidget : public QWidget {
    Q_OBJECT

   public:
    explicit PathBatchViewerWidget(QWidget *pParent = nullptr);
    ~PathBatchViewerWidget();

    CurveViewer &GetCurveViewerWidget();

    void RegisterRawPathDataFile(std::filesystem::path rawPathsFullpath);

   private:
    void SetButtonCallback();
    void ResetButtonCallback();
    void AnimatePathsCallback(bool checked);

    void MakeDataPresent(uint32_t startIdx, uint32_t endIdx, uint32_t numPointsPerCurve);

   private:
    QHBoxLayout *m_pMainLayout = nullptr;
    CurveViewer *m_pCurveViewer = nullptr;

    QWidget *m_pPathControlContainer = nullptr;
    QVBoxLayout *m_pPathControlLayout = nullptr;

    QLineEdit *m_pStartPathIdxEdit = nullptr;
    QLineEdit *m_pEndPathIdxEdit = nullptr;

    QButtonGroup *m_pButtonGroup = nullptr;
    QPushButton *m_pSetPathDataButton = nullptr;
    QPushButton *m_pResetPathDataButton = nullptr;

    QWidget *m_pAnimationContainer = nullptr;
    QHBoxLayout *m_pAnimationLayout = nullptr;
    QCheckBox *m_pAnimatePathsCB = nullptr;
    QLineEdit *m_pCurrentAnimationIdxLabel = nullptr;
    QLineEdit *m_pCurrentAnimationCurveWeightLable = nullptr;

    std::filesystem::path m_pathToRawPaths = "";

    std::vector<float> m_loadedCurvePoints;
    uint32_t m_numLoadedCurves = 0;
    uint32_t m_drawIdx = 0;
};