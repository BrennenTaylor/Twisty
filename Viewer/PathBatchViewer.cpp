#include "MainWindow.h"

#include "PathBatchViewerWidget.h"
#include "CurveViewer.h"

#include "CurveUtils.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <nlohmann/json.hpp>

#include <QApplication>
#include <QSurfaceFormat>

#include <cstdint>
#include <fstream>
#include <filesystem>

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Call as: %s PathDirectory RawPathFilename", argv[0]);
        return 1;
    }

    QApplication app(argc, argv);

    // Instantiate resources using app
    MainWindow mainWindow;
    mainWindow.show();

    PathBatchViewerWidget pathBatchViewerWidget;
    mainWindow.setCentralWidget(&pathBatchViewerWidget);

    CurveViewer *pCurveViewer = pathBatchViewerWidget.GetCurveViewerWidget();


    //const std::string FixedBinaryFilename("Paths_FixedOrder.pbd");
    std::string pathsDirectory(argv[1]);
    std::string rawPathFilename(argv[2]);

    std::filesystem::path pathsDirectoryPath = pathsDirectory;
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    std::filesystem::path fixedBinaryFullPath(pathsDirectoryPath);
    fixedBinaryFullPath.append(rawPathFilename);
    std::cout << "rawPathsPath: " << fixedBinaryFullPath << std::endl;

    pathBatchViewerWidget.RegisterRawPathDataFile(fixedBinaryFullPath);

    std::filesystem::path indexPath = pathsDirectoryPath;
    indexPath.append("index.json");

    if (!std::filesystem::exists(indexPath)) {
        std::cout << indexPath << " file does not exist" << std::endl;
        return 1;
    }

    std::fstream indexFS(indexPath);
    if (!indexFS.is_open()) {
        std::cout << "Failed to open: " << indexPath << std::endl;
        return 1;
    }

    nlohmann::json readJson;
    indexFS >> readJson;

    const std::string experimentName = readJson["experiment_name"];
    const std::string seedCurveFilename = readJson["seed_curve"];
    std::filesystem::path seedCurvePath = pathsDirectoryPath;
    seedCurvePath.append(seedCurveFilename);

    std::cout << "Seed curve path: " << seedCurvePath << std::endl;

    std::ifstream seedCurveFS(seedCurvePath, std::ios::binary);
    if (!seedCurveFS.is_open()) {
        printf("Failed to open %s\n", seedCurvePath.string().c_str());
        return 1;
    }

    uint32_t numSegments = 0;
    seedCurveFS.read((char *)&numSegments, sizeof(uint32_t));

    // We need to create an initial curve object
    std::unique_ptr<twisty::Curve> upInitialCurve = std::make_unique<twisty::Curve>(numSegments);
    seedCurveFS.read((char *)&upInitialCurve->m_arclength, sizeof(float));
    seedCurveFS.read((char *)&upInitialCurve->m_basePos, sizeof(Farlor::Vector3));
    seedCurveFS.read((char *)&upInitialCurve->m_baseTangent, sizeof(Farlor::Vector3));
    seedCurveFS.read((char *)&upInitialCurve->m_targetPos, sizeof(Farlor::Vector3));
    seedCurveFS.read((char *)&upInitialCurve->m_targetTangent, sizeof(Farlor::Vector3));

    std::cout << "Base Curve Info:" << std::endl;
    std::cout << "\tNum Segements: " << upInitialCurve->m_numSegments << std::endl;
    std::cout << "\tArclength: " << upInitialCurve->m_arclength << std::endl;
    std::cout << "\tBase Pos: " << upInitialCurve->m_basePos << std::endl;
    std::cout << "\tBase Tangent: " << upInitialCurve->m_baseTangent << std::endl;
    std::cout << "\tTarget Pos: " << upInitialCurve->m_targetPos << std::endl;
    std::cout << "\tTarget Tangent: " << upInitialCurve->m_targetTangent << std::endl;


    seedCurveFS.read(
          (char *)&upInitialCurve->m_curvatures[0], sizeof(float) * upInitialCurve->m_numSegments);
    seedCurveFS.read((char *)&upInitialCurve->m_positions[0],
          sizeof(Farlor::Vector3) * upInitialCurve->m_numSegments);
    seedCurveFS.read((char *)&upInitialCurve->m_tangents[0],
          sizeof(Farlor::Vector3) * upInitialCurve->m_numSegments);

    pCurveViewer->SetInitialCurve(*upInitialCurve);

    return app.exec();
}