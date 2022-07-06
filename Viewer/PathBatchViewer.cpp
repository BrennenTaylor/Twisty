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
    if (argc < 4) {
        printf("Call as: %s PathDirectory RawPathFilename RawWeightFilename", argv[0]);
        return 1;
    }

    QApplication app(argc, argv);

    // Instantiate resources using app
    MainWindow mainWindow;
    mainWindow.show();

    PathBatchViewerWidget pathBatchViewerWidget;
    mainWindow.setCentralWidget(&pathBatchViewerWidget);

    CurveViewer &curveViewer = pathBatchViewerWidget.GetCurveViewerWidget();


    //const std::string FixedBinaryFilename("Paths_FixedOrder.pbd");
    std::string pathsDirectory(argv[1]);
    std::string rawPathFilename(argv[2]);
    std::string rawWeightFilename(argv[3]);

    std::filesystem::path pathsDirectoryPath = pathsDirectory;
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    std::filesystem::path fixedBinaryFullPath(pathsDirectoryPath);
    fixedBinaryFullPath.append(rawPathFilename);
    std::cout << "rawPathsPath: " << fixedBinaryFullPath << std::endl;

    std::filesystem::path fixedWeightsBinaryFullPath(pathsDirectoryPath);
    fixedWeightsBinaryFullPath.append(rawWeightFilename);
    std::cout << "fixedWeightsBinaryFullPath: " << fixedWeightsBinaryFullPath << std::endl;

    pathBatchViewerWidget.RegisterRawPathDataFile(fixedBinaryFullPath, fixedWeightsBinaryFullPath);

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

    // We need to create an initial curve object
    std::unique_ptr<twisty::Curve> upInitialCurve = twisty::Curve::LoadCurveFromStream(seedCurveFS);
    std::cout << "Base Curve Info:" << std::endl;
    std::cout << "\tNum Segements: " << upInitialCurve->m_numSegments << std::endl;
    std::cout << "\tDs: " << upInitialCurve->m_ds << std::endl;
    std::cout << "\tArclength: " << upInitialCurve->m_boundaryConditions.arclength << std::endl;
    std::cout << "\tStart Pos: " << upInitialCurve->m_boundaryConditions.m_startPos << std::endl;
    std::cout << "\tStart Dir: " << upInitialCurve->m_boundaryConditions.m_startDir << std::endl;
    std::cout << "\tEnd Pos: " << upInitialCurve->m_boundaryConditions.m_endPos << std::endl;
    std::cout << "\tEnd Dir: " << upInitialCurve->m_boundaryConditions.m_endDir << std::endl;

    curveViewer.SetInitialCurve(*upInitialCurve);

    return app.exec();
}