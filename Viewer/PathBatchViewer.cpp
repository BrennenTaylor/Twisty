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

    CurveViewer &curveViewer = pathBatchViewerWidget.GetCurveViewerWidget();

    std::string pathsDirectory(argv[1]);
    std::string rawPathFilename(argv[2]);

    std::filesystem::path pathsDirectoryPath = pathsDirectory;
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    // All valid directories must have an Boundary Conditions File
    std::filesystem::path bcPath = pathsDirectoryPath;
    bcPath.append("BoundaryConditions.bcf");

    if (!std::filesystem::exists(bcPath)) {
        std::cout << bcPath << " file does not exist, provide one" << std::endl;
        return 1;
    }

    std::fstream bcFS(bcPath, std::ios::binary);
    if (!bcFS.is_open()) {
        std::cout << "Failed to open: " << bcPath << std::endl;
        return 1;
    }

    // TODO: Make this safer?
    twisty::PerturbUtils::BoundaryConditions bc;
    bcFS.read((char *)bc.m_startPos.m_data.data(), sizeof(Farlor::Vector3));
    bcFS.read((char *)bc.m_startDir.m_data.data(), sizeof(Farlor::Vector3));
    bcFS.read((char *)bc.m_endPos.m_data.data(), sizeof(Farlor::Vector3));
    bcFS.read((char *)bc.m_endDir.m_data.data(), sizeof(Farlor::Vector3));
    bcFS.read((char *)&bc.arclength, sizeof(float));

    uint32_t numSegments = 0;
    bcFS.read((char *)&numSegments, sizeof(uint32_t));

    curveViewer.SetExperimentConditions(bc, numSegments);

    // All valid directories must have an index.json
    std::filesystem::path indexPath = pathsDirectoryPath;
    indexPath.append("index.json");

    if (!std::filesystem::exists(indexPath)) {
        std::cout << indexPath << " file does not exist, provide one" << std::endl;
        return 1;
    }

    std::fstream indexFS(indexPath);
    if (!indexFS.is_open()) {
        std::cout << "Failed to open: " << indexPath << std::endl;
        return 1;
    }

    nlohmann::json readJson;
    indexFS >> readJson;

    std::filesystem::path fixedBinaryFullPath(pathsDirectoryPath);
    fixedBinaryFullPath.append(rawPathFilename);
    std::cout << "rawPathsPath: " << fixedBinaryFullPath << std::endl;

    pathBatchViewerWidget.RegisterRawPathDataFile(fixedBinaryFullPath);


    const std::string experimentName = readJson.value("experiment_name", "");

    // Detect whether or not we have a seed curve file
    const std::string seedCurveFilename = readJson.value("seed_curve", "");
    // Load up the initial curve
    if (!seedCurveFilename.empty()) {
        std::filesystem::path seedCurvePath = pathsDirectoryPath;
        seedCurvePath.append(seedCurveFilename);

        std::cout << "Seed curve path: " << seedCurvePath << std::endl;

        std::ifstream seedCurveFS(seedCurvePath, std::ios::binary);
        if (!seedCurveFS.is_open()) {
            printf("Failed to open %s\n", seedCurvePath.string().c_str());
            return 1;
        }

        // We need to create an initial curve object
        std::unique_ptr<twisty::Curve> upInitialCurve
              = twisty::Curve::LoadCurveFromStream(seedCurveFS);
        std::cout << "Base Curve Info:" << std::endl;
        std::cout << "\tNum Segements: " << upInitialCurve->m_numSegments << std::endl;
        std::cout << "\tDs: " << upInitialCurve->m_ds << std::endl;
        std::cout << "\tArclength: " << upInitialCurve->m_boundaryConditions.arclength << std::endl;
        std::cout << "\tStart Pos: " << upInitialCurve->m_boundaryConditions.m_startPos
                  << std::endl;
        std::cout << "\tStart Dir: " << upInitialCurve->m_boundaryConditions.m_startDir
                  << std::endl;
        std::cout << "\tEnd Pos: " << upInitialCurve->m_boundaryConditions.m_endPos << std::endl;
        std::cout << "\tEnd Dir: " << upInitialCurve->m_boundaryConditions.m_endDir << std::endl;

        // Takes ownership
        curveViewer.SetInitialCurve(std::move(upInitialCurve));
        upInitialCurve = nullptr;
    }

    return app.exec();
}