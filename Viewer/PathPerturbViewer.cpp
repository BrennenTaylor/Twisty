#include "MainWindow.h"

#include "PathPerturbViewerWidget.h"
#include "RunningCurveViewer.h"

#include "CurveUtils.h"
#include "ExperimentRunnerCpu.h"
#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"
#include "TwistyYamlUtils.h"

#include <QApplication>
#include <QSurfaceFormat>

#include <fmt/format.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstdint>
#include <fstream>
#include <filesystem>

std::filesystem::path GetExperimentDirectory(const std::string experimentDirectoryName)
{
    // Get currect directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "CWD: " << cwd << std::endl;

    std::string experimentDirectoryAppend(experimentDirectoryName);
    std::filesystem::path experimentDirectoryPath = cwd;
    experimentDirectoryPath.append(experimentDirectoryAppend);
    return experimentDirectoryPath;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fmt::print("Call as: {} NumSegments CurveCacheSize\n", argv[0]);
        return 1;
    }

    QApplication app(argc, argv);

    // Instantiate resources using app
    MainWindow mainWindow;
    mainWindow.show();

    PathPerturbViewerWidget PathPerturbViewerWidget;
    mainWindow.setCentralWidget(&PathPerturbViewerWidget);

    RunningCurveViewer* pRunningCurveViewer = PathPerturbViewerWidget.GetRunningCurveViewerWidget();

    const uint32_t numSegments = atoi(argv[1]);
    const uint32_t curveCacheSize = atoi(argv[2]);
    
    const twisty::Range defaultBounds = { -1.0f, 1.0f };
    const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    twisty::RayGeometry rayEmitter(emitterStart, emitterDir);

    const Farlor::Vector3 recieverPos{ 10.0f, 0.0f, 0.0f };
    const Farlor::Vector3 recieverDir{ 1.0, 0.0f, 0.0f };

    twisty::RayGeometry rayReciever(recieverPos, recieverDir);


    const twisty::Range arclengthRange = { 10.0f, 30.0f };

    const uint32_t randomSeed = 0;
    twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, arclengthRange, randomSeed);

    uint32_t numPathsToSkip = 0;
    uint32_t numPathsToGenerate = std::stoi(argv[1]);

    twisty::ExperimentRunner::ExperimentParameters experimentParams;
    experimentParams.numPathsInExperiment = numPathsToGenerate;
    experimentParams.exportGeneratedCurves = true;
    experimentParams.experimentName = "PurturbViewer";
    experimentParams.numSegmentsPerCurve = numSegments;
    experimentParams.maximumBootstrapCurveError = 0.5f;
    experimentParams.curvePurturbSeed = 0;

    std::unique_ptr<twisty::Curve> upInitialCurve = bootstrapper.CreateCurve(numSegments);

    pRunningCurveViewer->SetInitialCurve(*upInitialCurve);
    pRunningCurveViewer->SetCurveCacheSize(curveCacheSize);
    pRunningCurveViewer->InitializeExperiment();

    return app.exec();
}