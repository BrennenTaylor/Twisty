#include "ExperimentRunner.h"

#include "Curve.h"
#include "nlohmann/json.hpp"

#include <iomanip>
#include <sstream>

namespace twisty {
ExperimentRunner::ExperimentRunner(
      ExperimentParameters &experimentParams, Bootstrapper &bootstrapper)
    : m_experimentParams(experimentParams)
    , m_bootstrapper(bootstrapper)
    , m_upInitialCurve(nullptr)
    , m_exportPathBatchesMutex()
    , m_pathBatchJsonIndex()
{
    m_experimentDirPath = m_experimentParams.experimentDirPath;
    if (!std::filesystem::exists(m_experimentDirPath)) {
        std::filesystem::create_directories(m_experimentDirPath);
    }
}

ExperimentRunner::~ExperimentRunner() { }

std::optional<ExperimentRunner::ExperimentResults> ExperimentRunner::RunExperiment()
{
    auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();

    /* --------------------- */

    std::cout << "Random Seeds: " << std::endl;
    std::cout << "\tBootstrap seed: " << m_experimentParams.bootstrapSeed << std::endl;
    std::cout << "\tPerturb seed: " << m_experimentParams.curvePurturbSeed << std::endl;

    m_upInitialCurve = m_bootstrapper.CreateCurveGeometricSafe(
          m_experimentParams.numSegmentsPerCurve, m_experimentParams.arclength);
    if (!m_upInitialCurve) {
        printf("Both bootstrap versions failed, now we have to error out.\n");
        return {};
    }

    if (m_experimentParams.outputPathBatches) {
        BeginPathBatchOutput();

        std::stringstream pathBinaryFilenameSS;
        pathBinaryFilenameSS << m_experimentParams.pathBatchPrepend;
        pathBinaryFilenameSS << "Paths_Binary"
                             << ".pbd";

        std::filesystem::path binaryFilePath = m_pathBatchOutputPath;
        binaryFilePath.append(pathBinaryFilenameSS.str());
        m_curvesBinaryFile.open(binaryFilePath, std::ios::binary);

        std::stringstream pathMetadataFilenameSS;
        pathMetadataFilenameSS << m_experimentParams.pathBatchPrepend;
        pathMetadataFilenameSS << "Paths_Metadata"
                               << ".pmd";

        std::filesystem::path metadataFilePath = m_pathBatchOutputPath;
        metadataFilePath.append(pathMetadataFilenameSS.str());
        m_curvesMetadataFile.open(metadataFilePath);
    }

    RunnerSpecificResults runnerSpecificResult = RunnerSpecificRunExperiment();

    auto runExperimentTimeEnd = std::chrono::high_resolution_clock::now();

    std::cout << "Experiment Time Reporting: " << std::endl;
    auto runExperimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(
          runExperimentTimeEnd - runExperimentTimeStart);
    std::cout << "\tTotal Experiment Time: " << runExperimentTimeMs.count() << "ms" << std::endl;

    std::cout << "\tRunner Specific Setup (ms): " << runnerSpecificResult.setupMsCount << std::endl;
    std::cout << "\tRunner Specific Perturb (ms): " << runnerSpecificResult.runExperimentMsCount
              << std::endl;
    std::cout << "\tRunner Specific Weighting (ms): " << runnerSpecificResult.weightingMsCount
              << std::endl;

    if (m_experimentParams.outputPathBatches) {
        EndPathBatchOutput();

        m_curvesBinaryFile.close();
        m_curvesMetadataFile.close();
    }
    runnerSpecificResult.experimentResults->totalExperimentMS = runExperimentTimeMs.count();
    runnerSpecificResult.experimentResults->setupExperimentMS = runnerSpecificResult.setupMsCount;
    runnerSpecificResult.experimentResults->perturbExperimentMS
          = runnerSpecificResult.runExperimentMsCount;
    runnerSpecificResult.experimentResults->weightingExperimentMS
          = runnerSpecificResult.weightingMsCount;
    return runnerSpecificResult.experimentResults;
}

bool ExperimentRunner::BeginPathBatchOutput()
{
    std::filesystem::path generatedCurvesDirPath = m_experimentDirPath;
    generatedCurvesDirPath /= m_experimentParams.perExperimentDirSubfolder;
    generatedCurvesDirPath /= "GeneratedCurves";
    if (!std::filesystem::exists(generatedCurvesDirPath)) {
        std::filesystem::create_directories(generatedCurvesDirPath);
    }

    m_pathBatchOutputPath = generatedCurvesDirPath.string();

    m_pathBatchJsonIndex.clear();

    m_pathBatchJsonIndex["experiment_name"] = m_experimentParams.experimentName;

    std::stringstream seedCurveSS;
    seedCurveSS << m_experimentParams.experimentName;
    seedCurveSS << "_Seed_Curve.tcf";

    std::filesystem::path seedCurvePath = generatedCurvesDirPath;
    seedCurvePath.append(seedCurveSS.str());
    m_pathBatchJsonIndex["seed_curve"] = seedCurveSS.str();

    m_pathBatchJsonIndex["path_batch_links"] = nlohmann::json::array();

    // TODO: Collapse this into one function?
    std::ofstream seedCurveOutfile(seedCurvePath.string(), std::ios::binary);
    twisty::Curve::WriteCurveToStream(seedCurveOutfile, *m_upInitialCurve);

    return true;
}

void ExperimentRunner::OutputPathBatch(PathBatch &pathBatch)
{
    std::stringstream pathBatchNameSS;
    pathBatchNameSS << m_experimentParams.experimentName;
    pathBatchNameSS << "_PathBatch_";
    pathBatchNameSS << pathBatch.index;
    pathBatchNameSS << ".tpb";

    std::filesystem::path outputDirectoryPath = m_pathBatchOutputPath;

    std::filesystem::path pathBatchFilename = outputDirectoryPath;
    pathBatchFilename.append(pathBatchNameSS.str());

    std::ofstream pathBatchOutfile(pathBatchFilename.string(), std::ios::binary);
    if (!pathBatchOutfile.is_open()) {
        std::cout << "Failed to open " << pathBatchFilename.string() << std::endl;
        return;
    }

    m_pathBatchJsonIndex["path_batch_links"].push_back(pathBatchNameSS.str());

    pathBatchOutfile.write((char *)&pathBatch.numberOfPathsInBatch, sizeof(uint32_t));
    pathBatchOutfile.write((char *)&pathBatch.m_curvatures[0],
          sizeof(float) * m_experimentParams.numSegmentsPerCurve * pathBatch.numberOfPathsInBatch);
    pathBatchOutfile.write((char *)&pathBatch.m_positions[0],
          sizeof(Farlor::Vector3) * m_experimentParams.numSegmentsPerCurve
                * pathBatch.numberOfPathsInBatch);
    pathBatchOutfile.write((char *)&pathBatch.m_tangents[0],
          sizeof(Farlor::Vector3) * m_experimentParams.numSegmentsPerCurve
                * pathBatch.numberOfPathsInBatch);

    // Also, we will write it out to an individual batch file
    pathBatchOutfile.close();
    std::cout << "Completed writing path batch file" << std::endl;
}

void ExperimentRunner::EndPathBatchOutput()
{
    std::stringstream indexJsonSS;
    indexJsonSS << "index.json";

    std::filesystem::path outputDirectoryPath = m_pathBatchOutputPath;

    std::filesystem::path indexJsonPath = outputDirectoryPath;
    indexJsonPath.append(indexJsonSS.str());

    std::ofstream jsonOfstream(indexJsonPath.string());
    jsonOfstream << std::setw(4) << m_pathBatchJsonIndex << std::endl;
    jsonOfstream.close();
}
}