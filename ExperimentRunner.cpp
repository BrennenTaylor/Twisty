#include "ExperimentRunner.h"

#include "Curve.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace twisty {

std::string format_duration(std::chrono::milliseconds ms)
{
    using namespace std::chrono;
    auto secs = duration_cast<seconds>(ms);
    ms -= duration_cast<milliseconds>(secs);
    auto mins = duration_cast<minutes>(secs);
    secs -= duration_cast<seconds>(mins);
    auto hour = duration_cast<hours>(mins);
    mins -= duration_cast<minutes>(hour);

    std::stringstream ss;
    ss << hour.count() << " Hours : " << mins.count() << " Minutes : " << secs.count()
       << " Seconds : " << ms.count() << " Milliseconds";
    return ss.str();
}

std::string GetCurrentTimeForFileName()
{
    auto time = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%F_%T");  // ISO 8601 without timezone information.
    auto s = ss.str();
    std::replace(s.begin(), s.end(), ':', '-');
    return s;
}

void ProcessRandomSeed(uint32_t &seed)
{
    if (seed == 0) {
        seed = static_cast<uint32_t>(time(0));
    }
    return;
}

ExperimentRunner::ExperimentRunner(
      ExperimentParameters &experimentParams, Bootstrapper &bootstrapper)
    : m_experimentParams(experimentParams)
    , m_bootstrapper(bootstrapper)
    , m_upInitialCurve(nullptr)
    , m_exportPathBatchesMutex()
    , m_pathBatchJsonIndex()
{
    m_experimentDirPath = std::filesystem::path(m_experimentParams.experimentDirPath);
    if (!experimentParams.perExperimentDirSubfolder.empty()) {
        m_experimentDirPath /= experimentParams.perExperimentDirSubfolder;
    }
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
        printf("Curve bootstrapping failed.\n");
        throw std::runtime_error("Failed to generate bootstrap curve");
    }

    if (m_experimentParams.outputBigFloatWeights) {
        StartWeightConvergenceWrite();
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

    std::cout << format_duration(runExperimentTimeMs) << std::endl;

    std::cout << "\tRunner Specific Setup (ms): "
              << format_duration(std::chrono::milliseconds(runnerSpecificResult.setupMs))
              << std::endl;
    std::cout << "\tRunner Specific Perturb (ms): "
              << format_duration(std::chrono::milliseconds(runnerSpecificResult.runExperimentMs))
              << std::endl;
    std::cout << "\tRunner Specific Weighting (ms): "
              << format_duration(std::chrono::milliseconds(runnerSpecificResult.weightingMs))
              << std::endl;

    if (m_experimentParams.outputBigFloatWeights) {
        EndWeightConvergenceWrite();
    }

    if (m_experimentParams.outputPathBatches) {
        EndPathBatchOutput();

        m_curvesBinaryFile.close();
        m_curvesMetadataFile.close();
    }
    runnerSpecificResult.experimentResults->totalExperimentMs = runExperimentTimeMs.count();
    runnerSpecificResult.experimentResults->setupExperimentMs = runnerSpecificResult.setupMs;
    runnerSpecificResult.experimentResults->perturbExperimentMs
          = runnerSpecificResult.runExperimentMs;
    runnerSpecificResult.experimentResults->weightingExperimentMs
          = runnerSpecificResult.weightingMs;
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


void ExperimentRunner::StartWeightConvergenceWrite()
{
    std::filesystem::path convergenceWeightsPath = m_experimentDirPath;
    convergenceWeightsPath /= "ConvergenceWeights/";
    if (!std::filesystem::exists(convergenceWeightsPath)) {
        std::filesystem::create_directories(convergenceWeightsPath);
    }
    m_weightConvergenceFile.open(convergenceWeightsPath.string() + "ConvergenceWeights.txt");
    if (!m_weightConvergenceFile.is_open()) {
        throw std::runtime_error("Failed to open weight convergence file");
    }
}

void ExperimentRunner::UpdateConvergenceWeight(
      const uint64_t numNewPaths, const boost::multiprecision::cpp_dec_float_100 weightContribution)
{
    m_numWeightConvergencePaths += numNewPaths;
    m_weightConvergenceCombinedWeight += weightContribution;
    m_weightConvergenceFile << m_numWeightConvergencePaths << ", "
                            << m_weightConvergenceCombinedWeight / m_numWeightConvergencePaths
                            << std::endl;
}

void ExperimentRunner::EndWeightConvergenceWrite() { m_weightConvergenceFile.close(); }
}