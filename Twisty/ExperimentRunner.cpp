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

ExperimentRunner::ExperimentParameters ExperimentRunner::ParseExperimentParamsFromConfig(
      const nlohmann::json &experimentConfig)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    try {
        // Values loaded from the config file
        experimentParams.numPathsInExperiment
              = experimentConfig.at("experiment").at("experimentParams").at("pathsToGenerate")
                      .get<uint64_t>();


        experimentParams.numPathsToSkip
              = experimentConfig.at("experiment").at("experimentParams").at("pathsToSkip").get<uint64_t>();
        experimentParams.experimentName
              = experimentConfig.at("experiment").at("experimentParams").at("name").get<std::string>();
        experimentParams.experimentDirPath
              = experimentConfig.at("experiment").at("experimentParams").at("experimentDir")
                      .get<std::string>();
        experimentParams.experimentDirPath += "/" + experimentParams.experimentName;
        experimentParams.experimentDirPath += "/" + twisty::GetCurrentTimeForFileName() + "/";

        experimentParams.maxPerturbThreads
              = experimentConfig.at("experiment").at("experimentParams").at("maxPerturbThreads").get<int>();
        experimentParams.maxWeightThreads
              = experimentConfig.at("experiment").at("experimentParams").at("maxWeightThreads").get<int>();

        experimentParams.outputBigFloatWeights
              = experimentConfig.at("experiment").at("experimentParams").at("outputBigFloatWeights")
                      .get<bool>();
        experimentParams.outputPathBatches
              = experimentConfig.at("experiment").at("experimentParams").at("outputPathBatches").get<bool>();
        experimentParams.useGpu
              = experimentConfig.at("experiment").at("experimentParams").at("useGpu").get<bool>();

        experimentParams.numSegmentsPerCurve
              = experimentConfig.at("experiment").at("experimentParams").at("numSegments").get<int>();

        // Seeds
        experimentParams.bootstrapSeed
              = experimentConfig.at("experiment").at("experimentParams").at("random").at("bootstrapSeed")
                      .get<int>();
        experimentParams.curvePurturbSeed
              = experimentConfig.at("experiment").at("experimentParams").at("random").at("perturbSeed")
                      .get<int>();

        // Weighting parameter stuff
        switch (int weightFunction
                = experimentConfig.at("experiment").at("experimentParams").at("weighting").at("weightFunction")
                        .get<int>();
                weightFunction) {
            // Radiative Transfer weight function
            case 0: {
                experimentParams.weightingParameters.weightingMethod
                      = twisty::WeightingMethod::RadiativeTransfer;
            } break;

            // Simplified Model
            case 1: {
                experimentParams.weightingParameters.weightingMethod
                      = twisty::WeightingMethod::SimplifiedModel;
            } break;

            // Default to the simplified model
            default: {
                experimentParams.weightingParameters.weightingMethod
                      = twisty::WeightingMethod::RadiativeTransfer;
            } break;
        }

        // Perturb method stuff
        switch (int perturbMethod
                = experimentConfig.at("experiment").at("experimentParams").at("perturbMethod").get<int>();
                perturbMethod) {
            // Simplified Model
            case 1: {
                experimentParams.perturbMethod
                      = twisty::ExperimentRunner::PerturbMethod::GeometricMinCurvature;
            } break;

            // Simplified Model
            case 2: {
                experimentParams.perturbMethod
                      = twisty::ExperimentRunner::PerturbMethod::GeometricCombined;
            } break;

            // Default to the simplified model
            case 0:
            default: {
                experimentParams.perturbMethod
                      = twisty::ExperimentRunner::PerturbMethod::GeometricRandom;
            } break;
        }

        // Weighting parameter stuff
        experimentParams.weightingParameters.mu
              = experimentConfig.at("experiment").at("experimentParams").at("weighting").at("mu").get<float>();
        experimentParams.weightingParameters.eps
              = experimentConfig.at("experiment").at("experimentParams").at("weighting").at("eps").get<float>();
        experimentParams.weightingParameters.numStepsInt
              = (int)experimentConfig.at("experiment").at("experimentParams").at("weighting").at("numStepsInt")
                      .get<int>();
        experimentParams.weightingParameters.numCurvatureSteps
              = (int)experimentConfig.at("experiment").at("experimentParams").at("weighting").at("numCurvatureSteps")
                                           .get<int>();
        experimentParams.weightingParameters.absorption
              = experimentConfig.at("experiment").at("experimentParams").at("weighting").at("absorption")
                      .get<float>();

        // In the case we have the simplified model, we need to disable absorption
        if (experimentParams.weightingParameters.weightingMethod
              == twisty::WeightingMethod::SimplifiedModel) {
            std::cout << "Disabling absorption for simplified model." << std::endl;
            experimentParams.weightingParameters.absorption = 0.0f;
        }

        experimentParams.weightingParameters.scatter
              = experimentConfig.at("experiment").at("experimentParams").at("weighting").at("scatter")
                      .get<float>();

        // TODO: Should these be configurable in the file?
        experimentParams.weightingParameters.minBound = 0.0f;
        experimentParams.weightingParameters.maxBound
              = 10.0f / experimentParams.weightingParameters.eps;
    } catch (std::exception &ex) {
        std::cout << "What: " << ex.what() << std::endl;
    }

    return experimentParams;
}


ExperimentRunner::ExperimentRunner(ExperimentParameters &experimentParams)
    : m_experimentParams(experimentParams)
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

std::optional<ExperimentRunner::ExperimentResults> ExperimentRunner::RunExperiment(
      twisty::PerturbUtils::BoundaryConditions boundaryConditions)
{
    auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();
    /* --------------------- */

    std::cout << "Random Seeds: " << std::endl;
    std::cout << "\tBootstrap seed: " << m_experimentParams.bootstrapSeed << std::endl;
    std::cout << "\tPerturb seed: " << m_experimentParams.curvePurturbSeed << std::endl;

    if (m_experimentParams.outputBigFloatWeights) {
        StartWeightConvergenceWrite();
    }

    if (m_experimentParams.outputPathBatches) {
        BeginPathBatchOutput(boundaryConditions, m_experimentParams.numSegmentsPerCurve);

        std::stringstream pathMetadataFilenameSS;
        pathMetadataFilenameSS << m_experimentParams.pathBatchPrepend;
        pathMetadataFilenameSS << "Paths_Metadata"
                               << ".pmd";

        std::filesystem::path metadataFilePath = m_pathBatchOutputPath;
        metadataFilePath.append(pathMetadataFilenameSS.str());
        m_curvesMetadataFile.open(metadataFilePath);

        std::stringstream pathBinaryFilenameSS;
        pathBinaryFilenameSS << m_experimentParams.pathBatchPrepend;
        pathBinaryFilenameSS << "Paths_Binary"
                             << ".pbd";

        {
            std::filesystem::path binaryFilePath = m_pathBatchOutputPath;
            binaryFilePath.append(pathBinaryFilenameSS.str());
            m_curvesBinaryFile.open(binaryFilePath, std::ios::binary);
        }

        std::stringstream log10PathWeightsBinaryFilenameSS;
        log10PathWeightsBinaryFilenameSS << m_experimentParams.pathBatchPrepend;
        log10PathWeightsBinaryFilenameSS << "Log10PathWeights_Binary"
                                         << ".bdt";
        {
            std::filesystem::path binaryFilePath = m_pathBatchOutputPath;
            binaryFilePath.append(log10PathWeightsBinaryFilenameSS.str());
            m_log10PathWeightsBinaryFile.open(binaryFilePath, std::ios::binary);
        }

        std::stringstream log10PathWeightsTextFilenameSS;
        log10PathWeightsTextFilenameSS << m_experimentParams.pathBatchPrepend;
        log10PathWeightsTextFilenameSS << "Log10PathWeights_Text"
                                       << ".pwt";
        {
            std::filesystem::path textFilePath = m_pathBatchOutputPath;
            textFilePath.append(log10PathWeightsTextFilenameSS.str());
            m_log10PathWeightsTextFile.open(textFilePath);
        }

        std::stringstream fiveSegmentBinaryFilenameSS;
        fiveSegmentBinaryFilenameSS << m_experimentParams.pathBatchPrepend;
        fiveSegmentBinaryFilenameSS << "FiveSegment_Binary"
                                    << ".bdt";
        {
            std::filesystem::path binaryFilePath = m_pathBatchOutputPath;
            binaryFilePath.append(fiveSegmentBinaryFilenameSS.str());
            m_fiveSegmentBinaryFile.open(binaryFilePath, std::ios::binary);
        }
    }

    RunnerSpecificResults runnerSpecificResult = RunnerSpecificRunExperiment(boundaryConditions);

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

bool ExperimentRunner::BeginPathBatchOutput(
      twisty::PerturbUtils::BoundaryConditions boundaryConditions, uint32_t numSegments)
{
    std::filesystem::path generatedCurvesDirPath = m_experimentDirPath;
    generatedCurvesDirPath /= "GeneratedCurves";
    if (!std::filesystem::exists(generatedCurvesDirPath)) {
        std::filesystem::create_directories(generatedCurvesDirPath);
    }

    m_pathBatchOutputPath = generatedCurvesDirPath.string();


    // Export geometry
    {
        std::filesystem::path outputDirectoryPath
              = std::filesystem::path(m_pathBatchOutputPath) / "BoundaryConditions.bcf";

        std::ofstream boundaryConditionFile(outputDirectoryPath.string(), std::ios::binary);
        if (!boundaryConditionFile.is_open()) {
            std::cout << "Failed to open " << outputDirectoryPath.string() << std::endl;
            return false;
        }

        boundaryConditionFile.write(
              (char *)boundaryConditions.m_startPos.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write(
              (char *)boundaryConditions.m_startDir.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write(
              (char *)boundaryConditions.m_endPos.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write(
              (char *)boundaryConditions.m_endDir.m_data.data(), sizeof(Farlor::Vector3));
        boundaryConditionFile.write((char *)&boundaryConditions.arclength, sizeof(float));
        boundaryConditionFile.write((char *)&numSegments, sizeof(uint32_t));
    }


    m_pathBatchJsonIndex.clear();

    m_pathBatchJsonIndex["experiment_name"] = m_experimentParams.experimentName;
    m_pathBatchJsonIndex["path_batch_links"] = nlohmann::json::array();

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