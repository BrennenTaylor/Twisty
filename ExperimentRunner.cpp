#include "ExperimentRunner.h"

#include <filesystem>
#include <sstream>

namespace twisty
{
    ExperimentRunner::ExperimentRunner(ExperimentParameters& experimentParams, Bootstrapper& bootstrapper)
        : m_experimentParams(experimentParams)
        , m_bootstrapper(bootstrapper)
        , m_upInitialCurve(nullptr)
        , m_pathBatchJsonIndex(rapidjson::kObjectType)
        , m_pathBatchLinks()
    {
    }

    ExperimentRunner::~ExperimentRunner()
    {
    }

    bool ExperimentRunner::BeginPathBatchOutput()
    {
        std::filesystem::path outputDirectoryPath = std::filesystem::current_path();
        outputDirectoryPath.append(m_experimentParams.experimentName);
        if (!std::filesystem::exists(outputDirectoryPath))
        {
            std::filesystem::create_directory(outputDirectoryPath);
        }

        m_pathBatchJsonIndex.RemoveAllMembers();
        // Define the document as an object, not an array
        m_pathBatchJsonIndex.SetObject();

        rapidjson::Document::AllocatorType& allocator = m_pathBatchJsonIndex.GetAllocator();

        rapidjson::Value experimentNameValue(m_experimentParams.experimentName.c_str(), allocator);
        m_pathBatchJsonIndex.AddMember("experiment_name", experimentNameValue, allocator);

        std::stringstream seedCurveSS;
        seedCurveSS << m_experimentParams.experimentName;
        seedCurveSS << "_Seed_Curve.tcf";

        std::filesystem::path seedCurvePath = outputDirectoryPath;
        seedCurvePath.append(seedCurveSS.str());

        rapidjson::Value seedCurveFilenameValue(seedCurveSS.str().c_str(), allocator);
        m_pathBatchJsonIndex.AddMember("seed_curve", seedCurveFilenameValue, allocator);

        std::ofstream seedCurveOutfile(seedCurvePath.string(), std::ios::binary);

        m_pathBatchLinks = rapidjson::Value(rapidjson::kArrayType);

        // We need to output base curve information first
        seedCurveOutfile.write((char*)&m_upInitialCurve->m_numSegments, sizeof(uint32_t));
        seedCurveOutfile.write((char*)&m_upInitialCurve->m_arclength, sizeof(float));

        seedCurveOutfile.write((char*)&m_upInitialCurve->m_basePos, sizeof(float) * 3);
        seedCurveOutfile.write((char*)&m_upInitialCurve->m_baseTangent, sizeof(float) * 3);
        seedCurveOutfile.write((char*)&m_upInitialCurve->m_targetPos, sizeof(float) * 3);
        seedCurveOutfile.write((char*)&m_upInitialCurve->m_targetTangent, sizeof(float) * 3);

        seedCurveOutfile.write((char*)&m_upInitialCurve->m_curvatures[0], sizeof(float) * m_upInitialCurve->m_numSegments);
        seedCurveOutfile.write((char*)&m_upInitialCurve->m_positions[0], sizeof(Farlor::Vector3) * m_upInitialCurve->m_numSegments);
        seedCurveOutfile.write((char*)&m_upInitialCurve->m_tangents[0], sizeof(Farlor::Vector3) * m_upInitialCurve->m_numSegments);

        seedCurveOutfile.write((char*)&m_experimentParams.numPathsInExperiment, sizeof(uint32_t));

        return true;
    }

    void ExperimentRunner::OutputPathBatch(PathBatch& pathBatch)
    {
        rapidjson::Document::AllocatorType& allocator = m_pathBatchJsonIndex.GetAllocator();

        std::stringstream pathBatchNameSS;
        pathBatchNameSS << m_experimentParams.experimentName;
        pathBatchNameSS << "_PathBatch_";
        pathBatchNameSS << pathBatch.index;
        pathBatchNameSS << ".tpb";

        std::filesystem::path outputDirectoryPath = std::filesystem::current_path();
        outputDirectoryPath.append(m_experimentParams.experimentName);

        std::filesystem::path pathBatchFilename = outputDirectoryPath;
        pathBatchFilename.append(pathBatchNameSS.str());

        std::ofstream pathBatchOutfile(pathBatchFilename.string(), std::ios::binary);
        if (!pathBatchOutfile.is_open())
        {
            std::cout << "Failed to open " << pathBatchFilename.string() << std::endl;
            return;
        }

        rapidjson::Value pathBatchValue;
        pathBatchValue.SetString(pathBatchNameSS.str().c_str(), allocator);
        m_pathBatchLinks.PushBack(pathBatchValue, allocator);

        pathBatchOutfile.write((char*)& pathBatch.numberOfPathsInBatch, sizeof(uint32_t));
        pathBatchOutfile.write((char*)&pathBatch.m_curvatures[0], sizeof(float) * m_experimentParams.numSegmentsPerCurve * pathBatch.numberOfPathsInBatch);
        pathBatchOutfile.write((char*)&pathBatch.m_positions[0], sizeof(Farlor::Vector3) * m_experimentParams.numSegmentsPerCurve * pathBatch.numberOfPathsInBatch);
        pathBatchOutfile.write((char*)&pathBatch.m_tangents[0], sizeof(Farlor::Vector3) * m_experimentParams.numSegmentsPerCurve * pathBatch.numberOfPathsInBatch);

        // Also, we will write it out to an individual batch file
        pathBatchOutfile.close();
        std::cout << "Completed writing path batch file" << std::endl;
    }

    void ExperimentRunner::EndPathBatchOutput()
    {
        m_pathBatchJsonIndex.AddMember("path_batch_links", m_pathBatchLinks,
            m_pathBatchJsonIndex.GetAllocator());

        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        m_pathBatchJsonIndex.Accept(writer);

        std::stringstream indexJsonSS;
        indexJsonSS << "index.json";

        std::filesystem::path outputDirectoryPath = std::filesystem::current_path();
        outputDirectoryPath.append(m_experimentParams.experimentName);

        std::filesystem::path indexJsonPath = outputDirectoryPath;
        indexJsonPath.append(indexJsonSS.str());

        std::ofstream jsonOfstream(indexJsonPath.string());
        jsonOfstream << buffer.GetString() << std::endl;
        jsonOfstream.close();
    }
}