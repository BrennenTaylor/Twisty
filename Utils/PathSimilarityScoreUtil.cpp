#include "CurveUtils.h"
#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"

#include <fmt/format.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstdint>
#include <fstream>
#include <filesystem>
#include <map>

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

uint32_t CalculatePathScore(const twisty::Curve& keyCurve, const twisty::Curve& testCurve)
{
    assert(keyCurve.m_numSegments == testCurve.m_numSegments);

    uint32_t score = 0;
    for (uint32_t i = 0; i < keyCurve.m_numSegments; i++)
    {
        if (keyCurve.m_segments[i].m_curvature == testCurve.m_segments[i].m_curvature)
        {
            score++;
        }
        if (keyCurve.m_segments[i].m_torsion == testCurve.m_segments[i].m_torsion)
        {
            score++;
        }
    }
    return score;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::string programName(argv[0]);
        fmt::print("Call as: {} Directory\n", programName);
        return 1;
    }

    std::string experimentDirectoryName = argv[1];
    fmt::print("Experiment director name: {}\n", experimentDirectoryName);

    std::filesystem::path experimentDirectoryPath = GetExperimentDirectory(experimentDirectoryName);
    std::cout << "Experiment Directory: " << experimentDirectoryPath << std::endl;

    if (!std::filesystem::exists(experimentDirectoryPath))
    {
        std::cout << experimentDirectoryPath << " does not exist" << std::endl;
        return 1;
    }

    std::filesystem::path indexPath = experimentDirectoryPath;
    indexPath.append("index.json");

    if (!std::filesystem::exists(indexPath))
    {
        std::cout << indexPath << " file does not exist" << std::endl;
        return 1;
    }

    std::fstream indexFS(indexPath);
    if (!indexFS.is_open())
    {
        std::cout << "Failed to open: " << indexPath << std::endl;
        return 1;
    }

    rapidjson::IStreamWrapper indexFS_wrapper(indexFS);
    rapidjson::Document indexDocument;
    indexDocument.ParseStream(indexFS_wrapper);

    assert(indexDocument.IsObject());

    assert(indexDocument.HasMember("experiment_name"));
    assert(indexDocument["experiment_name"].IsString());
    std::string experimentName = indexDocument["experiment_name"].GetString();

    assert(indexDocument.HasMember("seed_curve"));
    assert(indexDocument["seed_curve"].IsString());
    std::string seedCurveFilename = indexDocument["seed_curve"].GetString();
    std::filesystem::path seedCurvePath = experimentDirectoryPath;
    seedCurvePath.append(seedCurveFilename);

    std::cout << "Seed curve path: " << seedCurvePath << std::endl;

    std::ifstream seedCurveFS(seedCurvePath);
    if (!seedCurveFS.is_open())
    {
        fmt::print("Failed to open {}\n", seedCurvePath.string());
        return false;
    }

    uint32_t numSegments = 0;
    seedCurveFS >> numSegments;

    // We need to create an initial curve object
    std::unique_ptr<twisty::Curve> upInitialCurve = std::make_unique<twisty::Curve>(numSegments);
    seedCurveFS >> upInitialCurve->m_arclength;
    seedCurveFS >> upInitialCurve->m_basePos.x;
    seedCurveFS >> upInitialCurve->m_basePos.y;
    seedCurveFS >> upInitialCurve->m_basePos.z;

    seedCurveFS >> upInitialCurve->m_baseTangent.x;
    seedCurveFS >> upInitialCurve->m_baseTangent.y;
    seedCurveFS >> upInitialCurve->m_baseTangent.z;

    seedCurveFS >> upInitialCurve->m_baseNormal.x;
    seedCurveFS >> upInitialCurve->m_baseNormal.y;
    seedCurveFS >> upInitialCurve->m_baseNormal.z;

    seedCurveFS >> upInitialCurve->m_baseBinormal.x;
    seedCurveFS >> upInitialCurve->m_baseBinormal.y;
    seedCurveFS >> upInitialCurve->m_baseBinormal.z;

    seedCurveFS >> upInitialCurve->m_targetPos.x;
    seedCurveFS >> upInitialCurve->m_targetPos.y;
    seedCurveFS >> upInitialCurve->m_targetPos.z;

    seedCurveFS >> upInitialCurve->m_targetTangent.x;
    seedCurveFS >> upInitialCurve->m_targetTangent.y;
    seedCurveFS >> upInitialCurve->m_targetTangent.z;

    for (uint32_t i = 0; i < upInitialCurve->m_numSegments; ++i)
    {
        seedCurveFS >> upInitialCurve->m_segments[i].m_curvature;
        seedCurveFS >> upInitialCurve->m_segments[i].m_torsion;
        upInitialCurve->m_segments[i].m_length = upInitialCurve->m_arclength / upInitialCurve->m_numSegments;
        upInitialCurve->m_segments[i].UpdateRotation();
    }

    // Iterate over array
    assert(indexDocument.HasMember("path_batch_links"));
    assert(indexDocument["path_batch_links"].IsArray());

    auto pathBatchLinksValue = indexDocument["path_batch_links"].GetArray();

    // We have read in the similarity scores

    std::stringstream experimentSS;
    experimentSS << experimentName;
    experimentSS << "_Experiment_Scores.tess";
    std::string experimentSSFilename = experimentSS.str();

    std::filesystem::path experimentSSPath = experimentDirectoryPath;
    experimentSSPath.append(experimentSSFilename);
    std::ofstream experimentSSOS(experimentSSPath);
    if (!experimentSSOS.is_open())
    {
        std::cout << "Failed to open: " << experimentSSPath << std::endl;
    }
    std::cout << "Experiment SS path: " << experimentSSPath << std::endl;

    uint32_t pathBatchIdx = 0;
    uint32_t runningPathCount = 0;
    std::map<uint32_t, uint32_t> experimentScores;
    for (uint32_t i = 0; i < upInitialCurve->m_numSegments * 2; i++)
    {
        experimentScores[i] = 0;
    }
    for (auto itr = pathBatchLinksValue.Begin(); itr != pathBatchLinksValue.End(); itr++)
    {
        assert(itr->IsString());
        std::string pathBatchFilename = itr->GetString();

        std::filesystem::path pathBatchPath = experimentDirectoryPath;
        pathBatchPath.append(pathBatchFilename);

        std::cout << "Path Batch path: " << pathBatchPath << std::endl;

        std::ifstream pathBatchFS(pathBatchPath);
        if (!pathBatchFS.is_open())
        {
            fmt::print("Failed to open {}\n", pathBatchPath.string());
            return false;
        }

        // Load in the number of paths in the batch from file
        uint32_t numPathsInBatch = 0;
        pathBatchFS >> numPathsInBatch;

        std::stringstream pathBatchScoreSS;
        pathBatchScoreSS << experimentName;
        pathBatchScoreSS << "_PathBatch_Scores_";
        pathBatchScoreSS << pathBatchIdx;
        pathBatchScoreSS << ".pbss";
        std::string pathBatchScoreFilename = pathBatchScoreSS.str();

        std::filesystem::path pathBatchScorePath = experimentDirectoryPath;
        pathBatchScorePath.append(pathBatchScoreFilename);
        std::ofstream pathBatchScoreOS(pathBatchScorePath);
        if (!pathBatchScoreOS.is_open())
        {
            std::cout << "Failed to open: " << pathBatchScorePath << std::endl;
        }
        std::cout << "Path batch score path: " << pathBatchScorePath << std::endl;

        std::map<uint32_t, uint32_t> pathBatchScores;
        for (uint32_t i = 0; i < upInitialCurve->m_numSegments * 2; i++)
        {
            pathBatchScores[i] = 0;
        }

        std::vector<twisty::Curve> curves(numPathsInBatch);
        for (uint32_t pathIdx = 0; pathIdx < numPathsInBatch; ++pathIdx)
        {
            twisty::Curve pathBatchCurve = *upInitialCurve;
            for (uint32_t segIdx = 0; segIdx < numSegments; ++segIdx)
            {
                pathBatchFS >> pathBatchCurve.m_segments[segIdx].m_curvature;
                pathBatchFS >> pathBatchCurve.m_segments[segIdx].m_torsion;
                pathBatchCurve.m_segments[segIdx].m_length = pathBatchCurve.m_arclength / pathBatchCurve.m_numSegments;
                pathBatchCurve.m_segments[segIdx].UpdateRotation();
            }
            // Copy it over
            curves[pathIdx] = pathBatchCurve;
        }

        std::vector<uint32_t> pathScores(numPathsInBatch);

#pragma omp parallel for shared(curves, pathScores)
        for (int32_t pathIdx = 0; pathIdx < numPathsInBatch; ++pathIdx)
        {
            twisty::Curve &pathBatchCurve = curves[pathIdx];
            
            uint32_t pathScore = CalculatePathScore(*upInitialCurve, pathBatchCurve);

            pathScores[pathIdx] = pathScore;
        }

        for (uint32_t pathIdx = 0; pathIdx < numPathsInBatch; ++pathIdx)
        {
            experimentScores[pathScores[pathIdx]] = experimentScores[pathScores[pathIdx]] + 1;
            pathBatchScores[pathScores[pathIdx]] = pathBatchScores[pathScores[pathIdx]] + 1;
        }

        for (const auto& entry : pathBatchScores)
        {
            pathBatchScoreOS << entry.first << ", " << entry.second << ", " << static_cast<float>(entry.second) / static_cast<float>(numPathsInBatch) << "\n";
        }

        pathBatchScoreOS.close();

        pathBatchIdx++;
        runningPathCount += numPathsInBatch;
    }

    for (const auto& entry : experimentScores)
    {
        experimentSSOS << entry.first << ", " << entry.second << ", " << static_cast<float>(entry.second) / static_cast<float>(runningPathCount) << "\n";
    }

    experimentSSOS.close();
}