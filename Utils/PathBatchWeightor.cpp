#include "CurveUtils.h"
#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"
#include "ExperimentRunner.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <chrono>
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

int main(int argc, char* argv[])
{
    std::cout << "Not implemented atm. Commented out and needs updating." << std::endl;

//    if (argc < 2)
//    {
//        std::string programName(argv[0]);
//        fmt::print("Call as: {} Directory\n", programName);
//        return 1;
//    }
//
//    std::string experimentDirectoryName = argv[1];
//    fmt::print("Experiment director name: {}\n", experimentDirectoryName);
//
//    std::filesystem::path experimentDirectoryPath = GetExperimentDirectory(experimentDirectoryName);
//    std::cout << "Experiment Directory: " << experimentDirectoryPath << std::endl;
//
//    if (!std::filesystem::exists(experimentDirectoryPath))
//    {
//        std::cout << experimentDirectoryPath << " does not exist" << std::endl;
//        return 1;
//    }
//
//    std::filesystem::path indexPath = experimentDirectoryPath;
//    indexPath.append("index.json");
//
//    if (!std::filesystem::exists(indexPath))
//    {
//        std::cout << indexPath << " file does not exist" << std::endl;
//        return 1;
//    }
//
//    std::fstream indexFS(indexPath);
//    if (!indexFS.is_open())
//    {
//        std::cout << "Failed to open: " << indexPath << std::endl;
//        return 1;
//    }
//
//    rapidjson::IStreamWrapper indexFS_wrapper(indexFS);
//    rapidjson::Document indexDocument;
//    indexDocument.ParseStream(indexFS_wrapper);
//
//    assert(indexDocument.IsObject());
//
//    assert(indexDocument.HasMember("experiment_name"));
//    assert(indexDocument["experiment_name"].IsString());
//    std::string experimentName = indexDocument["experiment_name"].GetString();
//
//    assert(indexDocument.HasMember("seed_curve"));
//    assert(indexDocument["seed_curve"].IsString());
//    std::string seedCurveFilename = indexDocument["seed_curve"].GetString();
//    std::filesystem::path seedCurvePath = experimentDirectoryPath;
//    seedCurvePath.append(seedCurveFilename);
//
//    std::cout << "Seed curve path: " << seedCurvePath << std::endl;
//
//    std::ifstream seedCurveFS(seedCurvePath, std::ios::binary);
//    if (!seedCurveFS.is_open())
//    {
//        fmt::print("Failed to open {}\n", seedCurvePath.string());
//        return false;
//    }
//
//    // We need to create an initial curve object
//    std::unique_ptr<twisty::Curve> upInitialCurve = twisty::Curve::LoadCurveFromStream(seedCurveFS);
//
//    // Iterate over array
//    assert(indexDocument.HasMember("path_batch_links"));
//    assert(indexDocument["path_batch_links"].IsArray());
//
//    auto pathBatchLinksValue = indexDocument["path_batch_links"].GetArray();
//
//    const double mu = 0.1;
//    const uint32_t numStepsInt = 2000;
//    const double minBound = 0.0;
//    const double maxBound = 100.0;
//    const double eps = 0.5f;
//
//    float ds = upInitialCurve->m_arclength / 200.0f;
//    float scatter = 0.08f / ds;
//
//    double minCurvature = 0.0f;
//    double maxCurvature = (2.0f / (upInitialCurve->m_arclength / upInitialCurve->m_numSegments)) * 1.1f;
//    uint32_t numCurvatureSteps = 100000;
//
//    /*twisty::PathSpaceUtils::RegularizedIntegral regIntEvaluator(ds, mu, numStepsInt, minBound, maxBound, eps);*/
//    twisty::PathSpaceUtils::LookupTableIntegral lookupEvaluator(ds, mu, numStepsInt, minBound, maxBound, eps,
//        minCurvature, maxCurvature, numCurvatureSteps, scatter);
//
//    std::stringstream weightSS;
//    weightSS << experimentName;
//    weightSS << "_Experiment_Weights.tew";
//    std::string weightFilename = weightSS.str();
//
//    std::filesystem::path experimentWeightPath = experimentDirectoryPath;
//    experimentWeightPath.append(weightFilename);
//    std::ofstream experimentWeightOS(experimentWeightPath);
//    if (!experimentWeightOS.is_open())
//    {
//        std::cout << "Failed to open: " << experimentWeightPath << std::endl;
//    }
//    std::cout << "Experiment weight path: " << experimentWeightPath << std::endl;
//
//    auto experimentWeightingStart = std::chrono::high_resolution_clock::now();
//
//    std::vector<long long> pathBatchTiming;
//
//    uint32_t pathBatchIdx = 0;
//    uint32_t runningPathCount = 0;
//    twisty::BigFloat totalExperimentWeight = 0.0;
//    for (auto itr = pathBatchLinksValue.Begin(); itr != pathBatchLinksValue.End(); itr++)
//    {
//        assert(itr->IsString());
//        std::string pathBatchFilename = itr->GetString();
//
//        std::filesystem::path pathBatchPath = experimentDirectoryPath;
//        pathBatchPath.append(pathBatchFilename);
//
//        std::cout << "Path Batch path: " << pathBatchPath << std::endl;
//
//        std::ifstream pathBatchFS(pathBatchPath, std::ios::binary);
//        if (!pathBatchFS.is_open())
//        {
//            fmt::print("Failed to open {}\n", pathBatchPath.string());
//            return false;
//        }
//
//        // Load in the number of paths in the batch from file
//        twisty::Curve& initialCurve = *upInitialCurve;
//
//        uint32_t numPaths = 0;
//        pathBatchFS.read((char*)& numPaths, sizeof(uint32_t));
//
//        std::stringstream pathBatchWeightSS;
//        pathBatchWeightSS << experimentName;
//        pathBatchWeightSS << "_PathBatch_Weights_";
//        pathBatchWeightSS << pathBatchIdx;
//        pathBatchWeightSS << ".pbw";
//        std::string pathBatchWeightFilename = pathBatchWeightSS.str();
//
//        std::filesystem::path pathBatchWeightPath = experimentDirectoryPath;
//        pathBatchWeightPath.append(pathBatchWeightFilename);
//        std::ofstream pathBatchWeightOS(pathBatchWeightPath);
//        if (!pathBatchWeightOS.is_open())
//        {
//            std::cout << "Failed to open: " << pathBatchWeightPath << std::endl;
//        }
//        std::cout << "Path batch weight path: " << pathBatchWeightPath << std::endl;
//
//        twisty::BigFloat totalPathBatchWeight = 0.0;
//        std::vector<twisty::Curve> curves(numPaths);
//
//        twisty::ExperimentRunner::PathBatch pb;
//        pb.m_curvatures = std::vector<float>(numPaths * initialCurve.m_numSegments);
//        pb.m_positions = std::vector<Farlor::Vector3>(numPaths * initialCurve.m_numSegments);
//        pb.m_tangents = std::vector<Farlor::Vector3>(numPaths * initialCurve.m_numSegments);
//        pb.numberOfPathsInBatch = numPaths;
//
//        pathBatchFS.read((char*)& pb.m_curvatures[0], sizeof(float) * numPaths * initialCurve.m_numSegments);
//        pathBatchFS.read((char*)& pb.m_positions[0], sizeof(Farlor::Vector3) * numPaths * initialCurve.m_numSegments);
//        pathBatchFS.read((char*)& pb.m_tangents[0], sizeof(Farlor::Vector3) * numPaths * initialCurve.m_numSegments);
//
//        for (uint32_t pathIdx = 0; pathIdx < numPaths; ++pathIdx)
//        {
//            twisty::Curve pathBatchCurve = initialCurve;
//            memcpy(&pathBatchCurve.m_curvatures[0], &pb.m_curvatures[pathIdx * initialCurve.m_numSegments], sizeof(float) * initialCurve.m_numSegments);
//            memcpy(&pathBatchCurve.m_positions[0], &pb.m_positions[pathIdx * initialCurve.m_numSegments], sizeof(Farlor::Vector3) * initialCurve.m_numSegments);
//            memcpy(&pathBatchCurve.m_tangents[0], &pb.m_tangents[pathIdx * initialCurve.m_numSegments], sizeof(Farlor::Vector3) * initialCurve.m_numSegments);
//
//            // Copy it over
//            curves[pathIdx] = pathBatchCurve;
//        }
//
//        // Initialize the curve weightor
//        // This is before the multithreaded part, so its ok to construct the table
//        // However, once we actually start the parallel weight calculation step, we need to stick to read only.
//        //PathWeightorHomogeneousMedium pathWeightor;
//
//        auto pathBatchWeightingStart = std::chrono::high_resolution_clock::now();
//
//        std::vector<twisty::BigFloat> pathWeights(numPaths);
//#pragma omp parallel for shared(curves, pathWeights, totalPathBatchWeight, lookupEvaluator)
//        for (int32_t pathIdx = 0; pathIdx < numPaths; ++pathIdx)
//        {
//            twisty::Curve& pathBatchCurve = curves[pathIdx];
//            twisty::BigFloat pathWeight = twisty::PathSpaceUtils::WeightPath(pathBatchCurve, [](Farlor::Vector3 pos)->float {return 0.0f; }, [scatter](Farlor::Vector3 pos)->float {return scatter; }, lookupEvaluator);
//
//            pathWeights[pathIdx] = pathWeight;
//        }
//
//        // Calculate timing of actualy weighting of paths
//        auto pathBatchWeightingEnd = std::chrono::high_resolution_clock::now();
//        auto pathBatchTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(pathBatchWeightingEnd - pathBatchWeightingStart);
//        pathBatchTiming.push_back(pathBatchTimeMs.count());
//
//        for (uint32_t pathIdx = 0; pathIdx < numPaths; ++pathIdx)
//        {
//            totalPathBatchWeight += pathWeights[pathIdx];
//
//            twisty::BigFloat runningAverage = totalPathBatchWeight / (pathIdx + 1);
//            pathBatchWeightOS << runningPathCount + pathIdx << ", " << pathWeights[pathIdx] << ", " << runningAverage << std::endl;
//        }
//
//        pathBatchWeightOS << "Total, " << totalPathBatchWeight << std::endl;
//        pathBatchWeightOS << "Average, " << totalPathBatchWeight / numPaths << std::endl;
//
//        pathBatchWeightOS.close();
//
//        pathBatchIdx++;
//        runningPathCount += numPaths;
//        totalExperimentWeight += totalPathBatchWeight;
//    }
//
//    auto experimentWeightingEnd = std::chrono::high_resolution_clock::now();
//    auto experimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(experimentWeightingEnd - experimentWeightingStart);
//
//    experimentWeightOS << "Total, " << totalExperimentWeight << std::endl;
//    experimentWeightOS << "Average, " << totalExperimentWeight / runningPathCount << std::endl;
//
//    std::cout << "Experiment Weighting Timing: " << std::endl;
//    std::cout << "\tExperiment Time (ms): " << experimentTimeMs.count() << std::endl;
//    unsigned long long totalPbTime = 0;
//    for (uint32_t i = 0; i < pathBatchTiming.size(); i++)
//    {
//        totalPbTime += pathBatchTiming[i];
//        std::cout << "\tPath Batch " << i << " time (ms): " << pathBatchTiming[i] << std::endl;
//    }
//    std::cout << "\tTotal Pb Time (ms): " << totalPbTime << std::endl;
//    std::cout << "\tAvg Pb Time (ms): " << static_cast<float>(totalPbTime) / static_cast<float>(pathBatchIdx) << std::endl;
//
//    experimentWeightOS.close();
}