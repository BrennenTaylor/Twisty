#include <boost\multiprecision\cpp_dec_float.hpp>

#include <fmt/format.h>

#include <Curve.h>

#include <FMath/Vector3.h>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        fmt::print("Call as: {} PathToExp numPathsInExperiment exportRate", argv[0]);
        return false;
    }

    const std::string FixedBinaryFilename("Paths_FixedOrder.pbd");
    const std::string SpringLengthsFilename("SpringLengths.txt");

    std::string pathToExperiment = std::string(argv[1]);
    const uint32_t numPathsInExperiment = std::stoi(argv[2]);
    const uint32_t exportRate = std::stoi(argv[3]);

    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path pathsDirectoryPath = currentPath;
    pathsDirectoryPath.append(pathToExperiment);
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    std::filesystem::path fixedBinaryFullPath(pathsDirectoryPath);
    fixedBinaryFullPath.append(FixedBinaryFilename);
    std::cout << "fixedBinaryFullPath: " << fixedBinaryFullPath << std::endl;

    std::filesystem::path springLengthPath(pathsDirectoryPath);
    springLengthPath.append(SpringLengthsFilename);
    std::cout << "springLengthPath: " << springLengthPath << std::endl;

    std::filesystem::path indexPath = pathsDirectoryPath;
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
    std::filesystem::path seedCurvePath = pathsDirectoryPath;
    seedCurvePath.append(seedCurveFilename);

    std::cout << "Seed curve path: " << seedCurvePath << std::endl;

    std::ifstream seedCurveFS(seedCurvePath, std::ios::binary);
    if (!seedCurveFS.is_open())
    {
        fmt::print("Failed to open {}\n", seedCurvePath.string());
        return false;
    }

    uint32_t numSegments = 0;
    seedCurveFS.read((char*)&numSegments, sizeof(uint32_t));

    // We need to create an initial curve object
    std::unique_ptr<twisty::Curve> upInitialCurve = std::make_unique<twisty::Curve>(numSegments);
    seedCurveFS.read((char*)&upInitialCurve->m_arclength, sizeof(float));
    seedCurveFS.read((char*)&upInitialCurve->m_basePos, sizeof(Farlor::Vector3));
    seedCurveFS.read((char*)&upInitialCurve->m_baseTangent, sizeof(Farlor::Vector3));
    seedCurveFS.read((char*)&upInitialCurve->m_targetPos, sizeof(Farlor::Vector3));
    seedCurveFS.read((char*)&upInitialCurve->m_targetTangent, sizeof(Farlor::Vector3));

    seedCurveFS.read((char*)&upInitialCurve->m_curvatures[0], sizeof(float) * upInitialCurve->m_numSegments);
    seedCurveFS.read((char*)&upInitialCurve->m_positions[0], sizeof(Farlor::Vector3) * upInitialCurve->m_numSegments);
    seedCurveFS.read((char*)&upInitialCurve->m_tangents[0], sizeof(Farlor::Vector3) * upInitialCurve->m_numSegments);

    std::cout << "Initial Curve Base Pos: " << upInitialCurve->m_basePos << std::endl;
    std::cout << "Initial Curve End Pos: " << upInitialCurve->m_targetPos << std::endl;
    float segmentLength = upInitialCurve->m_arclength / upInitialCurve->m_numSegments;

    // Open up files
    std::ifstream rawPathFile(fixedBinaryFullPath, std::ios::binary);
    std::ofstream springLengthFile(springLengthPath);

    // Stores the number of floats needed for 2 paths
    const uint64_t numSegmentsPerPath = upInitialCurve->m_numSegments;
    const uint64_t numPosPerPath = (numSegmentsPerPath + 1);
    const uint64_t numBytesPerPath = numPosPerPath * sizeof(Farlor::Vector3);
    std::vector<Farlor::Vector3> dataBuffer(numPosPerPath);

    uint64_t numOutput = 0;

    springLengthFile << numPathsInExperiment << std::endl;
    springLengthFile << "Spring Lengths: " << std::endl;

    for (uint64_t i = 0; i < numPathsInExperiment; ++i)
    {
        const uint64_t offset = i * (uint64_t)numBytesPerPath;
        rawPathFile.seekg(offset, std::ios::beg);
        rawPathFile.read((char*)dataBuffer.data(), numBytesPerPath);

        double pathLength = 0.0;
        for (uint32_t segIdx = 0; segIdx < (numSegmentsPerPath-1); ++segIdx)
        {
            auto& leftPoint = dataBuffer[segIdx];
            auto& rightPoint = dataBuffer[segIdx + 1];

            double segLength = (rightPoint - leftPoint).Magnitude();
            pathLength += segLength;
        }

        if ((i % exportRate) == 0)
        {
            springLengthFile << pathLength << std::endl;
        }
    }
}