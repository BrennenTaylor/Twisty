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
    if (argc < 2)
    {
        fmt::print("Call as: {} pathFromLocalToDir", argv[0]);
        return false;
    }

    const std::string BinaryFilename("Paths_Binary.pbd");
    const std::string MetadataFilename("Paths_Metadata.pmd");
    const std::string FixedBinaryFilename("Paths_FixedOrder.pbd");

    std::string pathFromLocal(argv[1]);

    std::filesystem::path currentPath = std::filesystem::current_path();

    std::filesystem::path pathsDirectoryPath = currentPath;
    pathsDirectoryPath.append(pathFromLocal);
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    std::filesystem::path rawBinaryFullPath(pathsDirectoryPath);
    rawBinaryFullPath.append(BinaryFilename);
    std::cout << "rawBinaryFullPath: " << rawBinaryFullPath << std::endl;
    std::ifstream rawBinaryFile(rawBinaryFullPath.c_str(), std::ios::binary);
    

    std::filesystem::path metadataFullPath(pathsDirectoryPath);
    metadataFullPath.append(MetadataFilename);
    std::cout << "metadataFullPath: " << metadataFullPath << std::endl;
    std::ifstream metadataFile(metadataFullPath.c_str());

    std::filesystem::path fixedBinaryFullPath(pathsDirectoryPath);
    fixedBinaryFullPath.append(FixedBinaryFilename);
    std::cout << "fixedBinaryFullPath: " << fixedBinaryFullPath << std::endl;
    std::ofstream fixedBinaryFile(fixedBinaryFullPath.c_str(), std::ios::binary);


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

    struct Metadata
    {
        uint32_t  threadIdx = 0;
        uint32_t  pathBatchCount = 0;
        uint32_t  pathCount = 0;
        uint32_t indexIntoFile = 0;
    };

    std::vector<Metadata> pathMetadata;

    uint32_t numPathsReadMetadata = 0;
    std::string line;
    while (std::getline(metadataFile, line))
    {
        if (line.empty())
        {
            break;
        }

        std::stringstream lineSS(line);

        Metadata entry;
        lineSS >> entry.threadIdx;
        lineSS >> entry.pathBatchCount;
        lineSS >> entry.pathCount;

        entry.indexIntoFile = numPathsReadMetadata;
        
        numPathsReadMetadata += entry.pathCount;
        pathMetadata.push_back(entry);
    }

    std::sort(pathMetadata.begin(), pathMetadata.end(),
        [](const Metadata& left, const Metadata& right) {
            if (left.threadIdx != right.threadIdx)
            {
                return left.threadIdx < right.threadIdx;
            }

            return left.pathBatchCount < right.pathBatchCount;
        });


    std::cout << "Num paths read: " << numPathsReadMetadata << std::endl;

    // Num bytes per float * 3 floats per pos * (m + 1) pos per curve
    const uint32_t bytesPerCurve = sizeof(Farlor::Vector3) * (numSegments + 1);
    std::vector<Farlor::Vector3> tempStorage(numSegments + 1);
    std::cout << "Bytes per curve: " << bytesPerCurve << std::endl;

    //for (uint32_t pathIdx = 0; pathIdx < numPathsReadMetadata; ++pathIdx)
    //{
    //    rawBinaryFile.read((char*)tempStorage.data(), bytesPerCurve);
    //    if (tempStorage[0] != upInitialCurve->m_basePos)
    //    {
    //        std::cout << "Path Failed: " << pathIdx << std::endl;
    //        std::cout << "\tRead Start Pos: " << tempStorage[0] << std::endl;
    //        std::cout << "\tActual Start Pos: " << upInitialCurve->m_basePos << std::endl;
    //    }

    //    if (tempStorage[numSegments] != upInitialCurve->m_targetPos)
    //    {
    //        std::cout << "Path Failed: " << pathIdx << std::endl;
    //        std::cout << "\tRead Target Pos: " << tempStorage[numSegments] << std::endl;
    //        std::cout << "\tActual Target Pos: " << upInitialCurve->m_targetPos << std::endl;
    //    }
    //}

    uint64_t numRead = 0;
    for (auto& entry : pathMetadata)
    {
        std::cout << "<" << entry.threadIdx << ", " << entry.pathBatchCount << ", " << entry.pathBatchCount << ", " << entry.indexIntoFile << ">" << std::endl;

        uint64_t numPosInCurve = (numSegments + 1);
        uint64_t numPosInEntry = entry.pathCount * numPosInCurve;
        uint64_t numBytesInEntry = numPosInEntry * sizeof(Farlor::Vector3);

        numRead += entry.pathCount;

        if (tempStorage.size() < numPosInEntry)
        {
            tempStorage.resize(numPosInEntry);
        }

        uint64_t seekLocation = entry.indexIntoFile;
        std::cout << "Seek location: " << seekLocation << std::endl;
        //assert((seekLocation % (uint64_t)bytesPerCurve) == 0);
        assert((numBytesInEntry % (uint64_t)bytesPerCurve) == 0);

        //// Seek the number of bytes
        //if (seekLocation > 0)
        //{
        //    rawBinaryFile.seekg(bytesPerCurve, std::ios::beg);
        //    for (uint32_t seekIdx = 1; seekIdx < seekLocation; ++seekIdx)
        //    {
        //        rawBinaryFile.seekg(bytesPerCurve, std::ios::cur);
        //    }
        //}

        uint64_t numIdxBytes = (uint64_t)(entry.indexIntoFile) * (uint64_t)(bytesPerCurve);

        rawBinaryFile.seekg(numIdxBytes, std::ios::beg);
        rawBinaryFile.read((char*)tempStorage.data(), numBytesInEntry);
        //rawBinaryFile.seekg(0);

        //// Lets do some verification
        //bool badFound = false;
        //for (uint32_t pathIdx = 0; pathIdx < entry.pathCount; ++pathIdx)
        //{
        //    if (tempStorage[pathIdx * numPosInCurve + 0] != upInitialCurve->m_basePos)
        //    {
        //        badFound = true;
        //        std::cout << "Path Failed: " << pathIdx << std::endl;
        //        std::cout << "\tThread Idx: " << entry.threadIdx << std::endl;
        //        std::cout << "\tPath Batch Count: " << entry.pathBatchCount << std::endl;
        //        std::cout << "\tNum paths in entry: " << entry.pathCount << std::endl;
        //        std::cout << "\tRead Start Pos: " << tempStorage[pathIdx * numPosInCurve + 0] << std::endl;
        //        std::cout << "\tActual Start Pos: " << upInitialCurve->m_basePos << std::endl;
        //    }

        //    if (tempStorage[pathIdx * numPosInCurve + numSegments] != upInitialCurve->m_targetPos)
        //    {
        //        badFound = true;
        //        std::cout << "Path Failed: " << pathIdx << std::endl;
        //        std::cout << "\tThread Idx: " << entry.threadIdx << std::endl;
        //        std::cout << "\tPath Batch Count: " << entry.pathBatchCount << std::endl;
        //        std::cout << "\tNum paths in entry: " << entry.pathCount << std::endl;
        //        std::cout << "\tRead Target Pos: " << tempStorage[pathIdx * numPosInCurve + numSegments] << std::endl;
        //        std::cout << "\tActual Target Pos: " << upInitialCurve->m_targetPos << std::endl;
        //    }

        //    if (badFound)
        //    {
        //        break;
        //    }
        //}

        //if (badFound)
        //{
        //    for (uint32_t pathIdx = 0; pathIdx < entry.pathCount; ++pathIdx)
        //    {
        //        std::cout << "Path: " << pathIdx << std::endl;

        //        for (uint32_t pointIdx = 0; pointIdx < numPosInCurve; ++pointIdx)
        //        {
        //            std::cout << "\t" << tempStorage[pathIdx * numPosInCurve + pointIdx] << std::endl;
        //        }
        //    }
        //    return 1;
        //}

        fixedBinaryFile.write((char*)tempStorage.data(), numBytesInEntry);
    }

    std::cout << "Num Read: " << numRead << std::endl;
}