#include <boost\multiprecision\cpp_dec_float.hpp>

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

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Call as: %s pathFromLocalToDir", argv[0]);
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

    if (!std::filesystem::exists(indexPath)) {
        std::cout << indexPath << " file does not exist" << std::endl;
        return 1;
    }

    std::fstream indexFS(indexPath);
    if (!indexFS.is_open()) {
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
    if (!seedCurveFS.is_open()) {
        printf("Failed to open %s\n", seedCurvePath.string());
        return false;
    }

    // We need to create an initial curve object
    std::unique_ptr<twisty::Curve> upInitialCurve = twisty::Curve::LoadCurveFromStream(seedCurveFS);

    struct Metadata {
        uint32_t threadIdx = 0;
        uint32_t pathBatchCount = 0;
        uint32_t pathCount = 0;
        uint32_t indexIntoFile = 0;
    };

    std::vector<Metadata> pathMetadata;

    uint32_t numPathsReadMetadata = 0;
    std::string line;
    while (std::getline(metadataFile, line)) {
        if (line.empty()) {
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
          [](const Metadata &left, const Metadata &right) {
              if (left.threadIdx != right.threadIdx) {
                  return left.threadIdx < right.threadIdx;
              }

              return left.pathBatchCount < right.pathBatchCount;
          });


    std::cout << "Num paths read: " << numPathsReadMetadata << std::endl;

    // Num bytes per float * 3 floats per pos * (m + 1) pos per curve
    const uint32_t bytesPerCurve = sizeof(Farlor::Vector3) * (upInitialCurve->m_numSegments + 1);
    std::vector<Farlor::Vector3> tempStorage(upInitialCurve->m_numSegments + 1);
    std::cout << "Bytes per curve: " << bytesPerCurve << std::endl;


    uint64_t numRead = 0;
    for (auto &entry : pathMetadata) {
        std::cout << "<" << entry.threadIdx << ", " << entry.pathBatchCount << ", "
                  << entry.pathBatchCount << ", " << entry.indexIntoFile << ">" << std::endl;

        uint64_t numPosInCurve = (upInitialCurve->m_numSegments + 1);
        uint64_t numPosInEntry = entry.pathCount * numPosInCurve;
        uint64_t numBytesInEntry = numPosInEntry * sizeof(Farlor::Vector3);

        numRead += entry.pathCount;

        if (tempStorage.size() < numPosInEntry) {
            tempStorage.resize(numPosInEntry);
        }

        uint64_t seekLocation = entry.indexIntoFile;
        std::cout << "Seek location: " << seekLocation << std::endl;
        assert((numBytesInEntry % (uint64_t)bytesPerCurve) == 0);

        uint64_t numIdxBytes = (uint64_t)(entry.indexIntoFile) * (uint64_t)(bytesPerCurve);

        rawBinaryFile.seekg(numIdxBytes, std::ios::beg);
        rawBinaryFile.read((char *)tempStorage.data(), numBytesInEntry);
        fixedBinaryFile.write((char *)tempStorage.data(), numBytesInEntry);
    }

    std::cout << "Num Read: " << numRead << std::endl;
}