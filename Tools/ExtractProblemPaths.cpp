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

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fmt::print("Call as: {} PathToExp FilenameOnly threshold", argv[0]);
        return false;
    }

    const std::string FixedBinaryFilename("Paths_FixedOrder.pbd");
    const std::string ProblemPathsFilename("Paths_Problem.pbd");
    const std::string ProblemPathsMetadataFilename("Paths_Problem_Metadata.txt");

    std::string pathToExperiment = std::string(argv[1]);

    std::string weightsFilename = std::string(argv[2]);
    weightsFilename += ".txt";

    float threshold = std::stof(argv[3]);

    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path pathsDirectoryPath = currentPath;
    pathsDirectoryPath.append(pathToExperiment);
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    std::filesystem::path fixedBinaryFullPath(pathsDirectoryPath);
    fixedBinaryFullPath.append(FixedBinaryFilename);
    std::cout << "fixedBinaryFullPath: " << fixedBinaryFullPath << std::endl;

    std::filesystem::path problemPathsFullPath(pathsDirectoryPath);
    problemPathsFullPath.append(ProblemPathsFilename);
    std::cout << "problemPathsFullPath: " << problemPathsFullPath << std::endl;

    std::filesystem::path problemPathsMetadataFullPath(pathsDirectoryPath);
    problemPathsMetadataFullPath.append(ProblemPathsMetadataFilename);
    std::cout << "problemPathsMetadataFullPath: " << problemPathsMetadataFullPath << std::endl;

    std::filesystem::path weightValuesFullPath(pathsDirectoryPath);
    weightValuesFullPath.append(weightsFilename);
    std::cout << "weightValuesFullPath: " << weightValuesFullPath << std::endl;


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



    std::ifstream inDataFile(weightValuesFullPath);
    uint32_t numWeightValues = 0;
    inDataFile >> numWeightValues;

    std::cout << "Number of weight values: " << numWeightValues << std::endl;

    // Open up files
    std::ifstream rawPathFile(fixedBinaryFullPath, std::ios::binary);
    std::ofstream problemPathFile(problemPathsFullPath, std::ios::binary);
    std::ofstream problemPathMetadataFile(problemPathsMetadataFullPath);

    // Stores the number of floats needed for 2 paths
    const uint64_t numSegmentsPerPath = 200;
    const uint64_t numPosPerPath = (numSegmentsPerPath + 1);
    const uint64_t numBytesPerPath = numPosPerPath * sizeof(Farlor::Vector3);
    std::vector<Farlor::Vector3> dataBuffer(numPosPerPath * 2);

    std::vector<Farlor::Vector3> tanBuffer(numPosPerPath * 2);
    std::vector<float> curvatureBuffer(numSegmentsPerPath * 2);

    // Read em all in
    boost::multiprecision::cpp_dec_float_100 runningPathSum = 0.0;
    boost::multiprecision::cpp_dec_float_100 previousVal = 0.0;
    boost::multiprecision::cpp_dec_float_100 previousAvg = 1.0;

    uint64_t numOutput = 0;

    for (uint64_t i = 0; i < numWeightValues; ++i)
    {
        boost::multiprecision::cpp_dec_float_100 currentVal = 0.0;
        inDataFile >> currentVal;

        runningPathSum += currentVal;
        boost::multiprecision::cpp_dec_float_100 avg = runningPathSum / (boost::multiprecision::cpp_dec_float_100)(i + 1);

        if ((avg / previousAvg) > threshold)
        {
            problemPathMetadataFile << "Export Idx: " << numOutput << std::endl;
            problemPathMetadataFile << "prevPathIdx: " << i - 1 << std::endl;
            problemPathMetadataFile << "currentPathIdx: " << i << std::endl;
            problemPathMetadataFile << "Previous Avg: " << previousAvg << std::endl;
            problemPathMetadataFile << "Current Avg: " << avg << std::endl;
            problemPathMetadataFile << "Previous Val: " << previousVal << std::endl;
            problemPathMetadataFile << "Current Val: " << currentVal << std::endl;


            // Do prev path
            double prevPathWeight = 0.0;
            {
                problemPathMetadataFile << "Prev Path: " << std::endl;
                // Update points from current buffer
                problemPathMetadataFile << "Points: " << std::endl;
                for (int32_t pointIdx = 0; pointIdx < numPosPerPath; ++pointIdx)
                {
                    problemPathMetadataFile << "\t" << dataBuffer[pointIdx] << std::endl;
                }

                // Update all tangents
                problemPathMetadataFile << "Tans: " << std::endl;
                for (int32_t tanIdx = 0; tanIdx < numSegmentsPerPath; ++tanIdx)
                {
                    Farlor::Vector3 leftPos = dataBuffer[tanIdx];
                    Farlor::Vector3 rightPos = dataBuffer[tanIdx + 1];

                    tanBuffer[tanIdx] = (rightPos - leftPos).Normalized();

                    problemPathMetadataFile << "\t" << tanBuffer[tanIdx] << std::endl;
                }

                // Update curvature values
                problemPathMetadataFile << "Curvatures: " << std::endl;
                for (int32_t curvatureIdx = 0; curvatureIdx < numSegmentsPerPath; ++curvatureIdx)
                {
                    Farlor::Vector3 leftTan = tanBuffer[curvatureIdx];
                    Farlor::Vector3 rightTan = tanBuffer[(curvatureIdx + 1)];

                    Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);

                    const float curvature = temp.Magnitude();
                    curvatureBuffer[curvatureIdx] = curvature;

                    problemPathMetadataFile << "\t" << curvatureBuffer[curvatureIdx] << std::endl;

                    //// Also, cache the weight of that changed segment
                    //float distance = curvature - minCurvature;
                    //float realIdx = distance / curvatureStepSize;
                    //int32_t leftIdx = floor(realIdx);
                    //int32_t rightIdx = leftIdx + 1;

                    //double leftLookup = lookupTable[leftIdx];
                    //double rightLookup = lookupTable[rightIdx];

                    //float leftDist = distance - (leftIdx * curvatureStepSize);

                    //double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                    ////std::cout << "Interpolated result: " << interpolatedResult << std::endl;
                    //double interpolatedResultLog = log(interpolatedResult);

                    //double segmentWeight = interpolatedResultLog;

                    //// Take natural log of this constant
                    //segmentWeight += lnAbsorbtionConst;

                    //// Add segment weighting into running path weight
                    //prevPathWeight += segmentWeight;
                }
            }


            // Do next path
            double nextPathWeight = 0.0;
            {
                problemPathMetadataFile << "\tNext Path: " << std::endl;
                // Update points from current buffer
                problemPathMetadataFile << "Points: " << std::endl;
                for (int32_t pointIdx = 0; pointIdx < numPosPerPath; ++pointIdx)
                {
                    problemPathMetadataFile << "\t" << dataBuffer[pointIdx + numPosPerPath] << std::endl;
                }

                // Update all tangents
                problemPathMetadataFile << "Tans: " << std::endl;
                for (int32_t tanIdx = 0; tanIdx < numSegmentsPerPath; ++tanIdx)
                {
                    Farlor::Vector3 leftPos = dataBuffer[tanIdx + numPosPerPath];
                    Farlor::Vector3 rightPos = dataBuffer[tanIdx + 1 + numPosPerPath];

                    tanBuffer[tanIdx + numPosPerPath] = (rightPos - leftPos).Normalized();

                    problemPathMetadataFile << "\t" << tanBuffer[tanIdx + numPosPerPath] << std::endl;
                }

                // Update curvature values
                problemPathMetadataFile << "Curvatures: " << std::endl;
                for (int32_t curvatureIdx = 0; curvatureIdx < numSegmentsPerPath; ++curvatureIdx)
                {
                    Farlor::Vector3 leftTan = tanBuffer[curvatureIdx + numPosPerPath];
                    Farlor::Vector3 rightTan = tanBuffer[(curvatureIdx + 1) + numPosPerPath];

                    Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);

                    const float curvature = temp.Magnitude();
                    curvatureBuffer[curvatureIdx + numSegmentsPerPath] = curvature;

                    problemPathMetadataFile << "\t" << curvatureBuffer[curvatureIdx + numSegmentsPerPath] << std::endl;

                    //// Also, cache the weight of that changed segment
                    //float distance = curvature - minCurvature;
                    //float realIdx = distance / curvatureStepSize;
                    //int32_t leftIdx = floor(realIdx);
                    //int32_t rightIdx = leftIdx + 1;

                    //double leftLookup = lookupTable[leftIdx];
                    //double rightLookup = lookupTable[rightIdx];

                    //float leftDist = distance - (leftIdx * curvatureStepSize);

                    //double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                    ////std::cout << "Interpolated result: " << interpolatedResult << std::endl;
                    //double interpolatedResultLog = log(interpolatedResult);

                    //double segmentWeight = interpolatedResultLog;

                    //// Take natural log of this constant
                    //segmentWeight += lnAbsorbtionConst;

                    //// Add segment weighting into running path weight
                    //prevPathWeight += segmentWeight;
                }
            }




            problemPathMetadataFile << std::endl;

            const uint64_t prevOffsetBytes = (i - 1) * (uint64_t)numBytesPerPath;
            rawPathFile.seekg(prevOffsetBytes, std::ios::beg);
            rawPathFile.read((char*)dataBuffer.data(), numBytesPerPath * 2);
            problemPathFile.write((char*)dataBuffer.data(), numBytesPerPath * 2);

            numOutput++;
        }

        previousVal = currentVal;
        previousAvg = avg;
    }

    problemPathMetadataFile << "Num Output: " << numOutput << std::endl;
}