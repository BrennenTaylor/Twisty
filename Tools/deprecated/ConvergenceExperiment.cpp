#include <boost\multiprecision\cpp_dec_float.hpp>

//typedef boost::multiprecision::number < boost::multiprecision::cpp_dec_float<1000>> cpp_dec_float_custom;
typedef boost::multiprecision::cpp_dec_float_100 cpp_dec_float_custom;

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <filesystem>

int main(int argc, char* argv[])
{
    std::cout << "argc: " << argc << std::endl;
    if (argc < 4)
    {
        std::cout << "Call as: " << argv[0] << " exportRate dataPathDir compressedWeightsLog10" << std::endl;
        return false;
    }

    uint32_t exportRate = std::stoi(argv[1]);
    std::cout << "Export rate: " << exportRate << std::endl;


    std::filesystem::path generatedCurvesDirPath = std::string(argv[2]);

    std::cout << "Export path: " << generatedCurvesDirPath << std::endl;

    std::string dataFilename = generatedCurvesDirPath.string();
    dataFilename += "\\BigFloatWeights.txt";


    std::string outputFilename = generatedCurvesDirPath.string();
    outputFilename += "\\Output";

    bool compressedWeightsLog10 = (std::stoi(argv[3]) == 0) ? false : true;
    std::cout << "Compressed weights: " << compressedWeightsLog10 << std::endl;

    std::cout << "Data filename: " << dataFilename << std::endl;
    std::ifstream inDataFile(dataFilename);
    if (!inDataFile.is_open()) {
        std::cout << "Failed to open data file" << std::endl;
        std::cout << "Error: " << strerror(errno) << std::endl;
        return 1;
    }
    uint32_t numWeightValues = 0;
    inDataFile >> numWeightValues;

    std::cout << "Number of weight values: " << numWeightValues << std::endl;




    const uint32_t numSplits = 1;
    std::vector<std::ofstream> outFilesBasic(numSplits);
    std::vector<std::ofstream> outFilesLn(numSplits);
    for (uint32_t i = 0; i < numSplits; ++i)
    {
        // Output Slice Weights
        std::stringstream outputSS;

        outputSS << outputFilename;
        outputSS << "_ConvergencePathWeights";
        outputSS << "_";
        outputSS << i;
        outputSS << ".txt";
        outFilesBasic[i] = std::ofstream(outputSS.str());


        std::stringstream outputLnSS;
        outputLnSS << outputFilename;
        outputLnSS << "_ConvergencePathWeights_Ln";
        outputLnSS << "_";
        outputLnSS << i;
        outputLnSS << ".txt";
        outFilesLn[i] = std::ofstream(outputLnSS.str());
    }



    // Path Counts

    uint32_t numPathsPerSplit = numWeightValues / numSplits;

    for (uint32_t splitIdx = 0; splitIdx < numSplits; ++splitIdx)
    {
        outFilesBasic[splitIdx] << "Path Counts" << std::endl;
        outFilesLn[splitIdx] << "Path Counts" << std::endl;

        for (uint32_t weightIdx = 0; weightIdx < numPathsPerSplit; ++weightIdx)
        {
            if ((weightIdx % exportRate) == 0)
            {
                outFilesBasic[splitIdx] << weightIdx << std::endl;
                outFilesLn[splitIdx] << weightIdx << std::endl;
            }
        }

        outFilesBasic[splitIdx] << "Path Weights" << std::endl;
        outFilesLn[splitIdx] << "Ln Path Weights" << std::endl;

        cpp_dec_float_custom weightMin = 0.0;
        cpp_dec_float_custom weightMax = 0.0;

        // Read em all in
        cpp_dec_float_custom runningPathSum = 0.0;
        for (uint32_t weightIdx = 0; weightIdx < numPathsPerSplit; ++weightIdx)
        {
            // Read in the current value
            double currentVal = 0.0;
            inDataFile >> currentVal;

            if (weightIdx == 0)
            {
                weightMin = currentVal;
                weightMax = currentVal;
            }
            else
            {
                if (currentVal < weightMin)
                {
                    weightMin = currentVal;
                }

                if (currentVal > weightMax)
                {
                    weightMax = currentVal;
                }
            }

            // Convert to big float
            cpp_dec_float_custom currentValBigFloat = currentVal;

            // If they are stored as big floats, we want to expand it out
            if (compressedWeightsLog10)
            {
                currentValBigFloat = boost::multiprecision::pow(10.0, currentValBigFloat);
            }

            // Keep adding in the value
            runningPathSum += currentValBigFloat;
            cpp_dec_float_custom avg = runningPathSum / (cpp_dec_float_custom)(weightIdx + 1);

            if ((weightIdx % exportRate) == 0)
            {
                outFilesBasic[splitIdx] << avg << std::endl;
                outFilesLn[splitIdx] << boost::multiprecision::log10(avg) << std::endl;
            }
        }
        outFilesLn[splitIdx] << "Min, Max" << std::endl;
        outFilesLn[splitIdx] << weightMin << std::endl;
        outFilesLn[splitIdx] << weightMax << std::endl;
    }
}