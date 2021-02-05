#include <boost\multiprecision\cpp_dec_float.hpp>

//typedef boost::multiprecision::number < boost::multiprecision::cpp_dec_float<1000>> cpp_dec_float_custom;
typedef boost::multiprecision::cpp_dec_float_100 cpp_dec_float_custom;

#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        fmt::print("Call as: {} exportRate dataFilename outputFilename compressedWeightsLog10", argv[0]);
        return false;
    }

    uint32_t exportRate = std::stoi(argv[1]);

    std::string dataFilename = argv[2];
    std::string outFilename = argv[3];
    dataFilename += ".txt";

    bool compressedWeightsLog10 = (std::stoi(argv[4]) == 0) ? false : true;
    std::cout << "Compressed weights: " << compressedWeightsLog10 << std::endl;

    std::ifstream inDataFile(dataFilename);
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

        outputSS << outFilename;
        outputSS << "_ConvergencePathWeights";
        outputSS << "_";
        outputSS << i;
        outputSS << ".txt";
        outFilesBasic[i] = std::ofstream(outputSS.str());


        std::stringstream outputLnSS;
        outputLnSS << outFilename;
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