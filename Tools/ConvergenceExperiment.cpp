#include <boost\multiprecision\cpp_dec_float.hpp>

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
        fmt::print("Call as: {} exportRate dataFilename outputFilename seed", argv[0]);
        return false;
    }

    uint32_t exportRate = std::stoi(argv[1]);

    std::string dataFilename = argv[2];
    std::string outFilename = argv[3];
    uint32_t seed = std::stoi(argv[4]);
    if (seed == 0)
    {
        seed = time(0);
    }
    dataFilename += ".txt";

    std::ifstream inDataFile(dataFilename);
    uint32_t numWeightValues = 0;
    inDataFile >> numWeightValues;

    std::cout << "Number of weight values: " << numWeightValues << std::endl;




    const uint32_t numSplits = 10;
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


        // Read em all in
        boost::multiprecision::cpp_dec_float_100 runningPathSum = 0.0;

        boost::multiprecision::cpp_dec_float_100 previousVal = 0.0;
        boost::multiprecision::cpp_dec_float_100 previousAvg = 1.0;

        for (uint32_t weightIdx = 0; weightIdx < numPathsPerSplit; ++weightIdx)
        {
            boost::multiprecision::cpp_dec_float_100 currentVal = 0.0;
            inDataFile >> currentVal;

            runningPathSum += currentVal;
            boost::multiprecision::cpp_dec_float_100 avg = runningPathSum / (boost::multiprecision::cpp_dec_float_100)(weightIdx + 1);

            //double threshold = 1.2;
            //if ((avg / previousAvg) > threshold)
            //{
            //    std::cout << "i: " << i << std::endl;
            //    std::cout << "Previous Avg: " << previousAvg << std::endl;
            //    std::cout << "Current Avg: " << avg << std::endl;
            //    std::cout << "Previous Val: " << previousVal << std::endl;
            //    std::cout << "Current Val: " << currentVal << std::endl;
            //}

            previousVal = currentVal;
            previousAvg = avg;

            if ((weightIdx % exportRate) == 0)
            {
                outFilesBasic[splitIdx] << avg << std::endl;
                outFilesLn[splitIdx] << boost::multiprecision::log(avg) << std::endl;
            }
        }
    }
}