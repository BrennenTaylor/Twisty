#include <boost/multiprecision/cpp_dec_float.hpp>

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
        printf("Call as: %s sliceSize numSlices dataFilename seed", argv[0]);
        return false;
    }

    const uint32_t sliceSize = std::stoi(argv[1]);
    const uint32_t numSlices = std::stoi(argv[2]);
    std::string dataFilename = argv[3];
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
    std::cout << "Slice Size: " << sliceSize << std::endl;
    std::cout << "Number of slices: " << numSlices << std::endl;
    std::cout << "Lower idx: " << 0 << std::endl;
    std::cout << "Greater idx: " << numWeightValues - 1 - sliceSize << std::endl;

    std::vector<boost::multiprecision::cpp_dec_float_100> pathWeights(numWeightValues);
    for (uint32_t i = 0; i < numWeightValues; ++i)
    {
        inDataFile >> pathWeights[i];
    }

    std::vector<boost::multiprecision::cpp_dec_float_100> sliceWeights(numSlices);
    std::mt19937 randomGen(seed);
    std::uniform_int_distribution<int> randomSliceStartDist(0, numWeightValues - 1 - sliceSize);

    // Do slice weight stuff
    for (uint32_t sliceIdx = 0; sliceIdx < numSlices; ++sliceIdx)
    {
        boost::multiprecision::cpp_dec_float_100 sliceWeight = 0.0;
        uint32_t sliceStartIdx = randomSliceStartDist(randomGen);

        for (uint32_t weightIdx = sliceStartIdx; weightIdx < sliceStartIdx + sliceSize; ++weightIdx)
        {
            sliceWeight += pathWeights[weightIdx];
        }
        sliceWeights[sliceIdx] = sliceWeight;
    }

    // Output Slice Weights
    std::stringstream outputSS;

    outputSS << argv[3];
    outputSS << "_Slices_";
    outputSS << sliceSize;
    outputSS << ".txt";
    std::ofstream outFile(outputSS.str());

    outFile << numSlices << std::endl;
    outFile << "Slice Weights" << std::endl;
    for (uint32_t i = 0; i < numSlices; i++)
    {
        outFile << sliceWeights[i] << std::endl;
    }

    outFile << "Loged Slice Weights" << std::endl;
    for (uint32_t i = 0; i < numSlices; i++)
    {
        outFile <<  boost::multiprecision::log(sliceWeights[i]) << std::endl;
    }
}