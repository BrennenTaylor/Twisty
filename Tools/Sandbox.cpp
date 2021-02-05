#include <boost/multiprecision/cpp_dec_float.hpp>

#include <fstream>
#include <limits>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Use format: " << argv[0] << " filename" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];

    std::ifstream inFile(filename);
    if (!inFile.is_open())
    {
        std::cout << "Failed to open: " << filename << std::endl;
        return 1;
    }
    
    uint32_t numWeights = 0;
    inFile >> numWeights;

    std::vector<boost::multiprecision::cpp_dec_float_100> weights(numWeights);
    for (uint32_t i = 0; i < numWeights; ++i)
    {
        inFile >> weights[i];
    }

    std::vector<double> log10Weights(numWeights);
    for (uint32_t i = 0; i < numWeights; ++i)
    {
        boost::multiprecision::cpp_dec_float_100 log10Weight = boost::multiprecision::log10(weights[i]);
        log10Weights[i] = log10Weight.convert_to<double>();
    }

    boost::multiprecision::cpp_dec_float_100 totalWeight = 0.0;
    for (uint32_t weightIdx = 0; weightIdx < numWeights; ++weightIdx)
    {
        totalWeight += weights[weightIdx];
    }
    std::cout << "Total weight: " << totalWeight << std::endl;

    boost::multiprecision::cpp_dec_float_100 maxWeight = weights[0];
    for (uint32_t weightIdx = 1; weightIdx < numWeights; ++weightIdx)
    {
        if (weights[weightIdx] > maxWeight)
        {
            maxWeight = weights[weightIdx];
        }
    }
    std::cout << "Max weight: " << maxWeight << std::endl;

    boost::multiprecision::cpp_dec_float_100 maxPossibleWeight = maxWeight * numWeights;
    std::cout << "Max possible weight: " << maxPossibleWeight << std::endl;

    boost::multiprecision::cpp_dec_float_100 log10MaxWeight = boost::multiprecision::log10(maxWeight);
    std::cout << "Max weight log10: " << log10MaxWeight << std::endl;

    double log10MaxPossibleWeight = boost::multiprecision::log10(maxPossibleWeight).convert_to<double>();
    std::cout << "Max possible weight log10: " << log10MaxPossibleWeight << std::endl;

    double maxWeightDouble = maxWeight.convert_to<double>();
    std::cout << "Max weight double: " << maxWeightDouble << std::endl;

    double maxDouble = std::numeric_limits<double>::max();
    double log10MaxDouble = std::log10(maxDouble);

    std::cout << "Max double: " << maxDouble << std::endl;
    std::cout << "Max double log10: " << log10MaxDouble << std::endl;

    double difference = 300 - log10MaxPossibleWeight;
    boost::multiprecision::cpp_dec_float_100 bigFloatDifference = difference;
    for (uint32_t weightIdx = 0; weightIdx < numWeights; ++weightIdx)
    {
        log10Weights[weightIdx] += difference;
    }

    double totalWeightDouble = 0.0;
    for (uint32_t weightIdx = 0; weightIdx < numWeights; ++weightIdx)
    {
        double decompressedWeight = std::pow(10, log10Weights[weightIdx]);
        totalWeightDouble += decompressedWeight;
    }

    std::cout << "Total weight double: " << totalWeightDouble << std::endl;

    boost::multiprecision::cpp_dec_float_100 bigFloatTotalWeight = totalWeightDouble;
    boost::multiprecision::cpp_dec_float_100 log10BigFloatWeight = boost::multiprecision::log10(bigFloatTotalWeight);
    boost::multiprecision::cpp_dec_float_100 adjustedLog10BigFloatWeight = log10BigFloatWeight - bigFloatDifference;
    boost::multiprecision::cpp_dec_float_100 finalWeight = boost::multiprecision::pow(10, adjustedLog10BigFloatWeight);

    std::cout << "Final Weight: " << finalWeight << std::endl;
}