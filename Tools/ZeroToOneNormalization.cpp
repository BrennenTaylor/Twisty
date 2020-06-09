#include <fmt/format.h>

#include <boost\multiprecision\cpp_dec_float.hpp>

#include <fstream>
#include <ostream>
#include <string>
#include <iostream>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        fmt::print("Call as: {} dataFilename", argv[0]);
        return 0;
    }

    std::string inFilename(argv[1]);
    std::ifstream inFile(inFilename);

    uint32_t numValues = 0;
    inFile >> numValues;

    std::vector<boost::multiprecision::cpp_dec_float_100> readInValues(numValues);
    for (uint32_t i = 0; i < numValues; ++i)
    {
        inFile >> readInValues[i];
    }

    boost::multiprecision::cpp_dec_float_100 sum = 0.0;

    for (uint32_t i = 0; i < numValues; ++i)
    {
        sum += readInValues[i];
    }
    boost::multiprecision::cpp_dec_float_100 avg = sum /= numValues;

    // Min max normalization
    for (uint32_t i = 0; i < numValues; ++i)
    {
        readInValues[i] += avg;
    }

#if 0
    boost::multiprecision::cpp_dec_float_100 min = readInValues[0];
    boost::multiprecision::cpp_dec_float_100 max = readInValues[0];
    uint32_t minIdx = 0;
    uint32_t maxIdx = 0;

    for (uint32_t i = 1; i < numValues; ++i)
    {
        if (readInValues[i] < min)
        {
            min = readInValues[i];
            minIdx = i;
        }

        if (readInValues[i] > max)
        {
            max = readInValues[i];
            maxIdx = i;
        }
    }

    std::cout << "MinIdx: " << minIdx << std::endl;
    std::cout << "MaxIdx: " << maxIdx << std::endl;
    std::cout << "MinVal: " << min << std::endl;
    std::cout << "MaxVal: " << max << std::endl;

    // Min max normalization
    for (uint32_t i = 0; i < numValues; ++i)
    {
        boost::multiprecision::cpp_dec_float_100 initialValue = readInValues[i];

        readInValues[i] = (initialValue - min) / (max - min);
    }
#endif

    std::string outFilename = inFilename;
    outFilename += "_outfile.txt";
    std::ofstream outFile(outFilename);

    outFile << numValues << std::endl;
    for (uint32_t i = 0; i < numValues; ++i)
    {
        outFile << readInValues[i] << std::endl;
        //outFile << min << std::endl;
        //outFile << max << std::endl;
    }
}