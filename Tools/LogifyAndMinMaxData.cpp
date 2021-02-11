#include <boost\multiprecision\cpp_dec_float.hpp>

#include <fstream>
#include <ostream>
#include <string>
#include <iostream>
#include <vector>

// Actually performs min max and log versions

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Call as: %s numValues inFilename outFilename", argv[0]);
        return 0;
    }

    uint32_t numValues = std::stoi(argv[1]);
    std::string inFilename(argv[2]);
    std::string outfilename(argv[3]);

    std::string minMaxOutFilename(outfilename + std::string("_minmax.dat"));
    std::string logOutFilename(outfilename + std::string("_log.dat"));

    std::ifstream inputDataStream(inFilename);

    // Then, we write out the animation file
    // This could be pulled out to a utility function, but keeping it here is fine for now.
    std::ofstream minMaxDatastream(minMaxOutFilename);
    std::ofstream logDatastream(logOutFilename);

    std::vector< boost::multiprecision::cpp_dec_float_100> dataValues(numValues);

    for (uint32_t idx = 0; idx < numValues; ++idx)
    {
        dataValues[idx] = 0.0;
        inputDataStream >> dataValues[idx];
    }

    boost::multiprecision::cpp_dec_float_100 minVal = dataValues[0];
    boost::multiprecision::cpp_dec_float_100 maxVal = dataValues[0];
    for (uint32_t idx = 0; idx < numValues; ++idx)
    {
        if (dataValues[idx] < minVal)
        {
            minVal = dataValues[idx];
        }

        if (dataValues[idx] > maxVal)
        {
            maxVal = dataValues[idx];
        }
    }

    std::vector< boost::multiprecision::cpp_dec_float_100> logdataValues(numValues);
    for (uint32_t idx = 0; idx < numValues; ++idx)
    {
        minMaxDatastream << (dataValues[idx] - minVal) / (maxVal - minVal) << std::endl;
        logDatastream << boost::multiprecision::log(dataValues[idx]) << std::endl;
    }

    // boost::multiprecision::cpp_dec_float_100 minLogVal = logdataValues[0];
    // for (uint32_t idx = 0; idx < numValues; ++idx)
    // {
    //     if (logdataValues[idx] < minLogVal)
    //     {
    //         minLogVal = logdataValues[idx];
    //     }
    // }

    // std::cout << "Min Log Val: " << minLogVal << std::endl;

    // for (uint32_t idx = 0; idx < numValues; ++idx)
    // {
    //     if (minLogVal < 0)
    //     {
    //         logDatastream << (logdataValues[idx] + boost::multiprecision::abs(minLogVal)) << std::endl;
    //     }
    //     else
    //     {
    //         logDatastream << (logdataValues[idx] - minLogVal) << std::endl;
    //     }
    // }

    std::cout << "Done" << std::endl;

    return 0;
}