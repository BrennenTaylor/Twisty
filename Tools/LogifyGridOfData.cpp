#include <fmt/format.h>

#include <boost\multiprecision\cpp_dec_float.hpp>

#include <fstream>
#include <ostream>
#include <string>
#include <iostream>
#include <vector>

// Actually performs min max and log versions

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        fmt::print("Call as: {} imgWidth imgHeight inFilename outFilename", argv[0]);
        return 0;
    }

    uint32_t imgWidth = std::stoi(argv[1]);
    uint32_t imgHeight = std::stoi(argv[2]);

    std::string inFilename(argv[3]);

    std::string outfilename(argv[4]);

    std::string minMaxOutFilename(outfilename + std::string("_minmax.dat"));
    std::string logOutFilename(outfilename + std::string("_log.dat"));

    std::ifstream inputDataStream(inFilename);

    // Then, we write out the animation file
    // This could be pulled out to a utility function, but keeping it here is fine for now.
    std::ofstream minMaxDatastream(minMaxOutFilename);
    std::ofstream logDatastream(logOutFilename);

    std::vector< boost::multiprecision::cpp_dec_float_100> pixelData(imgWidth * imgHeight);

    for (uint32_t x = 0; x < imgWidth; ++x)
    {
        for (uint32_t y = 0; y < imgHeight; ++y)
        {
            pixelData[x + y * imgWidth] = 0.0;
            inputDataStream >> pixelData[x + y * imgWidth];
        }
    }

    boost::multiprecision::cpp_dec_float_100 minVal = pixelData[0];
    boost::multiprecision::cpp_dec_float_100 maxVal = pixelData[0];
    for (uint32_t x = 0; x < imgWidth; ++x)
    {
        for (uint32_t y = 0; y < imgHeight; ++y)
        {
            if (pixelData[x + y * imgWidth] < minVal)
            {
                minVal = pixelData[x + y * imgWidth];
            }

            if (pixelData[x + y * imgWidth] > maxVal)
            {
                maxVal = pixelData[x + y * imgWidth];
            }
        }
    }

    std::vector< boost::multiprecision::cpp_dec_float_100> logPixelData(imgWidth * imgHeight);
    for (uint32_t x = 0; x < imgWidth; ++x)
    {
        for (uint32_t y = 0; y < imgHeight; ++y)
        {
            minMaxDatastream << (pixelData[x + y * imgWidth] - minVal) / (maxVal - minVal) << " ";
            logPixelData[x + y * imgWidth] = boost::multiprecision::log(pixelData[x + y * imgWidth]);
        }
        minMaxDatastream << std::endl;
    }

    boost::multiprecision::cpp_dec_float_100 minLogVal = logPixelData[0];
    for (uint32_t x = 0; x < imgWidth; ++x)
    {
        for (uint32_t y = 0; y < imgHeight; ++y)
        {
            if (logPixelData[x + y * imgWidth] < minLogVal)
            {
                minLogVal = logPixelData[x + y * imgWidth];
            }
        }
    }

    std::cout << "Min Log Val: " << minLogVal << std::endl;

    for (uint32_t x = 0; x < imgWidth; ++x)
    {
        for (uint32_t y = 0; y < imgHeight; ++y)
        {

            //logDatastream << (logPixelData[x + y * imgWidth]) << " ";

            if (minLogVal < 0)
            {
                logDatastream << (logPixelData[x + y * imgWidth] + boost::multiprecision::abs(minLogVal)) << " ";
            }
            else
            {
                logDatastream << (logPixelData[x + y * imgWidth] - minLogVal) << " ";
            }
        }
        logDatastream << std::endl;
    }

    std::cout << "Done" << std::endl;

    return 0;
}