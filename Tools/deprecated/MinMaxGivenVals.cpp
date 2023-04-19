#include <boost/multiprecision/cpp_dec_float.hpp>

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
        printf("Call as: %s imgWidth imgHeight numValues inFilename outFilename", argv[0]);
        return 0;
    }

    uint32_t imgWidth = std::stoi(argv[1]);
    uint32_t imgHeight = std::stoi(argv[2]);

    std::string inFilename(argv[3]);

    std::string outfilename(argv[4]);

    std::string minMaxOutFilename(outfilename + std::string("_minmax_givenVals.dat"));

    std::ifstream inputDataStream(inFilename);

    // Then, we write out the animation file
    // This could be pulled out to a utility function, but keeping it here is fine for now.
    std::ofstream minMaxDatastream(minMaxOutFilename);

    boost::multiprecision::cpp_dec_float_100 minVal = 0.0;
    boost::multiprecision::cpp_dec_float_100 maxVal = 0.0;

    inputDataStream >> minVal;
    inputDataStream >> maxVal;

    for (int j = 0; j < 2; j++)
    {
        std::vector< boost::multiprecision::cpp_dec_float_100> pixelData(imgWidth * imgHeight);

        for (uint32_t x = 0; x < imgWidth; ++x)
        {
            for (uint32_t y = 0; y < imgHeight; ++y)
            {
                pixelData[x + y * imgWidth] = 0.0;
                inputDataStream >> pixelData[x + y * imgWidth];
            }
        }

        // boost::multiprecision::cpp_dec_float_100 minVal = pixelData[0];
        // boost::multiprecision::cpp_dec_float_100 maxVal = pixelData[0];
        // for (uint32_t x = 0; x < imgWidth; ++x)
        // {
        //     for (uint32_t y = 0; y < imgHeight; ++y)
        //     {
        //         if (pixelData[x + y * imgWidth] < minVal)
        //         {
        //             minVal = pixelData[x + y * imgWidth];
        //         }

        //         if (pixelData[x + y * imgWidth] > maxVal)
        //         {
        //             maxVal = pixelData[x + y * imgWidth];
        //         }
        //     }
        // }

        std::vector< boost::multiprecision::cpp_dec_float_100> logPixelData(imgWidth * imgHeight);
        for (uint32_t x = 0; x < imgWidth; ++x)
        {
            for (uint32_t y = 0; y < imgHeight; ++y)
            {
                minMaxDatastream << (pixelData[x + y * imgWidth] - minVal) / (maxVal - minVal) << " ";
            }
            minMaxDatastream << std::endl;
        }
    }

    std::cout << "Done" << std::endl;

    return 0;
}