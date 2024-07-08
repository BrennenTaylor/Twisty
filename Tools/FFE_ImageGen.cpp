#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <ExperimentUtils.h>

#define DEBUG_PRINT 0

int main(int argc, char *argv[])
{
    if (argc < 6) {
        printf("Call as: %s imgWidth imgHeight scaleF inFilename outFilename", argv[0]);
        return 0;
    }

    uint32_t imgWidth = std::stoi(argv[1]);
    uint32_t imgHeight = std::stoi(argv[2]);
    float imgScaleFactor = std::stof(argv[3]);

    std::string inFilename(argv[4]);
    std::string outFilename(argv[5]);
    outFilename += ".exr";
    std::ifstream inputDataStream(inFilename);

    // Then, we write out the animation file
    // This could be pulled out to a utility function, but keeping it here is fine
    // for now.

    int readWidth = 0;
    inputDataStream >> readWidth;
    int readHeight = 0;
    inputDataStream >> readHeight;
    std::cout << "read width: " << readWidth << std::endl;
    std::cout << "read height: " << readHeight << std::endl;

    std::vector<float> values;
    values.reserve(readWidth * readHeight);

    for (uint32_t y = 0; y < imgHeight; ++y) {
        for (uint32_t x = 0; x < imgWidth; ++x) {
            float dataValue = 0.0f;
            inputDataStream >> dataValue;
            values.push_back(dataValue);
#if DEBUG_PRINT
            printf("(%d, %d): %f\n", x, y, dataValue);
#endif
        }
    }

    if (SaveEXR(values, readWidth, readHeight, imgScaleFactor, outFilename.c_str())) {
        std::cout << "Failed to export" << std::endl;
        return 1;
    }

    std::cout << "Done" << std::endl;

    return 0;
}