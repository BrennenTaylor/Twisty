#include <ExperimentRunner.h>
#include <PathWeighters.h>

#include "MathConsts.h"

#include <FMath/FMath.h>

#include <nlohmann/json.hpp>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Call as: " << argv[0] << " pathsDirectory" << std::endl;
        return 1;
    }

    std::string pathsDirectory(argv[1]);

    std::filesystem::path pathsDirectoryPath = pathsDirectory;
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    // All valid directories must have an FiveSegment_Binary_FixedOrder.bdt
    std::filesystem::path fiveSegmentPath = pathsDirectoryPath;
    fiveSegmentPath.append("FiveSegment_Binary_FixedOrder.bdt");

    if (!std::filesystem::exists(fiveSegmentPath)) {
        std::cout << fiveSegmentPath << " file does not exist, provide one" << std::endl;
        return 1;
    }

    std::ifstream fiveSegmentFS(fiveSegmentPath, std::ios::binary);
    if (!fiveSegmentFS.is_open()) {
        std::cout << "Failed to open: " << fiveSegmentPath << std::endl;
        return 1;
    }

    fiveSegmentFS.seekg(0, fiveSegmentFS.end);
    uint64_t numBytesInFile = fiveSegmentFS.tellg();
    fiveSegmentFS.seekg(0, fiveSegmentFS.beg);

    uint64_t numBytesInCurve = sizeof(Farlor::Vector3);
    assert((numBytesInFile % numBytesInCurve) == 0
          && "File should exactly fit three values per curve");
    uint64_t numCurves = numBytesInFile / numBytesInCurve;

    std::vector<Farlor::Vector3> fiveSegmentAngleValues(numCurves);
    fiveSegmentFS.read((char *)fiveSegmentAngleValues.data(), sizeof(Farlor::Vector3) * numCurves);

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = twisty::TwistyPi;
    const uint32_t numPhi1Vals = 100;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const uint32_t numTheta1Vals = 1024;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    // Azimuthal
    const float theta2Min = -twisty::TwistyPi;
    const float theta2Max = twisty::TwistyPi;
    const uint32_t numTheta2Vals = 1024;
    const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

    std::vector<std::vector<uint64_t>> gridOfImages(numPhi1Vals);

    for (auto &image : gridOfImages) {
        image.resize(numTheta1Vals * numTheta2Vals);
    }


    uint64_t maxValue = 0;

    // Hash out the values
    for (const auto &curveAngles : fiveSegmentAngleValues) {
        const float phi1 = curveAngles.x;
        const float theta1 = curveAngles.y;
        const float theta2 = curveAngles.z;

        const float phiDist = (phi1 - phi1Min);
        uint32_t phiIdx = phiDist / dPhi1;
        phiIdx = std::min(phiIdx, numPhi1Vals - 1);

        const float theta1Dist = (theta1 - theta1Min);
        uint32_t theta1Idx = theta1Dist / dTheta1;
        theta1Idx = std::min(theta1Idx, numTheta1Vals - 1);

        const float theta2Dist = (theta2 - theta2Min);
        uint32_t theta2Idx = theta2Dist / dTheta2;
        theta2Idx = std::min(theta2Idx, numTheta2Vals - 1);

        //   std::cout << "Phi Idx: " << phiIdx << std::endl;
        //   std::cout << "Theta1 Idx: " << theta1Idx << std::endl;
        //   std::cout << "Theta2 Idx: " << theta2Idx << std::endl;

        // Increament Heatmap
        gridOfImages[phiIdx][theta1Idx * numTheta2Vals + theta2Idx]++;
        maxValue = std::max(maxValue, gridOfImages[phiIdx][theta1Idx * numTheta2Vals + theta2Idx]);
    }

    std::cout << "max value: " << maxValue << std::endl;


    for (int phi1Idx = 0; phi1Idx < numPhi1Vals; phi1Idx++) {
        std::vector<uint64_t> &currentImage = gridOfImages.at(phi1Idx);

        const std::string imageFilename
              = pathsDirectoryPath.string() + "/" + std::to_string(phi1Idx) + ".png";

        const int comp = 3;  // RGB
        std::vector<uint8_t> actualPixels(numTheta1Vals * numTheta2Vals * comp);
        for (uint32_t pixelIdx = 0; pixelIdx < actualPixels.size(); pixelIdx += comp) {
            uint32_t vectorPixelIdx = pixelIdx / 3;

            actualPixels[pixelIdx] = (currentImage[vectorPixelIdx] != 0) ? (255 / 4) : 0;

            if (actualPixels[pixelIdx] != 0) {
                actualPixels[pixelIdx + 1]
                      = static_cast<uint8_t>(static_cast<double>(currentImage[vectorPixelIdx])
                            / static_cast<double>(maxValue) * 255.0f);
                actualPixels[pixelIdx + 2] = 0;
            }
        }

        int errorCode = stbi_write_bmp(
              imageFilename.c_str(), numTheta1Vals, numTheta2Vals, comp, actualPixels.data());
        if (errorCode == 0) {
            std::cout << "Error out for some reason" << std::endl;
        }
    }
    std::cout << "Done" << std::endl;
}