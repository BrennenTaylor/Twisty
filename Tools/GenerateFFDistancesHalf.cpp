#include <boost/multiprecision/cpp_dec_float.hpp>

#include <FMath/FMath.h>

#include <fstream>
#include <ostream>
#include <string>
#include <iostream>
#include <vector>

// Actually performs min max and log versions

const float distanceFromPlane = 10.0;
const float frameLength = 10.0;

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Call as: %s imgWidth imgHeight outFilename", argv[0]);
        return 0;
    }

    uint32_t imgWidth = std::stoi(argv[1]);
    uint32_t imgHeight = std::stoi(argv[2]);
    std::string outfilename(argv[3]);

    std::string distancesFilename(outfilename + std::string("_distances.dat"));

    // Then, we write out the animation file
    // This could be pulled out to a utility function, but keeping it here is fine for now.
    std::ofstream distancesDataStream(distancesFilename);

    Farlor::Vector3 center(distanceFromPlane, 0.0f, 0.0f);
    Farlor::Vector3 bottomLeft = center - Farlor::Vector3(0.0f, frameLength / 2.0f, frameLength / 2.0f);

    for (uint32_t pixelIdxX = 0; pixelIdxX < ceil(imgWidth / 2.0); ++pixelIdxX)
    {
        for (uint32_t pixelIdxY = 0; pixelIdxY < ceil(imgHeight / 2.0); ++pixelIdxY)
        {
            const Farlor::Vector3 recieverPos = bottomLeft + Farlor::Vector3(0.0f, pixelIdxY * (frameLength / imgWidth), pixelIdxX * (frameLength / imgWidth))
                + Farlor::Vector3(0.0f, (frameLength / imgWidth) / 2.0f, (frameLength / imgWidth) / 2.0f);
            float distance = (recieverPos - center).Magnitude();
            distancesDataStream << distance << " ";
        }
        distancesDataStream << std::endl;
    }

    std::cout << "Done" << std::endl;

    return 0;
}