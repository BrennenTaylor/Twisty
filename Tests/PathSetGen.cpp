#include "GeometryBootstrapper.h"
#include "Geometry.h"
#include "Range.h"
#include "MathConsts.h"

#include <fstream>

using namespace twisty;

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cout << "Please use as " << argv[0] << " numPathToGen numSegmentsPerPath" << std::endl;
        return 1;
    }

    const uint32_t numPathsToGenerate = std::stoi(argv[1]);
    const uint32_t numSegmentsPerPath = std::stoi(argv[2]);

    std::ofstream pathsFile("PathSet.txt", std::ios::out);

    pathsFile << numPathsToGenerate << ", " << numSegmentsPerPath << std::endl;
    for (uint32_t i = 0; i < numPathsToGenerate; ++i)
    {
        // Bootstrap method
        const Range defaultBounds = {-1.0f, 1.0f};

        const Farlor::Vector3 emitterPos{ 0.0f, 0.0f, 0.0f };
        const float emitterRadius = 1.0f;
        const float emitterFOV = TwistyPi;
        SphereGeometry sphereEmitter(emitterPos, emitterRadius, emitterFOV);

        const Farlor::Vector3 recieverPos{0.0f, 10.0f, 0.0f};
        const float recieverRadius = 1.0f;
        const float recieverFov = TwistyPi;
        SphereGeometry sphereReciever(recieverPos, recieverRadius, recieverFov);

        const Range arclengthRange = {10.0f, 30.0f};

        GeometryBootstrapper bootstrapper(sphereEmitter, sphereReciever, arclengthRange, 0);

        auto upCurve = bootstrapper.CreateCurve(numSegmentsPerPath);
        uint32_t count = 0;
        for (auto& segment : upCurve->m_segments)
        {
            pathsFile << segment.m_weightEvalPos;
            if (count < numSegmentsPerPath - 1)
            {
                pathsFile << ", ";
            }
            count++;
        }
        pathsFile << std::endl;
    }

    pathsFile.close();
}