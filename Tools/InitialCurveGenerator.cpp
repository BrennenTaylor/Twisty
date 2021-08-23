#include <FMath/FMath.h>

#include <CurveUtils.h>
#include <GeometryBootstrapper.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

int main(int argc, char* argv[])
{
    const uint32_t numInitialArguments = 5;

    if (argc < numInitialArguments)
    {
        std::cout << "Please call as: " << argv[0] << " directoryName numSegmentsPerCurve minArclength numCurvesToGen [randomSeedsHere]" << std::endl;
        return 1;
    }

    // Allow specification of min and max arc lengths

    const std::string directoryName = argv[1];
    const uint32_t numSegments = std::stoi(argv[2]);
    const float arclength = std::stof(argv[3]);
    const uint32_t numCurves = std::stoi(argv[4]);

    if (argc < (numInitialArguments + numCurves))
    {
        std::cout << "Not enough seeds provided" << std::endl;
        return 1;
    }

    const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    twisty::RayGeometry rayEmitter(emitterStart, emitterDir);
    const Farlor::Vector3 recieverPos{ 10.0f, 0.0f, 0.0f };
    const Farlor::Vector3 recieverDir{ 1.0, 0.0f, 0.0f };
    twisty::RayGeometry rayReciever(recieverPos, recieverDir);

    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path outputDirectoryPath = currentPath / directoryName;
    if (!std::filesystem::exists(outputDirectoryPath))
    {
        std::filesystem::create_directories(outputDirectoryPath);
    }

    const std::string CurveListFilename = "CurveList.txt";
    std::filesystem::path curveFile = outputDirectoryPath / CurveListFilename;

    std::ofstream curveListStream(curveFile);

    
    for (uint32_t curveIdx = 0; curveIdx < numCurves; ++curveIdx)
    {
        const uint32_t bootstrapperSeed = std::stoi(argv[numInitialArguments + curveIdx]);

        twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, arclength, bootstrapperSeed);

        std::unique_ptr<twisty::Curve> upInitialCurve = nullptr;
        bool successfulGen = false;
        while (!successfulGen)
        {
            upInitialCurve = bootstrapper.CreateCurve(numSegments);
            if (!upInitialCurve)
            {
                printf("Failed to create bootstrap curve.\n");
                return false;
            }

            // Lets also get the error of the initial curve, just to know
            float curveError = twisty::CurveUtils::CalculateCurveError(*upInitialCurve);
            std::cout << "Seed curve error: " << curveError << std::endl;

            const float maximumBootstrapCurveError = 0.1f;
            if (curveError < maximumBootstrapCurveError)
            {
                successfulGen = true;
            }
        }

        std::stringstream curveFilenameSS;
        curveFilenameSS << "InitialCurve";
        curveFilenameSS << "_" << numSegments;
        curveFilenameSS << "_" << bootstrapperSeed;
        curveFilenameSS << ".tpb";

        curveListStream << curveFilenameSS.str() << std::endl;

        std::filesystem::path curveFile = outputDirectoryPath / curveFilenameSS.str();

        std::ofstream outputFile(curveFile, std::ios::binary);
        twisty::Curve::WriteCurveToStream(outputFile, *upInitialCurve);
        outputFile.close();
    }

    curveListStream.close();
}