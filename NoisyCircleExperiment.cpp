#include "FullExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "FullExperimentRunnerOptimalPerturbOptimized.h"

//#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
//#include "GpuFullExperimentRunnerGeneral.h"
//#endif

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

//#define DetectBackScatter

//#define ExportGeometryInfo
//#define ExportPixelCenters
//#define ExportNormals
//#define ExportArclengths

// Ok, we want to kick off an experiment per pixel.
const uint32_t numDirections = 1;
const uint32_t numArclengths = 1;
const float distanceFromPlane = 10.0f;
//
//Farlor::Vector3 UniformlySampleHemisphereFacingNegativeX(const float u, const float v)
//{
//    float z = u;
//    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
//    float phi = 2.0f * twisty::TwistyPi * v;
//    
//#ifdef DetectBackScatter
//    return Farlor::Vector3(-z, r * std::cos(phi), r * std::sin(phi)).Normalized();
//#else
//    return Farlor::Vector3(z, r * std::cos(phi), r * std::sin(phi)).Normalized();
//#endif
//}

int main(int argc, char *argv[])
{
    if (argc < 13)
    {
        std::cout << "Call as: " << argv[0] << " numPathsPerInternalExperiment numPathsToSkipPerInternalExperiment experimentName bootstrapperSeed perturbSeed normalGenSeed arclengthGenSeed startX startY startArclength frameLength framePixelCount numInitialCurves numPerInitialCurve" << std::endl;
        return 1;
    }

    const uint32_t numPathsToGenerate = std::stoi(argv[1]);
    const uint32_t numPathsToSkip = std::stoi(argv[2]);
    const std::string experimentName(argv[3]);
    int bootstrapperSeed = std::stoi(argv[4]);
    int perturbSeed = std::stoi(argv[5]);
    int normalGenSeed = std::stoi(argv[6]);
    int arclengthGenSeed = std::stoi(argv[7]);
    normalGenSeed = (normalGenSeed != 0) ? normalGenSeed : time(0);

    int startX = std::stoi(argv[8]);
    int startY = std::stoi(argv[9]);
    int startArclength = std::stoi(argv[10]);

    const int frameLength = std::stoi(argv[11]);
    const uint32_t framePixelCount = std::stoi(argv[12]);

    const uint32_t numInitialCurves = std::stoi(argv[13]);
    const uint32_t numPerInitialCurve = std::stoi(argv[14]);

    assert(startX < framePixelCount);
    assert(startY < framePixelCount);

    std::filesystem::path outputDirectoryPath = std::filesystem::current_path();
    outputDirectoryPath.append(experimentName);
    if (!std::filesystem::exists(outputDirectoryPath))
    {
        std::filesystem::create_directory(outputDirectoryPath);
    }

#ifdef ExportGeometryInfo
    std::filesystem::path geometryExportPath = outputDirectoryPath;
    geometryExportPath.append("GeometryData.ffg");

    std::ofstream geometryExportStream(geometryExportPath.string());
    if (!geometryExportStream.is_open())
    {
        std::cout << "Failed to geometry export outfile" << std::endl;
        exit(1);
    }
#endif

    // Lets test uniform random directions on sampling hemisphere
    std::mt19937 normalGenerator(normalGenSeed);
    std::uniform_real_distribution<float> uniformFloat(0.0f, 1.0f);
    // We have a rotated and non-rotated version to test for initial seed curve impact
    std::vector<boost::multiprecision::cpp_dec_float_100> nonRotatedPixels(framePixelCount * framePixelCount);
    for (uint32_t r = 0; r < framePixelCount; r++)
    {
        for (uint32_t c = 0; c < framePixelCount; c++)
        {
            const uint32_t frameIdx = r * framePixelCount + c;
            nonRotatedPixels[frameIdx] = 0.0;
        }
    }

    Farlor::Vector3 center(distanceFromPlane, 0.0f, 0.0f);
    Farlor::Vector3 bottomLeft = center - Farlor::Vector3(0.0f, frameLength / 2.0f, frameLength / 2.0f);

    // Bootstrap method
    const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    twisty::RayGeometry rayEmitter(emitterStart, emitterDir);

#ifdef ExportGeometryInfo
    // Export num x and y pixels
    geometryExportStream << framePixelCount << " " << framePixelCount << std::endl;
    // Then num directions and arclenghts
    geometryExportStream << numDirections << " " << numArclengths << std::endl;
    geometryExportStream << center.x << " " << center.y << " " << center.z << std::endl;
    geometryExportStream << bottomLeft.x << " " << bottomLeft.y << " " << bottomLeft.z << std::endl;
    geometryExportStream << emitterStart.x << " " << emitterStart.y << " " << emitterStart.z << std::endl;
    geometryExportStream << emitterDir.x << " " << emitterDir.y << " " << emitterDir.z << std::endl;
#endif

    bool IsFirst = true;

    for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX)
    {
        if (pixelIdxX != 3)
        {
            continue;
        }

        for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY)
        {
            if (pixelIdxY != 3)
            {
                continue;
            }

            // if (IsFirst)
            // {
            //     pixelIdxX = startX;
            //     pixelIdxY = startY;
            //     IsFirst = false;
            // }


            const Farlor::Vector3 recieverPos = bottomLeft + Farlor::Vector3(0.0f, pixelIdxY * (frameLength / framePixelCount), pixelIdxX * (frameLength / framePixelCount))
                + Farlor::Vector3(0.0f, (frameLength / framePixelCount) / 2.0f, (frameLength / framePixelCount) / 2.0f);




#if defined(ExportGeometryInfo)
            geometryExportStream << pixelIdxX << " " << pixelIdxY << std::endl;
#endif

#if defined(ExportGeometryInfo) && defined(ExportPixelCenters)
            geometryExportStream << recieverPos.x << " " << recieverPos.y << " " << recieverPos.z << std::endl;
#endif
        
            if (bootstrapperSeed == 0)
            {
                bootstrapperSeed = time(0);
            }

            if (perturbSeed == 0)
            {
                perturbSeed = time(0);
            }

            //for (uint32_t directionIdx = 0; directionIdx < numDirections; ++directionIdx)
            {
                //float u = uniformFloat(normalGenerator);
                //float v = uniformFloat(normalGenerator);
                //const Farlor::Vector3 recieverDir = UniformlySampleHemisphereFacingNegativeX(u, v);
                const Farlor::Vector3 recieverDir = (recieverPos - emitterStart).Normalized();




#if defined(ExportGeometryInfo) && defined(ExportNormals)
                geometryExportStream << "    " << recieverDir.x << " " << recieverDir.y << " " << recieverDir.z << std::endl;
#endif

                twisty::RayGeometry rayReciever(recieverPos, recieverDir);


                // Start at somehting close to 1.05 times the minimum arclength and go to 1.5 times the arclength

                const float minArclength = (recieverPos - emitterStart).Magnitude() * 1.05f;
                const float maxArclength = (recieverPos - emitterStart).Magnitude() * 1.5f;
                const float arclengthStepSize = (maxArclength - minArclength) / (numArclengths);

                for (uint32_t arclengthIdx = 0; arclengthIdx < numArclengths; ++arclengthIdx)
                {
                    std::stringstream innerBlockSS;
                    innerBlockSS << "<" << pixelIdxX << ", " << pixelIdxY << /*", " << directionIdx << ", " << arclengthIdx <<*/ ">";
                    std::cout << innerBlockSS.str() << std::endl;
                    
                    // Forces stratified arclength ranges
                    //double arclength = (recieverPos - emitterStart).Magnitude() * 1.25;
                    double targetArclength = minArclength + arclengthStepSize * arclengthIdx;
                    
                    boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;

                    std::mt19937 initialCurveGen(bootstrapperSeed);
                    for (uint32_t initialCurveIdx = 0; initialCurveIdx < numInitialCurves; ++initialCurveIdx)
                    {

                        int initialCurveSeed = initialCurveGen();
                        while (initialCurveSeed == 0)
                        {
                            initialCurveSeed = initialCurveGen();
                        }

                        boost::multiprecision::cpp_dec_float_100 maxResult = 0.0;

                        std::mt19937 perCurveGen(perturbSeed);
                        for (uint32_t perInitialCurveIdx = 0; perInitialCurveIdx < numPerInitialCurve; ++perInitialCurveIdx)
                        {
                            int perCurveSeed = perCurveGen();
                            while (perCurveSeed == 0)
                            {
                                perCurveSeed = perCurveGen();
                            }

                            twisty::ExperimentRunner::ExperimentParameters experimentParams;
                            experimentParams.numPathsInExperiment = numPathsToGenerate;
                            experimentParams.numPathsToSkip = numPathsToSkip;
                            experimentParams.exportGeneratedCurves = false;
                            experimentParams.experimentName = experimentName;
                            experimentParams.numSegmentsPerCurve = 200;
                            experimentParams.maximumBootstrapCurveError = 0.5f;
                            experimentParams.curvePurturbSeed = perCurveSeed;
                            experimentParams.rotateInitialSeedCurveRadians = 0.0f;

                            experimentParams.weightingParameters.mu = 0.1;
                            experimentParams.weightingParameters.eps = 0.1;
                            experimentParams.weightingParameters.numStepsInt = 2000;
                            experimentParams.weightingParameters.minBound = 0.0;
                            experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;
                            experimentParams.weightingParameters.numCurvatureSteps = 10000;
                            // Lets give some absorbtion as well
                            // Absorbtion 1/20 off the time
                            experimentParams.weightingParameters.absorbtion = 0.05;
                            // 1/5 scatter means one event every 5 units, thus 2 scattering events in the shortest
                            // or 5 in the longest 100 unit path
                            experimentParams.weightingParameters.scatter = 0.2;

                            
                            experimentParams.rotateInitialSeedCurveRadians = 0.0f;

                            twisty::GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, targetArclength, initialCurveSeed);
                            std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);
                            bool result = upExperimentRunner->Setup();

                            if (!result)
                            {
                                upExperimentRunner->Shutdown();
                                std::cout << "Failed to setup experiment runner." << std::endl;
                                return 1;
                            }

#if defined(ExportGeometryInfo) && defined(ExportArclengths)
                            twisty::Curve* pCurve = upExperimentRunner->GetInitialCurvePtr();
                            float arclength = pCurve->m_arclength;
                            geometryExportStream << "        " << arclength << std::endl;
#endif
                            twisty::ExperimentRunner::ExperimentResults results = upExperimentRunner->RunExperiment();
                            if (results.experimentWeight > maxResult)
                            {
                                maxResult = results.experimentWeight;
                            }
                            upExperimentRunner->Shutdown();
                        }

                        averagedResult += (maxResult * (1.0 / numInitialCurves));
                    }

                    const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
                    nonRotatedPixels[frameIdx] += averagedResult * (1.0 / (numArclengths));
                }
            }

            const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
            std::cout << "Non-Rotated Pixel Weight: " << nonRotatedPixels[frameIdx] << std::endl;
        }
    }

    // Export freeze frame pixel data
    {
        std::filesystem::path freezeFramePath = outputDirectoryPath;
        freezeFramePath.append("FreezeFrameNonRotated.dat");

        std::ofstream freezeFrameOutfile(freezeFramePath.string());
        if (!freezeFrameOutfile.is_open())
        {
            std::cout << "Failed to create non-rotated freezeframe outfile" << std::endl;
            exit(1);
        }

        // X
        freezeFrameOutfile << framePixelCount << " " << framePixelCount << std::endl;

        // Write out the pixel data
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX)
        {
            for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY)
            {
                // Output pixel
                const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
                freezeFrameOutfile << nonRotatedPixels[frameIdx] << " ";
            }
            freezeFrameOutfile << std::endl;
        }
    }

    return 0;
}
