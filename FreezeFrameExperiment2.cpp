#include "FullExperimentRunner.h"
#include "FullExperimentRunner2.h"

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#include "GpuFullExperimentRunnerGeneral.h"
#endif

#include "GeometryBootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"

#include <FMath/Vector3.h>

#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

using namespace twisty;

//#define DoubleWithRotated
//#define DetectBackScatter

//#define ExportGeometryInfo
//#define ExportPixelCenters
//#define ExportNormals
//#define ExportArclengths

// Ok, we want to kick off an experiment per pixel.
//const uint32_t numDirections = 15;
const uint32_t numArclengths = 100;
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

ExperimentRunner::ExperimentResults CalculateDipoleTransportKernel(const GeometryBootstrapper& geomBootstrapper,
    const ExperimentRunner::ExperimentParameters& experimentParams,
    const double s)
{
    ExperimentRunner::ExperimentResults results;

    const double g = 1.0 - experimentParams.weightingParameters.mu;
    const double p = (1.0 - g) * experimentParams.weightingParameters.scatter / 2.0;

    const Farlor::Vector3 r = p * (geomBootstrapper.GetTargetPosition() - geomBootstrapper.GetStartPosition());

    const double A = 1.0 / (p * s - std::tanh(p * s));
    const double B = std::tanh(p * s) / 2;
    const double C = 9.0 / (2.0 * p * s);
    const double D = 3.0 * A * B * B - (3.0 / (2.0 * std::sinh(2.0 * p * s)));
    const double E = 3.0 * A * B;
    const double F = 3.0 * A / 2.0;
    
    const double N_Num = p * p * p * std::sqrt(F) * (E * E - 2.0 * D * F) * std::exp(C);
    const double N_Denom = 4.0 * std::pow(TwistyPi, (5.0 / 2.0)) * (std::exp(E * E / F - D) - std::exp(D));
    const double N = N_Num / N_Denom;

    const double result = N * std::exp((-1.0 * C) + (-1.0 * D * geomBootstrapper.GetTargetNormal().Dot(geomBootstrapper.GetStartNormal()))
        + E * r.Dot(geomBootstrapper.GetTargetNormal() + geomBootstrapper.GetStartNormal())
        - F * r.Magnitude() * r.Magnitude() - experimentParams.weightingParameters.absorbtion * s);

    results.experimentWeight = result;
    return results;
}


int main(int argc, char *argv[])
{
    if (argc < 11)
    {
        fmt::print(
            "Call as: {} numPathsPerPixel numPathsToSkip experimentName bootstrapperSeed perturbSeed normalGenSeed arclengthGenSeed startX startY startArclength frameLength framePixelCount\n",
            argv[0]);
        return 1;
    }

    const uint32_t numPathsToGenerate = std::stoi(argv[1]);
    const uint32_t numPathsToSkip = std::stoi(argv[2]);
    const std::string experimentName(argv[3]);
    const int boostrapperSeed = std::stoi(argv[4]);
    const int perturbSeed = std::stoi(argv[5]);
    int normalGenSeed = std::stoi(argv[6]);
    int arclengthGenSeed = std::stoi(argv[7]);
    normalGenSeed = (normalGenSeed != 0) ? normalGenSeed : time(0);

    int startX = std::stoi(argv[8]);
    int startY = std::stoi(argv[9]);
    int startArclength = std::stoi(argv[10]);

    const int frameLength = std::stoi(argv[11]);
    const uint32_t framePixelCount = std::stoi(argv[12]);

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


#ifdef DoubleWithRotated
    std::vector<boost::multiprecision::cpp_dec_float_100> rotatedPixels(framePixelCount * framePixelCount);
#endif
    for (uint32_t r = 0; r < framePixelCount; r++)
    {
        for (uint32_t c = 0; c < framePixelCount; c++)
        {
            const uint32_t frameIdx = r * framePixelCount + c;
            nonRotatedPixels[frameIdx] = 0.0f;
#ifdef DoubleWithRotated
            rotatedPixels[frameIdx] = 0.0f;
#endif
        }
    }

    Farlor::Vector3 center(distanceFromPlane, 0.0f, 0.0f);
    Farlor::Vector3 bottomLeft = center - Farlor::Vector3(0.0f, frameLength / 2.0f, frameLength / 2.0f);

    // Bootstrap method
    const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    RayGeometry rayEmitter(emitterStart, emitterDir);

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
        for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY)
        {
            if (IsFirst)
            {
                pixelIdxX = startX;
                pixelIdxY = startY;
                IsFirst = false;
            }


            const Farlor::Vector3 recieverPos = bottomLeft + Farlor::Vector3(0.0f, pixelIdxY * (frameLength / framePixelCount), pixelIdxX * (frameLength / framePixelCount))
                + Farlor::Vector3(0.0f, (frameLength / framePixelCount) / 2.0f, (frameLength / framePixelCount) / 2.0f);




#if defined(ExportGeometryInfo)
            geometryExportStream << pixelIdxX << " " << pixelIdxY << std::endl;
#endif

#if defined(ExportGeometryInfo) && defined(ExportPixelCenters)
            geometryExportStream << recieverPos.x << " " << recieverPos.y << " " << recieverPos.z << std::endl;
#endif
        
            //for (uint32_t directionIdx = 0; directionIdx < numDirections; ++directionIdx)
            {
                //float u = uniformFloat(normalGenerator);
                //float v = uniformFloat(normalGenerator);
                //const Farlor::Vector3 recieverDir = UniformlySampleHemisphereFacingNegativeX(u, v);
                const Farlor::Vector3 recieverDir = (recieverPos - emitterStart).Normalized();




#if defined(ExportGeometryInfo) && defined(ExportNormals)
                geometryExportStream << "    " << recieverDir.x << " " << recieverDir.y << " " << recieverDir.z << std::endl;
#endif

                RayGeometry rayReciever(recieverPos, recieverDir);


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
                    const Range arclengthRange = { targetArclength, targetArclength };

                    // This is the range we want to meet
                    // The range of actual curvature/torsion * ds is below
                    Range kdsRange = { 0.0f, 2.0f };
                    Range tdsRange = { -1.0f, 1.0f };

                    int randomSeed = boostrapperSeed;
                    if (randomSeed == 0)
                    {
                        randomSeed = time(0);
                    }

                    ExperimentRunner::ExperimentParameters experimentParams;
                    experimentParams.numPathsInExperiment = numPathsToGenerate;
                    experimentParams.numPathsToSkip = numPathsToSkip;
                    experimentParams.exportGeneratedCurves = false;
                    experimentParams.experimentName = experimentName;
                    experimentParams.numSegmentsPerCurve = 200;
                    experimentParams.maximumBootstrapCurveError = 0.5f;
                    experimentParams.curvePerturbMethod = ExperimentRunner::CurvePerturbMethod::SimpleGeometry;
                    experimentParams.curvePurturbSeed = perturbSeed;
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


                    {
                        GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, arclengthRange, randomSeed);
#if defined(ExportGeometryInfo) && defined(ExportArclengths)
                        twisty::Curve* pCurve = upExperimentRunner->GetInitialCurvePtr();
                        float arclength = pCurve->m_arclength;
                        geometryExportStream << "        " << arclength << std::endl;
#endif

                        // Here, calculate the thing
                        twisty::ExperimentRunner::ExperimentResults results = CalculateDipoleTransportKernel(bootstrapper, experimentParams, targetArclength);

                        const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
                        nonRotatedPixels[frameIdx] += results.experimentWeight * (1.0 / (numArclengths));
                        std::cout << "Non-Rotated Experiment Contribution: " << results.experimentWeight << std::endl;
                    }
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

#ifdef DoubleWithRotated
    {
        std::filesystem::path freezeFramePath = outputDirectoryPath;
        freezeFramePath.append("FreezeFrameRotated.dat");

        std::ofstream freezeFrameOutfile(freezeFramePath.string());
        if (!freezeFrameOutfile.is_open())
        {
            std::cout << "Failed to create rotated freezeframe outfile" << std::endl;
            exit(1);
        }

        // X
        freezeFrameOutfile << framePixelCount << std::endl;
        // Y
        freezeFrameOutfile << framePixelCount << std::endl;

        // Write out the pixel data
        for (uint32_t pixelIdxX = 0; pixelIdxX < framePixelCount; ++pixelIdxX)
        {
            for (uint32_t pixelIdxY = 0; pixelIdxY < framePixelCount; ++pixelIdxY)
            {
                // Output pixel
                const uint32_t frameIdx = pixelIdxY * framePixelCount + pixelIdxX;
                freezeFrameOutfile << rotatedPixels[frameIdx] << std::endl;
            }
        }
    }
#endif

    return 0;
}
