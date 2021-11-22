#include "FullExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"

#include "Bootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"

#include <FMath/Vector3.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

const float distanceFromPlane = 10.0f;

const uint32_t numPathsPerInternal = 100000;
const uint32_t numPathsSkipPerInternal = 1000;

const double zMin = 0.0;
const double zMax = 10.0;

const double ringRadius = 1.75;

const uint32_t NumSegmentsPerCurve = 200;

Farlor::Vector3 UniformlySampleHemisphereFacingNegativeZ(const float u, const float v)
{
    float z = u;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * TwistyPi * v;

    return Farlor::Vector3(-1.0 * r * std::sin(phi), r * std::cos(phi), z).Normalized();
}

Farlor::Vector3 UniformlySampleSphereZAxis(const float u, const float v, bool flip)
{
    float z = u;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * TwistyPi * v;

    if (flip)
    {
        z = -z;
    }

    return Farlor::Vector3(-1.0 * r * std::sin(phi), r * std::cos(phi), z).Normalized();
}

int main(int argc, char* argv[])
{
    if (argc < 11)
    {
        std::cout << "Call as: " << argv[0] << " numZValues experimentName experimentOutputPath bootstrapperSeed perturbSeed numInitialCurves numPerInitialCurve numNormals normalSeed numArclengths" << std::endl;
        return 1;
    }

    const uint32_t numZValues = std::stoi(argv[1]);
    const std::string experimentName(argv[2]);
    const std::string experimentOutputPath(argv[3]);
    int bootstrapperSeed = std::stoi(argv[4]);
    int perturbSeed = std::stoi(argv[5]);
    const uint32_t numInitialCurves = std::stoi(argv[6]);
    const uint32_t numPerInitialCurve = std::stoi(argv[7]);
    const uint32_t numNormals = std::stoi(argv[8]);
    int normalSeed = std::stoi(argv[9]);
    const uint32_t numArclengths = std::stoi(argv[10]);

    std::filesystem::path outputDirectoryPath = std::filesystem::path(experimentOutputPath);
    std::cout << "Output Directory Path: " << outputDirectoryPath << std::endl;
    if (!std::filesystem::exists(outputDirectoryPath))
    {
        std::filesystem::create_directories(outputDirectoryPath);
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

    // We have a rotated and non-rotated version to test for initial seed curve impact
    std::vector<boost::multiprecision::cpp_dec_float_100> measuredZValues(numZValues);
    for (uint32_t z = 0; z < numZValues; z++)
    {
        measuredZValues[z] = 0.0;
    }

    // Ok, we want the ray emitter
    const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(0.0f, 0.0f, 1.0f).Normalized();
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

    const double deltaZ = (zMax - zMin) / (numZValues - 1);
    for (uint32_t z = 0; z < 4; ++z)
    {
        const double receiverZ = zMin + deltaZ * z;
        const Farlor::Vector3 recieverPos = Farlor::Vector3(ringRadius, 0.0f, receiverZ);


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

        // We probably want to add an order of magnitude here
        const Farlor::Vector3 recieverDir = (recieverPos - emitterStart).Normalized();

#if defined(ExportGeometryInfo) && defined(ExportNormals)
        geometryExportStream << "    " << recieverDir.x << " " << recieverDir.y << " " << recieverDir.z << std::endl;
#endif

        //twisty::RayGeometry rayReciever(recieverPos, recieverDir);


        // Start at somehting close to 1.05 times the minimum arclength and go to 1.5 times the arclength
        /*float targetArclength = (recieverPos - emitterStart).Magnitude() * 1.1f;
        targetArclength = std::max(targetArclength, 3.0f);*/

        // Lets change this to be the target arclength + a shifted amount based on the initial segments
        float targetArclength = (recieverPos - emitterStart).Magnitude();
        targetArclength += (targetArclength / NumSegmentsPerCurve) * 2.0;

        std::stringstream innerBlockSS;
        innerBlockSS << "<Z Value - " << z << ">";
        std::cout << innerBlockSS.str() << std::endl;

        boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;


        std::mt19937 normalGen(normalSeed);
        for (uint32_t normalIdx = 0; normalIdx < numNormals; ++normalIdx)
        {
            std::uniform_real_distribution<float> uniformFloats(0.0f, 1.0f);

            Farlor::Vector3 targetNormal = UniformlySampleSphereZAxis(uniformFloats(normalGen), uniformFloats(normalGen), (uniformFloats(normalGen) < 0.5f));
            std::cout << "\tTarget normal " << normalIdx << ": " << targetNormal << std::endl;

            const float minArclength = targetArclength;
            const float maxArclength = std::min(distanceFromPlane * 2.0, targetArclength * 2.0);

            const float deltaArclength = (maxArclength - minArclength) / (numArclengths - 1);

            for (uint32_t arclengthIdx = 0; arclengthIdx < numArclengths; ++arclengthIdx)
            {
                const float arclengthToUse = minArclength + arclengthIdx * deltaArclength;

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
                        experimentParams.numPathsInExperiment = numPathsPerInternal;
                        experimentParams.numPathsToSkip = numPathsSkipPerInternal;
                        experimentParams.arclength = targetArclength;
                        experimentParams.exportGeneratedCurves = false;
                        experimentParams.experimentName = experimentName;
                        experimentParams.numSegmentsPerCurve = NumSegmentsPerCurve;
                        experimentParams.maximumBootstrapCurveError = 0.5f;
                        experimentParams.bootstrapSeed = initialCurveSeed;
                        experimentParams.curvePurturbSeed = perCurveSeed;
                        experimentParams.rotateInitialSeedCurveRadians = 0.0f;

                        // Use a big mu value
                        experimentParams.weightingParameters.mu = 99.0;
                        experimentParams.weightingParameters.eps = 0.1;
                        experimentParams.weightingParameters.numStepsInt = 2000;
                        experimentParams.weightingParameters.minBound = 0.0;
                        experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;
                        experimentParams.weightingParameters.numCurvatureSteps = 10000;


                        experimentParams.weightingParameters.scatter = 0.99;
                        experimentParams.weightingParameters.absorbtion = 1.0 - experimentParams.weightingParameters.scatter;


                        experimentParams.rotateInitialSeedCurveRadians = 0.0f;

                        twisty::RayGeometry rayReciever(recieverPos, targetNormal);
                        twisty::Bootstrapper bootstrapper(rayEmitter, rayReciever);
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

                    averagedResult += (maxResult * (1.0 / (numInitialCurves * numNormals * numArclengths)));
                }
            }
        }

        measuredZValues[z] += averagedResult;
        std::cout << "Z Value: " << measuredZValues[z] << std::endl;
    }

    // Export freeze frame pixel data
    {
        std::filesystem::path zValuesOutputPath = outputDirectoryPath;
        zValuesOutputPath.append("ZValues.dat");

        std::ofstream zValuesOutputStream(zValuesOutputPath.string());
        if (!zValuesOutputStream.is_open())
        {
            std::cout << "Failed to create z values outfile" << std::endl;
            exit(1);
        }

        std::filesystem::path zDistancesOutputPath = outputDirectoryPath;
        zDistancesOutputPath.append("ZDistances.dat");

        std::ofstream zDistancesOutputStream(zDistancesOutputPath.string());
        if (!zDistancesOutputStream.is_open())
        {
            std::cout << "Failed to create z distances outfile" << std::endl;
            exit(1);
        }

        // Write out the pixel data
        for (uint32_t z = 0; z < numZValues; ++z)
        {
            zValuesOutputStream << measuredZValues[z] << std::endl;
            zDistancesOutputStream << (zMin + deltaZ * z) << std::endl;
        }
    }

    return 0;
}
