#include <FMath/Vector3.h>

#include <GeometryBootstrapper.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>
#include <string>
#include <random>

#include <boost/multiprecision/cpp_dec_float.hpp>


const double M_PI = 3.141592;

const float distanceFromPlane = 10.0f;

const double zMin = 0.0;
const double zMax = 10.0;

const double ringRadius = 1.75;

double GreensFunctionApprox(double scatteringCoefficient, double absorbtionCoefficient, double s, const Farlor::Vector3& x2, const Farlor::Vector3& n2, const Farlor::Vector3& x1, const Farlor::Vector3& n1)
{


    const double p = scatteringCoefficient / 2.0;

    const Farlor::Vector3 r = p * (x2 - x1);
    const double A = pow((p * s - tanh(p * s)), -1);
    const double B = tanh(p * s) / 2.0;
    const double C = 9.0 / (2.0 * p * s);
    const double D = 3.0 * A * B * B - ((3.0) / (2.0 * sinh(2.0 * p * s)));
    const double E = 3.0 * A * B;
    const double F = (3.0 / 2.0) * A;



    double normalization = 1.0f;
    {
        normalization *= (p * p * p);
        double num = sqrt(F) * (E * E - 2.0 * D * F);
        double den = 4.0 * pow(M_PI, (5.0 / 2.0)) * (exp((E * E) / (F - D)) - exp(D));
        normalization *= (num / den);
        normalization *= exp(C);
    }

    double inside = (-C - D * n2.Dot(n1) + E * r.Dot(n2 + n1) - F * r.SqrMagnitude() - absorbtionCoefficient * s);
    double result = normalization * exp(inside);


    return result;
}

Farlor::Vector3 UniformlySampleHemisphereFacingNegativeX(const float u, const float v)
{
    float z = u;
    float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    float phi = 2.0f * M_PI * v;

    return Farlor::Vector3(-1.0 * r * std::sin(phi), r * std::cos(phi), z).Normalized();
}

int main(int argc, char* argv[])
{
    if (argc < 7)
    {
        std::cout << "Call as: " << argv[0] << " numZValues experimentName experimentOutputPath numNormals normalSeed numArclengths" << std::endl;
        return 1;
    }

    const uint32_t numZValues = std::stoi(argv[1]);
    const std::string experimentName(argv[2]);
    const std::string experimentOutputPath(argv[3]);
    const uint32_t numNormals = std::stoi(argv[4]);
    int normalSeed = std::stoi(argv[5]);
    const uint32_t numArclengths = std::stoi(argv[6]);

    std::filesystem::path outputDirectoryPath = std::filesystem::path(experimentOutputPath);
    std::cout << "Output Directory Path: " << outputDirectoryPath << std::endl;
    if (!std::filesystem::exists(outputDirectoryPath))
    {
        std::filesystem::create_directories(outputDirectoryPath);
    }

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

    const double scatteringCoefficient = 0.99;
    const double absorbtionCoefficient = 1.0 - scatteringCoefficient;

    const double deltaZ = (zMax - zMin) / (numZValues - 1);
    for (uint32_t z = 0; z < numZValues; ++z)
    {
        const double receiverZ = zMin + deltaZ * z;
        const Farlor::Vector3 recieverPos = Farlor::Vector3(ringRadius, 0.0f, receiverZ);

        // We probably want to add an order of magnitude here
        const Farlor::Vector3 recieverDir = (recieverPos - emitterStart).Normalized();

        // Start at somehting close to 1.05 times the minimum arclength and go to 1.5 times the arclength
        float targetArclength = (recieverPos - emitterStart).Magnitude() * 1.1f;
        targetArclength = std::max(targetArclength, 3.0f);

        std::stringstream innerBlockSS;
        innerBlockSS << "<Z Value - " << z << ">";
        std::cout << innerBlockSS.str() << std::endl;

        const float minArclength = targetArclength;
        const float maxArclength = distanceFromPlane * 2.0;
        const float deltaArclength = (maxArclength - minArclength) / (numArclengths - 1);

        boost::multiprecision::cpp_dec_float_100 averagedResult = 0.0;
        for (uint32_t arclengthIdx = 0; arclengthIdx < numArclengths; ++arclengthIdx)
        {
            const float arclengthToUse = minArclength + arclengthIdx * deltaArclength;

            std::mt19937 normalGen(normalSeed);
            for (uint32_t normalIdx = 0; normalIdx < numNormals; ++normalIdx)
            {
                std::uniform_real_distribution<float> uniformFloats(0.0f, 1.0f);

                Farlor::Vector3 targetNormal = UniformlySampleHemisphereFacingNegativeX(uniformFloats(normalGen), uniformFloats(normalGen));
               
                std::cout << "\tTarget arclength " << arclengthIdx << ": " << arclengthToUse << std::endl;
                std::cout << "\tTarget normal " << normalIdx << ": " << targetNormal << std::endl;

                averagedResult = GreensFunctionApprox(scatteringCoefficient, absorbtionCoefficient, arclengthToUse, recieverPos, targetNormal, emitterStart, emitterDir) * (1.0 / (numNormals * numArclengths));
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
