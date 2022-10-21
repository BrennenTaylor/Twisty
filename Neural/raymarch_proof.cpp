#include "FMath/Vector3.h"
#include <FMath/FMath.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

const float DefaultAbsorbtion = 0.1f;
const float DefaultScattering = 0.1f;
const float DefaultDensity = 1.0f;
const float CubeSize = 19.0f;

struct VolumeData {
    bool inVolume = false;
    float absorbtion = 0.0f;
    float scattering = 0.0f;
};

VolumeData Cube(
      const Farlor::Vector3 &samplePointWS, const Farlor::Vector3 &cubeOrigin, float cubeFaceLength)
{
    const float halfFaceLength = cubeFaceLength * 0.5f;
    // Assume axis aligned
    if (samplePointWS.x > (cubeOrigin.x + halfFaceLength)
          || samplePointWS.x < (cubeOrigin.x - halfFaceLength)) {
        return { false, 0.0f, 0.0f };
    }
    if (samplePointWS.y > (cubeOrigin.y + halfFaceLength)
          || samplePointWS.y < (cubeOrigin.y - halfFaceLength)) {
        return { false, 0.0f, 0.0f };
    }
    if (samplePointWS.z > (cubeOrigin.z + halfFaceLength)
          || samplePointWS.z < (cubeOrigin.z - halfFaceLength)) {
        return { false, 0.0f, 0.0f };
    }
    return { true, DefaultAbsorbtion, DefaultScattering };
}

// TODO: Modify to include scattering?
float BeerLambert(float absorptionCoefficient, float scatteringCoefficient, float distanceTraveled)
{
    return std::exp(
          -distanceTraveled * DefaultDensity * (absorptionCoefficient + scatteringCoefficient));
}

float PhaseFunction(const Farlor::Vector3 &wo, const Farlor::Vector3 &wi)
{
    return 1.0f / (4.0f * std::numbers::pi_v<float>);
}

Farlor::Vector3 RayMarchToLight(const Farlor::Vector3 &rayOrigin, const Farlor::Vector3 &lightPos)
{
    const uint32_t resolution = 1024;
    const Farlor::Vector3 traceDir = (lightPos - rayOrigin).Normalized();
    const float stepSize = (lightPos - rayOrigin).Magnitude() / (resolution - 1);
    const Farlor::Vector3 stepVec = traceDir * stepSize;
    Farlor::Vector3 currentPos = rayOrigin;

    float distanceTraveledInMaterial = 0.0f;

    for (uint32_t i = 0; i < resolution; i++) {
        currentPos += stepVec;
        bool currentInMaterial
              = Cube(currentPos, Farlor::Vector3(0.0f, 0.0f, 0.0f), CubeSize).inVolume;
        if (currentInMaterial) {
            distanceTraveledInMaterial += stepSize;
        }
    }
    const float transmittence
          = BeerLambert(DefaultAbsorbtion, DefaultScattering, distanceTraveledInMaterial);
    return Farlor::Vector3 { transmittence, transmittence, transmittence };
}

Farlor::Vector3 RayMarchSingleScatter(const Farlor::Vector3 &rayOrigin,
      const Farlor::Vector3 &rayDir,
      const Farlor::Vector3 &lightPos)
{
    const float maxTraceDistance = 20.0f;
    const uint32_t resolution = 1024;
    const Farlor::Vector3 traceDir = rayDir.Normalized();
    const float stepSize = maxTraceDistance / (resolution - 1);
    const Farlor::Vector3 stepVec = traceDir * stepSize;
    Farlor::Vector3 currentPos = rayOrigin;

    float distanceTraveledInMaterial = 0.0f;
    Farlor::Vector3 accumulatedColor(0.0f, 0.0f, 0.0f);

    for (uint32_t i = 0; i < resolution; i++) {
        currentPos += stepVec;
        bool currentInMaterial
              = Cube(currentPos, Farlor::Vector3(0.0f, 0.0f, 0.0f), CubeSize).inVolume;
        if (currentInMaterial) {
            distanceTraveledInMaterial += stepSize;
            const float transmittence
                  = BeerLambert(DefaultAbsorbtion, DefaultScattering, distanceTraveledInMaterial);
            const float phaseWeight = PhaseFunction(traceDir, (lightPos - currentPos).Normalized());
            const Farlor::Vector3 colorUpdate = DefaultDensity * DefaultScattering * phaseWeight
                  * transmittence * RayMarchToLight(currentPos, lightPos);
            accumulatedColor += colorUpdate;
        }
    }
    return accumulatedColor;
}

int main()
{
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution uniform01(0.0f, 1.0f);

    const float sphereRadius = 10.0f;

    const uint32_t numDataSetPairs = 10;
    const uint32_t numDirectionsPerSample = 10;
    std::vector<Farlor::Vector3> sampledDirections(numDirectionsPerSample);
    std::vector<Farlor::Vector3> perSampleColorRM(numDirectionsPerSample);

    for (int dataPairIdx = 0; dataPairIdx < numDataSetPairs; dataPairIdx++) {
        for (auto &dir : sampledDirections) {
            // incorrect way
            float theta = 2.0f * std::numbers::pi_v<float> * uniform01(generator);
            float phi = std::acos(1.0f - 2.0f * uniform01(generator));
            dir.x = std::sin(phi) * std::cos(theta);
            dir.y = std::sin(phi) * std::sin(theta);
            dir.z = std::cos(phi);
            dir = dir.Normalized();
        }

        // Now we have samples, go ahead and ray march and raymarch

        // Generate the input vector
        float theta = 2.0f * std::numbers::pi_v<float> * uniform01(generator);
        float phi = std::acos(1.0f - 2.0f * uniform01(generator));
        Farlor::Vector3 inputPoint;
        inputPoint.x = std::sin(phi) * std::cos(theta);
        inputPoint.y = std::sin(phi) * std::sin(theta);
        inputPoint.z = std::cos(phi);
        inputPoint = inputPoint.Normalized() * sphereRadius;

        Farlor::Vector3 lightPos;
        // lightPos.x = uniform01(generator);
        // lightPos.y = uniform01(generator);
        // lightPos.z = uniform01(generator);

        lightPos.x = 0.0f;
        lightPos.y = 20.0f;
        lightPos.z = 0.0f;

        for (int sampleIdx = 0; sampleIdx < numDirectionsPerSample; sampleIdx++) {
            printf("\tData Pair %d, Sample %d\n", dataPairIdx, sampleIdx);
            Farlor::Vector3 &sampleDir = sampledDirections[sampleIdx];
            if (sampleDir.Dot((-1.0f * inputPoint).Normalized()) < 0.0f) {
                sampleDir *= -1.0f;
            }

            perSampleColorRM[sampleIdx] = RayMarchSingleScatter(inputPoint, sampleDir, lightPos);
            std::cout << "\tSample Color: " << perSampleColorRM[sampleIdx] << std::endl;
        }
    }
    std::cout << "Done" << std::endl;
}