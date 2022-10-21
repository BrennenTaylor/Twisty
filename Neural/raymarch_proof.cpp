#include <FMath/FMath.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

const float DefaultAbsorbtion = 0.1f;
const float DefaultScattering = 0.1f;

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

float BeerLambert(float absorptionCoefficient, float distanceTraveled)
{
    return std::exp(-absorptionCoefficient * distanceTraveled);
}

Farlor::Vector3 RayMarchToLight(const Farlor::Vector3 &rayOrigin, const Farlor::Vector3 &lightPos)
{
    const float stepSize = 0.1f;
}

Farlor::Vector3 RayMarchSingleScatter(
      const Farlor::Vector3 &rayOrigin, const Farlor::Vector3 &rayDir)
{
    const int maxStepsRM = 1000;
    const float stepSize = 0.1f;

    float opaqueVisiblity = 1.0f;
    float volumeDepth = 0.0f;

    for (int i = 0; i < maxStepsRM; i++) {
        volumeDepth += stepSize;
        if (volumeDepth > opaqueDepth)
            break;

        Farlor::Vector3 position = rayOrigin + volumeDepth * rayDir;
        bool isInVolume = QueryVolumetricDistanceField(position) < 0.0f;
        if (isInVolume) {
            float previousOpaqueVisiblity = opaqueVisiblity;
            opaqueVisiblity *= BeerLambert(ABSORPTION_COEFFICIENT, marchSize);
            float absorptionFromMarch = previousOpaqueVisiblity - opaqueVisiblity;
            for (int lightIndex = 0; lightIndex < NUM_LIGHTS; lightIndex++) {
                float lightDistance = length((GetLight(lightIndex).Position - position));
                Farlor::Vector3 lightColor
                      = GetLight(lightIndex).LightColor * GetLightAttenuation(lightDistance);
                volumetricColor += absorptionFromMarch * volumeAlbedo * lightColor;
            }
            volumetricColor += absorptionFromMarch * volumeAlbedo * GetAmbientLight();
        }
    }
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
        lightPos.x = uniform01(generator);
        lightPos.y = uniform01(generator);
        lightPos.z = uniform01(generator);


        // Parallelize this?
        for (int sampleIdx = 0; sampleIdx < numDirectionsPerSample; sampleIdx++) {
            perSampleColorRM[sampleIdx] = RayMarch(inputPoint, sampledDirections[sampleIdx]);
        }
    }
}