#include <FMath/FMath.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

float BeerLambert(float absorptionCoefficient, float distanceTraveled)
{
    return std::exp(-absorptionCoefficient * distanceTraveled);
}

Farlor::Vector3 RayMarch()
{
    float opaqueVisiblity = 1.0f;
    const float marchSize = 0.6f;
    for (int i = 0; i < MAX_VOLUME_MARCH_STEPS; i++) {
        volumeDepth += marchSize;
        if (volumeDepth > opaqueDepth)
            break;

        Farlor::Vector3 position = rayOrigin + volumeDepth * rayDirection;
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

    const uint32_t numDataSetPairs = 10;
    for (int)

        const uint32_t numDirectionsPerSample = 10;
    std::vector<Farlor::Vector3> sampledPoints(numDirectionsPerSample);
    for (auto &point : sampledPoints) {
        // incorrect way
        float theta = 2.0f * std::numbers::pi_v<float> * uniform01(generator);
        float phi = std::acos(1.0f - 2.0f * uniform01(generator));
        point.x = std::sin(phi) * std::cos(theta);
        point.y = std::sin(phi) * std::sin(theta);
        point.z = std::cos(phi);
    }

    // Now we have samples, go ahead and ray march and raymarch
}