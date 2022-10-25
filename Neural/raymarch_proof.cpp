#include "FMath/Vector3.h"
#include <FMath/FMath.h>


#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <vector>
#include <fstream>

const float DefaultAbsorbtion = 0.1f;
const float DefaultScattering = 0.1f;
const float DefaultDensity = 1.0f;
const float CubeSize = 19.0f;
const float VolumeSphereRadius = 9.0f;

struct VolumeData {
    bool inVolume = false;
    float absorbtion = 0.0f;
    float scattering = 0.0f;
};

VolumeData CubeOfVolume(
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

VolumeData SphereOfVolume(
      const Farlor::Vector3 &samplePointWS, const Farlor::Vector3 &sphereCenter, float sphereRadius)
{
    const float sphereRadius2 = (sphereRadius * sphereRadius);
    if ((samplePointWS - sphereCenter).SqrMagnitude() <= sphereRadius2) {
        return { true, DefaultAbsorbtion, DefaultScattering };
    }
    return { false, 0.0f, 0.0f };
}

VolumeData HollowSphereOfVolume(const Farlor::Vector3 &samplePointWS,
      const Farlor::Vector3 &sphereCenter,
      float sphereRadius,
      float innerRadius)
{
    const float sphereRadius2 = (sphereRadius * sphereRadius);
    const float innerRadius2 = (innerRadius * innerRadius);
    if ((samplePointWS - sphereCenter).SqrMagnitude() <= sphereRadius2) {
        if ((samplePointWS - sphereCenter).SqrMagnitude() >= innerRadius2) {
            return { true, DefaultAbsorbtion, DefaultScattering };
        }
    }
    return { false, 0.0f, 0.0f };
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

Farlor::Vector3 RayMarchToLight(const Farlor::Vector3 &rayOrigin, const Farlor::Vector3 &lightPos,
      std::mt19937 &generator, std::uniform_real_distribution<float> &uniform01)
{
    const uint32_t resolution = 1000;
    const Farlor::Vector3 traceDir = (lightPos - rayOrigin).Normalized();
    const float stepSize = (lightPos - rayOrigin).Magnitude() / (resolution - 1);
    const Farlor::Vector3 stepVec = traceDir * stepSize;
    Farlor::Vector3 currentPos = rayOrigin;

    float distanceTraveledInMaterial = 0.0f;

    for (uint32_t i = 0; i < resolution; i++) {
        currentPos += stepVec;
        Farlor::Vector3 samplePos = currentPos - (stepVec * uniform01(generator));
        // bool currentInMaterial
        //       = Cube(currentPos, Farlor::Vector3(0.0f, 0.0f, 0.0f), CubeSize).inVolume;
        // bool currentInMaterial
        //       = SphereOfVolume(currentPos, Farlor::Vector3(0.0f, 0.0f, 0.0f), VolumeSphereRadius)
        //               .inVolume;
        bool currentInMaterial = HollowSphereOfVolume(
              samplePos, Farlor::Vector3(0.0f, 0.0f, 0.0f), VolumeSphereRadius, 5.0f)
                                       .inVolume;
        if (currentInMaterial) {
            distanceTraveledInMaterial += stepSize;
        }
    }
    const float transmittence
          = BeerLambert(DefaultAbsorbtion, DefaultScattering, distanceTraveledInMaterial);
    return Farlor::Vector3 { transmittence, transmittence, transmittence };
}

Farlor::Vector3 RayMarchSingleScatter(const Farlor::Vector3 &rayOrigin,
      const Farlor::Vector3 &rayDir, const Farlor::Vector3 &lightPos, std::mt19937 &generator,
      std::uniform_real_distribution<float> &uniform01)
{
    // uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    // std::mt19937 generator(seed);
    // std::uniform_real_distribution uniform01(0.0f, 1.0f);

    const float maxTraceDistance = 100.0f;
    const uint32_t resolution = 1000;
    const Farlor::Vector3 traceDir = rayDir.Normalized();
    const float stepSize = maxTraceDistance / (resolution - 1);
    const Farlor::Vector3 stepVec = traceDir * stepSize;
    Farlor::Vector3 currentPos = rayOrigin;

    float distanceTraveledInMaterial = 0.0f;
    Farlor::Vector3 accumulatedColor(0.0f, 0.0f, 0.0f);

    for (uint32_t i = 0; i < resolution; i++) {
        const Farlor::Vector3 randomStepDist = stepVec * uniform01(generator);
        Farlor::Vector3 samplePos = currentPos + randomStepDist;
        // bool currentInMaterial
        //       = Cube(currentPos, Farlor::Vector3(0.0f, 0.0f, 0.0f), CubeSize).inVolume;
        // bool currentInMaterial
        //       = SphereOfVolume(currentPos, Farlor::Vector3(0.0f, 0.0f, 0.0f), VolumeSphereRadius)
        //               .inVolume;
        bool currentInMaterial = HollowSphereOfVolume(
              samplePos, Farlor::Vector3(0.0f, 0.0f, 0.0f), VolumeSphereRadius, 5.0f)
                                       .inVolume;
        if (currentInMaterial) {
            distanceTraveledInMaterial += randomStepDist.Magnitude();
            const float transmittence
                  = BeerLambert(DefaultAbsorbtion, DefaultScattering, distanceTraveledInMaterial);
            const float phaseWeight = PhaseFunction(traceDir, (lightPos - samplePos).Normalized());
            const Farlor::Vector3 colorUpdate = DefaultDensity * DefaultScattering * phaseWeight
                  * transmittence * RayMarchToLight(samplePos, lightPos, generator, uniform01);
            accumulatedColor += colorUpdate;
        }
        currentPos += stepVec;
    }
    return accumulatedColor;
}

struct Ray {
    Farlor::Vector3 origin = Farlor::Vector3(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 dir = Farlor::Vector3(0.0f, 0.0f, 0.0f);
};

class Image {
   public:
    Image()
        : pixels(dimX * dimY * spp)
        , pixelWidth(width / dimX)
        , pixelHeight(height / dimY)
    {
        bottomLeft = worldPos;
        bottomLeft.x -= (width / 2.0f);
        bottomLeft.y -= (height / 2.0f);
    }

    Farlor::Vector3 &AccessPixel(int xIdx, int yIdx, uint32_t sampleIdx)
    {
        return pixels[sampleIdx + xIdx * spp + yIdx * (dimX * spp)];
    }

    Ray GetRay(int xIdx, int yIdx, float e0 = 0.5f, float e1 = 0.5f)
    {
        Farlor::Vector3 samplePos = bottomLeft;

        // Moves to center of pixel
        samplePos.x += pixelWidth * (e0 + static_cast<float>(xIdx));
        samplePos.y += pixelHeight * (e1 + static_cast<float>(dimY - yIdx));
        return { samplePos, facingDir };
    }

    void Resolve()
    {
        const float scaler = 1.0f / spp;
        for (uint32_t yIdx = 0; yIdx < dimY; yIdx++) {
            for (uint32_t xIdx = 0; xIdx < dimX; xIdx++) {
                Farlor::Vector3 avgPixel(0.0f, 0.0f, 0.0f);
                for (uint32_t sampleIdx = 0; sampleIdx < spp; sampleIdx++) {
                    avgPixel += pixels[sampleIdx + xIdx * spp + yIdx * (dimX * spp)];
                }
                avgPixel *= scaler;
                pixels[0 + xIdx * spp + yIdx * (dimX * spp)];
            }
        }
    }

    uint32_t DimX() const { return dimX; }
    uint32_t DimY() const { return dimY; }
    uint32_t Spp() const { return spp; }

    void WriteExr(std::string filename)
    {
        EXRHeader header;
        InitEXRHeader(&header);

        EXRImage image;
        InitEXRImage(&image);

        image.num_channels = 3;

        std::vector<float> images[3];
        images[0].resize(dimX * dimY);
        images[1].resize(dimX * dimY);
        images[2].resize(dimX * dimY);

        for (int i = 0; i < dimX * dimY; i++) {
            images[0][i] = pixels[i * spp].x;
            images[1][i] = pixels[i * spp].y;
            images[2][i] = pixels[i * spp].z;
        }


        float *image_ptr[3];
        image_ptr[0] = &(images[2].at(0));  // B
        image_ptr[1] = &(images[1].at(0));  // G
        image_ptr[2] = &(images[0].at(0));  // R

        image.images = (unsigned char **)image_ptr;
        image.width = dimX;
        image.height = dimY;

        header.num_channels = 3;
        header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
        // Must be BGR(A) order, since most of EXR viewers expect this channel order.
        strncpy(header.channels[0].name, "B", 255);
        header.channels[0].name[strlen("B")] = '\0';
        strncpy(header.channels[1].name, "G", 255);
        header.channels[1].name[strlen("G")] = '\0';
        strncpy(header.channels[2].name, "R", 255);
        header.channels[2].name[strlen("R")] = '\0';

        header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
        header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
        for (int i = 0; i < header.num_channels; i++) {
            header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;  // pixel type of input image
            header.requested_pixel_types[i]
                  = TINYEXR_PIXELTYPE_HALF;  // pixel type of output image to be stored in .EXR
        }

        const char *err;
        int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
        if (ret != TINYEXR_SUCCESS) {
            fprintf(stderr, "Save EXR err: %s\n", err);
            return;
        }
        printf("Saved exr file. [ %s ] \n", filename.c_str());

        free(header.channels);
        free(header.pixel_types);
        free(header.requested_pixel_types);
    }

   private:
    Farlor::Vector3 worldPos = Farlor::Vector3(0.0f, 0.0f, -40.0f);
    Farlor::Vector3 facingDir = Farlor::Vector3(0.0f, 0.0f, 1.0f);  // Face down z
    float width = 30.0f;
    float height = 30.0f;
    uint32_t dimX = 256;
    uint32_t dimY = 256;
    uint32_t spp = 4;

    std::vector<Farlor::Vector3> pixels;
    Farlor::Vector3 bottomLeft;
    float pixelWidth;
    float pixelHeight;
};

int main()
{
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution uniform01(0.0f, 1.0f);


    // Render data pairs
    // const float rasterSphereRadius = 10.0f;

    // const uint32_t numDataSetPairs = 10;
    // const uint32_t numDirectionsPerSample = 10;
    // std::vector<Farlor::Vector3> sampledDirections(numDirectionsPerSample);
    // std::vector<Farlor::Vector3> perSampleColorRM(numDirectionsPerSample);

    // for (int dataPairIdx = 0; dataPairIdx < numDataSetPairs; dataPairIdx++) {
    //     for (auto &dir : sampledDirections) {
    //         // incorrect way
    //         float theta = std::acos(1.0f - 2.0f * uniform01(generator));
    //         float phi = 2.0f * std::numbers::pi_v<float> * uniform01(generator);
    //         dir.x = std::sin(theta) * std::cos(phi);
    //         dir.y = std::sin(theta) * std::sin(phi);
    //         dir.z = std::cos(theta);
    //         dir = dir.Normalized();
    //     }

    //     // Now we have samples, go ahead and ray march and raymarch

    //     // Generate the input vector
    //     float theta = std::acos(1.0f - 2.0f * uniform01(generator));
    //     float phi = 2.0f * std::numbers::pi_v<float> * uniform01(generator);
    //     Farlor::Vector3 inputPoint;
    //     inputPoint.x = std::sin(theta) * std::cos(phi);
    //     inputPoint.y = std::sin(theta) * std::sin(phi);
    //     inputPoint.z = std::cos(theta);
    //     inputPoint = inputPoint.Normalized() * VolumeSphereRadius;

    //     Farlor::Vector3 lightPos;
    //     // lightPos.x = uniform01(generator);
    //     // lightPos.y = uniform01(generator);
    //     // lightPos.z = uniform01(generator);

    //     lightPos.x = 0.0f;
    //     lightPos.y = 20.0f;
    //     lightPos.z = 0.0f;

    //     for (int sampleIdx = 0; sampleIdx < numDirectionsPerSample; sampleIdx++) {
    //         printf("\tData Pair %d, Sample %d\n", dataPairIdx, sampleIdx);
    //         Farlor::Vector3 &sampleDir = sampledDirections[sampleIdx];
    //         if (sampleDir.Dot((-1.0f * inputPoint).Normalized()) < 0.0f) {
    //             sampleDir *= -1.0f;
    //         }

    //         perSampleColorRM[sampleIdx] = RayMarchSingleScatter(inputPoint, sampleDir, lightPos);
    //         std::cout << "\tSample Color: " << perSampleColorRM[sampleIdx] << std::endl;
    //     }
    // }

    // Render image stuff
    Farlor::Vector3 lightPos;
    lightPos.x = 0.0f;
    lightPos.y = 40.0f;
    lightPos.z = 10.0f;
    Farlor::Vector3 lightIntensity(25.0f, 25.0f, 25.0f);

    int numPixelsLit = 0;

    std::ofstream testFile("Temp.txt");


    Image img;
    for (uint32_t yIdx = 0; yIdx < img.DimY(); yIdx++) {
        std::cout << "Row: " << yIdx << "\n";
        for (uint32_t xIdx = 0; xIdx < img.DimX(); xIdx++) {
            for (uint32_t sampleIdx = 0; sampleIdx < img.Spp(); sampleIdx++) {
                const float e0 = uniform01(generator);
                const float e1 = uniform01(generator);

                Ray sampleRay = img.GetRay(xIdx, yIdx, e0, e1);
                // std::cout << "Sample ray: " << sampleRay.origin << ", " << sampleRay.dir << std::endl;
                img.AccessPixel(xIdx, yIdx, sampleIdx)
                      = RayMarchSingleScatter(
                              sampleRay.origin, sampleRay.dir, lightPos, generator, uniform01)
                      * lightIntensity;
                if (img.AccessPixel(xIdx, yIdx, sampleIdx).Magnitude() > 0.0f)
                    numPixelsLit++;
                testFile << img.AccessPixel(xIdx, yIdx, sampleIdx) << std::endl;
            }
        }
    }
    img.Resolve();
    img.WriteExr("RayMarchTest.exr");
    std::cout << "Num lit: " << numPixelsLit / img.Spp() << std::endl;
    std::cout << "Percent lit: "
              << 100.0f * static_cast<float>(numPixelsLit / img.Spp()) / (img.DimX() * img.DimY())
              << std::endl;

    std::cout << "Done" << std::endl;
}