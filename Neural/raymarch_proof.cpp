#include "FMath/Vector3.h"
#include "FMath/Vector4.h"
#include <FMath/FMath.h>
#include <cmath>
#include <filesystem>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <vector>
#include <fstream>
#include <string>

const float DefaultAbsorbtion = 0.1f;
const float DefaultScattering = 0.1f;
const float OuterSphereRadius = 9.0f;
const float InnerSphereRadius = 5.0f;

// Render image stuff
Farlor::Vector3 lightIntensity(25.0f, 25.0f, 25.0f);
// const Farlor::Vector3 BackgroundColor(52.9 * 0.01, 80.8 * 0.01, 92.2 * 0.01);
const Farlor::Vector3 BackgroundColor(0.0f, 0.0f, 0.0f);

struct VolumeData {
    bool inVolume = false;
    float absorbtion = 0.0f;
    float scattering = 0.0f;
};

float CubeOfVolume(
      const Farlor::Vector3 &samplePointWS, const Farlor::Vector3 &cubeOrigin, float cubeFaceLength)
{
    const float halfFaceLength = cubeFaceLength * 0.5f;
    // Assume axis aligned
    if (samplePointWS.x > (cubeOrigin.x + halfFaceLength)
          || samplePointWS.x < (cubeOrigin.x - halfFaceLength)) {
        return { 0.0f };
    }
    if (samplePointWS.y > (cubeOrigin.y + halfFaceLength)
          || samplePointWS.y < (cubeOrigin.y - halfFaceLength)) {
        return { 0.0f };
    }
    if (samplePointWS.z > (cubeOrigin.z + halfFaceLength)
          || samplePointWS.z < (cubeOrigin.z - halfFaceLength)) {
        return { 0.0f };
    }
    return { 1.0f };
}

float SphereOfVolume(
      const Farlor::Vector3 &samplePointWS, const Farlor::Vector3 &sphereCenter, float sphereRadius)
{
    const float sphereRadius2 = (sphereRadius * sphereRadius);
    if ((samplePointWS - sphereCenter).SqrMagnitude() <= sphereRadius2) {
        return { 1.0f };
    }
    return { 0.0f };
}

float HollowSphereOfVolume(const Farlor::Vector3 &samplePointWS,
      const Farlor::Vector3 &sphereCenter,
      float sphereRadius,
      float innerRadius)
{
    const float sphereRadius2 = (sphereRadius * sphereRadius);
    const float innerRadius2 = (innerRadius * innerRadius);
    if ((samplePointWS - sphereCenter).SqrMagnitude() <= sphereRadius2) {
        if ((samplePointWS - sphereCenter).SqrMagnitude() >= innerRadius2) {
            return { 1.0f };
        }
    }
    return { 0.0f };
}

float BeerLambert(float absorptionCoefficient, float scatteringCoefficient, float distanceTraveled,
      float density)
{
    return std::exp(-distanceTraveled * density * (absorptionCoefficient + scatteringCoefficient));
}

float PhaseFunction(const Farlor::Vector3 &wo, const Farlor::Vector3 &wi)
{
    return 1.0f / (4.0f * std::numbers::pi_v<float>);
}

Farlor::Vector3 RayMarchToLight(const Farlor::Vector3 &rayOrigin, const Farlor::Vector3 &lightPos,
      std::mt19937 &raymarch_generator, std::uniform_real_distribution<float> &uniform01)
{
    const uint32_t numSteps = 1000;
    const Farlor::Vector3 traceDir = (lightPos - rayOrigin).Normalized();
    const float stepSize = (lightPos - rayOrigin).Magnitude() / (numSteps - 1);
    const Farlor::Vector3 stepVec = traceDir * stepSize;
    float transmittence = 1.0f;


    Farlor::Vector3 currentPos = rayOrigin;

    for (uint32_t stepIdx = 0; stepIdx < numSteps; stepIdx++) {
        const Farlor::Vector3 randomStepDist = stepVec * uniform01(raymarch_generator);
        Farlor::Vector3 samplePos = currentPos + randomStepDist;
        float sampledDensity = HollowSphereOfVolume(
              samplePos, Farlor::Vector3(0.0f, 0.0f, 0.0f), OuterSphereRadius, InnerSphereRadius);
        if (sampledDensity > 0.0f) {
            transmittence
                  *= BeerLambert(DefaultAbsorbtion, DefaultScattering, stepSize, sampledDensity);
        }
        currentPos += stepVec;
    }
    return Farlor::Vector3 { transmittence, transmittence, transmittence };
}

Farlor::Vector4 RayMarchSingleScatter(const Farlor::Vector3 &rayOrigin,
      const Farlor::Vector3 &rayDir, const Farlor::Vector3 &lightPos, std::mt19937 &raymarch_generator,
      std::uniform_real_distribution<float> &uniform01)
{
    const float maxTraceDistance = 100.0f;
    const uint32_t numSteps = 1000;
    const Farlor::Vector3 traceDir = rayDir.Normalized();
    const float stepSize = maxTraceDistance / (numSteps - 1);
    const Farlor::Vector3 stepVec = traceDir * stepSize;
    Farlor::Vector3 currentPos = rayOrigin;

    float transmittence = 1.0f;
    Farlor::Vector3 accumulatedColor(0.0f, 0.0f, 0.0f);

    for (uint32_t stepIdx = 0; stepIdx < numSteps; stepIdx++) {
        const Farlor::Vector3 randomStepDist = stepVec * uniform01(raymarch_generator);
        Farlor::Vector3 samplePos = currentPos + randomStepDist;
        float sampledDensity = HollowSphereOfVolume(
              samplePos, Farlor::Vector3(0.0f, 0.0f, 0.0f), OuterSphereRadius, InnerSphereRadius);
        if (sampledDensity > 0.0f) {
            transmittence *= BeerLambert(
                  DefaultAbsorbtion, DefaultScattering, stepSize, sampledDensity);
            const float phaseWeight = PhaseFunction(traceDir, (lightPos - samplePos).Normalized());
            const Farlor::Vector3 colorUpdate = sampledDensity * DefaultScattering * phaseWeight
                  * transmittence
                  * RayMarchToLight(samplePos, lightPos, raymarch_generator, uniform01);
            accumulatedColor += colorUpdate;
        }
        currentPos += stepVec;
    }
    return Farlor::Vector4(
          accumulatedColor.x, accumulatedColor.y, accumulatedColor.z, transmittence);
}

struct Ray {
    Farlor::Vector3 origin = Farlor::Vector3(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 dir = Farlor::Vector3(0.0f, 0.0f, 0.0f);
};

// Intersects ray r = p + td, |d| = 1, with sphere s and, if intersecting,
// returns t value of intersection and intersection point q

bool IntersectRaySphere(const Ray &ray, const Farlor::Vector3 &center, const float radius, float &t)
{
    Farlor::Vector3 m = ray.origin - center;
    float b = m.Dot(ray.dir);
    float c = m.Dot(m) - radius * radius;

    // Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0)
    if (c > 0.0f && b > 0.0f)
        return 0;
    float discr = b * b - c;

    // A negative discriminant corresponds to ray missing sphere
    if (discr < 0.0f)
        return false;

    // Ray now found to intersect sphere, compute smallest t value of intersection
    t = -b - std::sqrt(discr);

    // If t is negative, ray started inside sphere so clamp t to zero
    if (t < 0.0f)
        t = 0.0f;

    return true;
}

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

    void WriteExr(const std::string &filename)
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
    Farlor::Vector3 worldPos = Farlor::Vector3(0.0f, 0.0f, 40.0f);
    Farlor::Vector3 facingDir = Farlor::Vector3(0.0f, 0.0f, -1.0f);  // Face down z
    float width = 30.0f;
    float height = 30.0f;
    uint32_t dimX = 256;
    uint32_t dimY = 256;
    uint32_t spp = 1;

    std::vector<Farlor::Vector3> pixels;
    Farlor::Vector3 bottomLeft;
    float pixelWidth;
    float pixelHeight;
};

struct Sample {
    Farlor::Vector3 cart = Farlor::Vector3(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 sphere = Farlor::Vector3(0.0f, 0.0f, 1.0f);
};

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cout << "Incorrect number of arguments" << std::endl;
        return 0;
    }

    uint32_t seed = 1;//std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 sample_generator(seed);
    std::mt19937 raymarch_generator(seed + 1);
    std::uniform_real_distribution uniform01(0.0f, 1.0f);

    std::filesystem::path currentDir = std::filesystem::current_path();
    currentDir /= "Dataset";
    if (!std::filesystem::exists(currentDir)) {
        std::filesystem::create_directories(currentDir);
    }

    const std::string filename = (currentDir / "samples.csv").string();

    std::ofstream outfile(filename);

    // Render data pairs
    const float rasterSphereRadius = 10.0f;

    const uint32_t numDataSetPairs = std::stoi(argv[1]);
    const uint32_t numDirectionsPerSample = std::stoi(argv[2]);
    const uint32_t generateImage = std::stoi(argv[3]);

    std::vector<Sample> sampledDirections(numDirectionsPerSample);
    std::vector<Farlor::Vector3> perSampleColorRM(numDirectionsPerSample);

    Farlor::Vector3 lightPos;
    lightPos.x = 0.0f;
    lightPos.y = 40.0f;
    lightPos.z = 10.0f;

    for (int dataPairIdx = 0; dataPairIdx < numDataSetPairs; dataPairIdx++) {
        if ((dataPairIdx % 1) == 0) {
            std::cout << "\r"
                      << "Dataset: "
                      << static_cast<float>(dataPairIdx) / static_cast<float>(numDataSetPairs)
                        * 100.0f
                      << "%% done\t\t" << std::flush;
        }

        for (auto &dir : sampledDirections) {
            // incorrect way
            float theta = std::acos(1.0f - 2.0f * uniform01(sample_generator));
            float phi = 2.0f * std::numbers::pi_v<float> * uniform01(sample_generator);
            dir.sphere.x = theta;
            dir.sphere.y = phi;

            dir.cart.x = std::sin(theta) * std::cos(phi);
            dir.cart.y = std::sin(theta) * std::sin(phi);
            dir.cart.z = std::cos(theta);
            dir.cart = dir.cart.Normalized();
        }

        // Now we have samples, go ahead and ray march and raymarch

        // Generate the input vector
        float theta = std::acos(1.0f - 2.0f * uniform01(sample_generator));
        float phi = 2.0f * std::numbers::pi_v<float> * uniform01(sample_generator);
        Farlor::Vector3 inputPoint;
        inputPoint.x = std::sin(theta) * std::cos(phi);
        inputPoint.y = std::sin(theta) * std::sin(phi);
        inputPoint.z = std::cos(theta);
        inputPoint = inputPoint.Normalized() * OuterSphereRadius;


        for (int sampleIdx = 0; sampleIdx < numDirectionsPerSample; sampleIdx++) {
            Sample &sampleDir = sampledDirections[sampleIdx];
            if (sampleDir.cart.Dot((-1.0f * inputPoint).Normalized()) < 0.0f) {
                sampleDir.cart *= -1.0f;
                sampleDir.sphere.x = std::acos(sampleDir.cart.z);
                sampleDir.sphere.y = std::atan2(sampleDir.cart.y, sampleDir.cart.x);
            }

            const Farlor::Vector4 sampleColor = RayMarchSingleScatter(
                  inputPoint, sampleDir.cart, lightPos, raymarch_generator, uniform01);
            perSampleColorRM[sampleIdx].x = sampleColor.x;
            perSampleColorRM[sampleIdx].y = sampleColor.y;
            perSampleColorRM[sampleIdx].z = sampleColor.z;

            outfile << inputPoint.x << ", " << inputPoint.y << ", " << inputPoint.z << ", ";
            outfile << sampleDir.sphere.x << ", " << sampleDir.sphere.y << ", ";
            outfile << lightPos.x << ", " << lightPos.y << ", " << lightPos.z << ", ";
            outfile << perSampleColorRM[sampleIdx].x << ", " << perSampleColorRM[sampleIdx].y
                    << ", " << perSampleColorRM[sampleIdx].z << "\n";
        }
    }


    int numPixelsLit = 0;

    const std::string imageSamplesFilename = (currentDir / "image_samples.csv").string();

    std::ofstream imageSamplesOFS(imageSamplesFilename);

    Image img;
    for (uint32_t yIdx = 0; yIdx < img.DimY(); yIdx++) {
        for (uint32_t xIdx = 0; xIdx < img.DimX(); xIdx++) {
            const int pixelIdx = (xIdx + img.DimX() * yIdx);
            if ((pixelIdx % 100) == 0) {
                std::cout << "\r"
                          << "Image sample and gt generation: "
                          << static_cast<float>(pixelIdx)
                            / static_cast<float>(img.DimX() * img.DimY()) * 100.0f
                          << "%% done\t\t" << std::flush;
            }

            for (uint32_t sampleIdx = 0; sampleIdx < img.Spp(); sampleIdx++) {
                const float e0 = uniform01(sample_generator);
                const float e1 = uniform01(sample_generator);

                Ray sampleRay = img.GetRay(xIdx, yIdx, e0, e1);

                float t = 0.0f;
                bool intersectedProxy = IntersectRaySphere(
                      sampleRay, Farlor::Vector3(0.0f, 0.0f, 0.0f), rasterSphereRadius, t);

                imageSamplesOFS << (intersectedProxy ? 1 : 0) << ",";

                const Farlor::Vector3 intersectionPt = sampleRay.origin + t * sampleRay.dir;

                imageSamplesOFS << intersectionPt.x << ",";
                imageSamplesOFS << intersectionPt.y << ",";
                imageSamplesOFS << intersectionPt.z << ",";

                imageSamplesOFS << sampleRay.dir.x << ",";
                imageSamplesOFS << sampleRay.dir.y << ",";
                imageSamplesOFS << sampleRay.dir.z << ",";

                imageSamplesOFS << lightPos.x << ",";
                imageSamplesOFS << lightPos.y << ",";
                imageSamplesOFS << lightPos.z << std::endl;

                if (generateImage > 0) {
                    Farlor::Vector3 pixelColor(0.0f, 0.0f, 0.0f);
                    if (intersectedProxy) {
                        const Farlor::Vector4 marchedColor
                              = RayMarchSingleScatter(sampleRay.origin, sampleRay.dir, lightPos,
                                      raymarch_generator, uniform01);
                        const Farlor::Vector3 color(marchedColor.x, marchedColor.y, marchedColor.z);
                        const float transmittence = marchedColor.w;
                        pixelColor = (color * lightIntensity) + (transmittence * BackgroundColor * lightIntensity);
                    }
                    else {
                        pixelColor = BackgroundColor * lightIntensity;
                    }
                    img.AccessPixel(xIdx, yIdx, sampleIdx) = pixelColor;

                    if (img.AccessPixel(xIdx, yIdx, sampleIdx).Magnitude() > 0.0f)
                        numPixelsLit++;
                }
            }
        }
    }

    if (generateImage > 0) {
        img.Resolve();
        const std::string gtFilename = (currentDir / "gt.exr").string();
        img.WriteExr(gtFilename);
        std::cout << "Num lit: " << numPixelsLit / img.Spp() << std::endl;
        std::cout << "Percent lit: "
                  << 100.0f * static_cast<float>(numPixelsLit / img.Spp())
                    / (img.DimX() * img.DimY())
                  << std::endl;
    }

    std::cout << "Done" << std::endl;
}