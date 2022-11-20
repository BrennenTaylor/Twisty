#include <openvdb/Grid.h>
#include <openvdb/Metadata.h>
#include <openvdb/math/Coord.h>
#include <openvdb/math/Transform.h>
#include <openvdb/math/Ray.h>
#include <openvdb/math/Vec3.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/LevelSetUtil.h>

#include "FMath/Vector3.h"
#include "FMath/Vector4.h"
#include <FMath/FMath.h>
#include <cmath>
#include <filesystem>
#include <openvdb/version.h>

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

const float ModelScaler = 1.0f;
const float DensityScale = 1.0f;

const float DefaultAbsorbtion = 0.1f;
const float DefaultScattering = 0.1f;
const float VolumeSphereRadius = 9.0f * ModelScaler;

// Render image stuff
Farlor::Vector3 lightIntensity(25.0f, 25.0f, 25.0f);
// const Farlor::Vector3 SkyBlue(52.9 * 0.01, 80.8 * 0.01, 92.2 * 0.01);
const Farlor::Vector3 SkyBlue(0.0f, 0.0f, 0.0f);


const Farlor::Vector3 ScaledSkyBlue = SkyBlue * lightIntensity;

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

float BeerLambert(float absorptionCoefficient, float scatteringCoefficient, float distanceTraveled,
      float density)
{
    return std::exp(-distanceTraveled * density * (absorptionCoefficient + scatteringCoefficient));
}

float PhaseFunction(const Farlor::Vector3 &wo, const Farlor::Vector3 &wi)
{
    return 1.0f / (4.0f * std::numbers::pi_v<float>);
}

Farlor::Vector3 RayMarchToLight(openvdb::FloatGrid::Ptr grid,
      openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> &lightInter,
      const Farlor::Vector3 &rayOrigin, const Farlor::Vector3 &lightPos, std::mt19937 &generator,
      std::uniform_real_distribution<float> &uniform01)
{
    const openvdb::math::Transform &linearTransform = grid->transform();
    const Farlor::Vector3 traceDir = (lightPos - rayOrigin).Normalized();
    const float maxLightDistance = (lightPos - rayOrigin).Magnitude();

    openvdb::math::Ray<double> ray;
    ray.setEye(openvdb::math::Vec3<float>(rayOrigin.x, rayOrigin.y, rayOrigin.z));
    ray.setDir(openvdb::math::Vec3<float>(traceDir.x, traceDir.y, traceDir.z));

    float transmittence = 1.0f;
    bool hitBox = lightInter.setWorldRay(ray);
    if (!hitBox) {
        return Farlor::Vector3 { transmittence, transmittence, transmittence };
    }

    const float traceStepSize = 0.1f * ModelScaler;

    const Farlor::Vector3 stepVec = traceDir * traceStepSize;

    double t0 = 0.0f;
    double t1 = 0.0f;
    while (lightInter.march(t0, t1)) {
        const float worldT0 = lightInter.getWorldTime(t0);
        if (worldT0 > maxLightDistance) {
            break;
        }
        const float worldT1 = std::min((float)lightInter.getWorldTime(t1), maxLightDistance);

        const float distToCover = (worldT1 - worldT0);
        const int numSteps = static_cast<int>(distToCover / traceStepSize);

        Farlor::Vector3 currentPos = rayOrigin + (traceDir * worldT0);

        for (uint32_t stepIdx = 0; stepIdx < numSteps; stepIdx++) {
            const Farlor::Vector3 randomStepDist = stepVec * uniform01(generator);
            Farlor::Vector3 samplePos = currentPos + randomStepDist;

            // Compute the location in world space
            openvdb::Vec3d worldSpacePoint(samplePos.x, samplePos.y, samplePos.z);
            openvdb::Vec3d indexSpacePoint = linearTransform.worldToIndex(worldSpacePoint);

            openvdb::FloatGrid::ConstAccessor accessor = grid->getConstAccessor();
            openvdb::FloatGrid::ValueType sampledDensity
                  = openvdb::tools::BoxSampler::sample(grid->tree(), indexSpacePoint)
                  * DensityScale;

            if (sampledDensity > 0.0f) {
                transmittence *= BeerLambert(
                      DefaultAbsorbtion, DefaultScattering, traceStepSize, sampledDensity);
            }
            currentPos += stepVec;
        }
    }

    return Farlor::Vector3 { transmittence, transmittence, transmittence };
}

Farlor::Vector3 RayMarchSingleScatter(openvdb::FloatGrid::Ptr grid,
      openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> &inter,
      const Farlor::Vector3 &rayOrigin, const Farlor::Vector3 &rayDir,
      const Farlor::Vector3 &lightPos, std::mt19937 &generator,
      std::uniform_real_distribution<float> &uniform01)
{
    const openvdb::math::Transform &linearTransform = grid->transform();

    openvdb::math::Ray<double> ray;
    ray.setEye(openvdb::math::Vec3<float>(rayOrigin.x, rayOrigin.y, rayOrigin.z));
    ray.setDir(openvdb::math::Vec3<float>(rayDir.x, rayDir.y, rayDir.z));
    bool hitBox = inter.setWorldRay(ray);
    if (!hitBox) {
        return Farlor::Vector3(0.0f, 0.0f, 0.0f);
    }

    const float traceStepSize = 0.05f * ModelScaler;
    const Farlor::Vector3 traceDir = rayDir.Normalized();
    const Farlor::Vector3 stepVec = traceDir * traceStepSize;

    float transmittence = 1.0f;
    Farlor::Vector3 accumulatedColor(0.0f, 0.0f, 0.0f);

    openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> lightInter(inter);

    double t0 = 0.0f;
    double t1 = 0.0f;
    while (inter.march(t0, t1)) {
        const float worldT0 = inter.getWorldTime(t0);
        const float worldT1 = inter.getWorldTime(t1);
        // std::cout << worldT0 << ", " << worldT1 << std::endl;

        const float distToCover = (worldT1 - worldT0);
        const int numSteps = static_cast<int>(distToCover / traceStepSize);

        Farlor::Vector3 currentPos = rayOrigin + (rayDir * worldT0);

        for (uint32_t stepIdx = 0; stepIdx < numSteps; stepIdx++) {
            const Farlor::Vector3 randomStepDist = stepVec * uniform01(generator);
            Farlor::Vector3 samplePos = currentPos + randomStepDist;

            // Compute the location in world space
            openvdb::Vec3d worldSpacePoint(samplePos.x, samplePos.y, samplePos.z);
            openvdb::Vec3d indexSpacePoint = linearTransform.worldToIndex(worldSpacePoint);

            openvdb::FloatGrid::ConstAccessor accessor = grid->getConstAccessor();
            openvdb::FloatGrid::ValueType sampledDensity
                  = openvdb::tools::BoxSampler::sample(grid->tree(), indexSpacePoint)
                  * DensityScale;

            if (sampledDensity > 0.0f) {
                transmittence *= BeerLambert(
                      DefaultAbsorbtion, DefaultScattering, traceStepSize, sampledDensity);
                const float phaseWeight
                      = PhaseFunction(traceDir, (lightPos - samplePos).Normalized());
                const Farlor::Vector3 colorUpdate = sampledDensity * DefaultScattering * phaseWeight
                      * transmittence
                      * RayMarchToLight(
                            grid, lightInter, samplePos, lightPos, generator, uniform01);
                accumulatedColor += colorUpdate;
            }
            currentPos += stepVec;
        }
    }
    return accumulatedColor + SkyBlue * transmittence;
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
    // RHS
    Farlor::Vector3 worldPos = Farlor::Vector3(0.0f, 0.0f, 40.0f) * ModelScaler;
    Farlor::Vector3 facingDir = Farlor::Vector3(0.0f, 0.0f, -1.0f);  // Face down z
    float width = 30.0f * ModelScaler;
    float height = 30.0f * ModelScaler;
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
    uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution uniform01(0.0f, 1.0f);


    openvdb::initialize();
    openvdb::io::File vdbFile("bunny.vdb");
    vdbFile.open();

    openvdb::GridBase::Ptr baseGrid = vdbFile.readGrid(vdbFile.beginName().gridName());
    vdbFile.close();

    openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
    if (grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
        openvdb::tools::sdfToFogVolume(*grid);
    }

    openvdb::math::Transform::Ptr linearTransform
          = openvdb::math::Transform::createLinearTransform(0.02f);
    linearTransform->postTranslate(openvdb::Vec3d(1.0f, -7.0f, 0.0f));
    grid->setTransform(linearTransform);

    for (openvdb::MetaMap::MetaIterator iter = grid->beginMeta(); iter != grid->endMeta(); ++iter) {
        const std::string &name = iter->first;
        openvdb::Metadata::Ptr value = iter->second;
        std::string valueAsString = value->str();
        std::cout << name << " = " << valueAsString << std::endl;
    }

    openvdb::v10_0::math::CoordBBox bbox = grid->evalActiveVoxelBoundingBox();
    std::cout << grid->indexToWorld(bbox.min()) << ", " << grid->indexToWorld(bbox.max())
              << std::endl;

    openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> inter(*grid);

    std::filesystem::path currentDir = std::filesystem::current_path();
    currentDir /= "Dataset";
    if (!std::filesystem::exists(currentDir)) {
        std::filesystem::create_directories(currentDir);
    }

    const std::string filename = (currentDir / "samples.csv").string();

    std::ofstream outfile(filename);

    // Render data pairs
    const float rasterSphereRadius = 20.0f * ModelScaler;

    const uint32_t numDataSetPairs = std::stoi(argv[1]);
    const uint32_t numDirectionsPerSample = std::stoi(argv[2]);

    const uint32_t generateImage = std::stoi(argv[3]);

    std::vector<Sample> sampledDirections(numDirectionsPerSample);
    std::vector<Farlor::Vector3> perSampleColorRM(numDirectionsPerSample);

    Farlor::Vector3 lightPos;
    lightPos.x = 0.0f * ModelScaler;
    lightPos.y = 30.0f * ModelScaler;
    lightPos.z = 0.0f * ModelScaler;

    for (int dataPairIdx = 0; dataPairIdx < numDataSetPairs; dataPairIdx++) {
        if ((dataPairIdx % 1) == 0) {
            std::cout << "\r"
                      << "Dataset: "
                      << static_cast<float>(dataPairIdx) / static_cast<float>(numDataSetPairs)
                        * 100.0f
                      << "%% done" << std::flush;
        }

        for (auto &dir : sampledDirections) {
            // incorrect way
            float theta = std::acos(1.0f - 2.0f * uniform01(generator));
            float phi = 2.0f * std::numbers::pi_v<float> * uniform01(generator);
            dir.sphere.x = theta;
            dir.sphere.y = phi;

            dir.cart.x = std::sin(theta) * std::cos(phi);
            dir.cart.y = std::sin(theta) * std::sin(phi);
            dir.cart.z = std::cos(theta);
            dir.cart = dir.cart.Normalized();
        }

        // Now we have samples, go ahead and ray march and raymarch

        // Generate the input vector
        float theta = std::acos(1.0f - 2.0f * uniform01(generator));
        float phi = 2.0f * std::numbers::pi_v<float> * uniform01(generator);
        Farlor::Vector3 inputPoint;
        inputPoint.x = std::sin(theta) * std::cos(phi);
        inputPoint.y = std::sin(theta) * std::sin(phi);
        inputPoint.z = std::cos(theta);
        inputPoint = inputPoint.Normalized() * VolumeSphereRadius;


        for (int sampleIdx = 0; sampleIdx < numDirectionsPerSample; sampleIdx++) {
            Sample &sampleDir = sampledDirections[sampleIdx];
            if (sampleDir.cart.Dot((-1.0f * inputPoint).Normalized()) < 0.0f) {
                sampleDir.cart *= -1.0f;
                sampleDir.sphere.x = std::acos(sampleDir.cart.z);
                sampleDir.sphere.y = std::atan2(sampleDir.cart.y, sampleDir.cart.x);
            }

            perSampleColorRM[sampleIdx] = RayMarchSingleScatter(
                  grid, inter, inputPoint, sampleDir.cart, lightPos, generator, uniform01);

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
                          << "%% done" << std::flush;
            }

            for (uint32_t sampleIdx = 0; sampleIdx < img.Spp(); sampleIdx++) {
                const float e0 = uniform01(generator);
                const float e1 = uniform01(generator);

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
                    Farlor::Vector3 pixelColor = ScaledSkyBlue;
                    if (intersectedProxy) {
                        Farlor::Vector3 marchedColor = RayMarchSingleScatter(grid, inter,
                              sampleRay.origin, sampleRay.dir, lightPos, generator, uniform01);
                        if (marchedColor.SqrMagnitude() > 0.0f) {
                            pixelColor = marchedColor * lightIntensity;
                        }
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