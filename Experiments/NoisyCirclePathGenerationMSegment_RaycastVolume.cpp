#include "ExperimentBase.h"

#include "CombinedWeightUtils.h"
#include "Curve.h"
#include "CurvePerturbUtils.h"
#include "ExperimentRunner.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "boost/multiprecision/cpp_dec_float.hpp"


#include "Bootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "PathWeighters.h"
#include "boost/multiprecision/detail/default_ops.hpp"

#include <FMath/Vector3.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetUtil.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/RayIntersector.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

class VDBVolume {
   public:
    VDBVolume(const float outerRadius, float innerRadius)
    {
        m_grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
              outerRadius, openvdb::Vec3f(0.0f, 0.0f, 0.0f), 0.01f);
        openvdb::FloatGrid::Ptr removeGrid
              = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
                    innerRadius, openvdb::Vec3f(0.0f, 0.0f, 0.0f), 0.01f);
        m_grid = openvdb::tools::csgDifferenceCopy(*m_grid, *removeGrid);
        if (m_grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
            openvdb::tools::sdfToFogVolume(*m_grid);
        }
    }

    VDBVolume(std::string filename, float scale, float xOffset, float yOffset, float zOffset)
    {
        openvdb::io::File vdbFile(filename);
        vdbFile.open();

        openvdb::GridBase::Ptr baseGrid = vdbFile.readGrid(vdbFile.beginName().gridName());
        vdbFile.close();

        m_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        if (m_grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
            openvdb::tools::sdfToFogVolume(*m_grid);
        }

        openvdb::math::Transform::Ptr linearTransform
              = openvdb::math::Transform::createLinearTransform(scale);
        linearTransform->postRotate(-openvdb::math::pi<double>() / 2.0, openvdb::math::Y_AXIS);
        linearTransform->postTranslate(openvdb::Vec3d(xOffset, yOffset, zOffset));
        m_grid->setTransform(linearTransform);
    }

    // We auto center with just the scale
    VDBVolume(std::string filename, float scale)
    {
        openvdb::io::File vdbFile(filename);
        vdbFile.open();

        openvdb::GridBase::Ptr baseGrid = vdbFile.readGrid(vdbFile.beginName().gridName());
        vdbFile.close();

        m_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        if (m_grid->getGridClass() == openvdb::GRID_LEVEL_SET) {
            openvdb::tools::sdfToFogVolume(*m_grid);
        }

        openvdb::math::Transform::Ptr linearTransform
              = openvdb::math::Transform::createLinearTransform(scale);
        linearTransform->preRotate(-openvdb::math::pi<double>() / 2.0, openvdb::math::Y_AXIS);
        m_grid->setTransform(linearTransform);

        openvdb::math::CoordBBox bbox = m_grid->evalActiveVoxelBoundingBox();
        openvdb::Vec3d offset = m_grid->indexToWorld(bbox.max()) + m_grid->indexToWorld(bbox.min());
        offset *= 0.5;

        openvdb::math::Transform::Ptr offsetLinear
              = openvdb::math::Transform::createLinearTransform(scale);
        offsetLinear->preRotate(-openvdb::math::pi<double>() / 2.0, openvdb::math::Y_AXIS);
        offsetLinear->postTranslate(-offset + openvdb::Vec3d(8.0, 0.0, 0.0));
        m_grid->setTransform(offsetLinear);
    }

    void PrintMetadata() const
    {
        for (openvdb::MetaMap::MetaIterator iter = m_grid->beginMeta(); iter != m_grid->endMeta();
              ++iter) {
            const std::string &name = iter->first;
            openvdb::Metadata::Ptr value = iter->second;
            std::string valueAsString = value->str();
            std::cout << name << " = " << valueAsString << std::endl;
        }
    }

    void PrintWorldBB()
    {
        openvdb::math::CoordBBox bbox = m_grid->evalActiveVoxelBoundingBox();
        std::cout << m_grid->indexToWorld(bbox.min()) << ", " << m_grid->indexToWorld(bbox.max())
                  << std::endl;
    }

    openvdb::FloatGrid::Ptr AccessGrid() { return m_grid; }

   private:
    openvdb::FloatGrid::Ptr m_grid = nullptr;
};

void OutputRawData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawCombined,
      const int32_t imageWidth, const int32_t imageHeight);

class Camera {
   public:
    Camera()
        : position(0.0f, 0.0f, 0.0f)
        , fov(0.0f)
        , aspectRatio(0.0f)
        , imageWidth(0)
        , imageHeight(0)
    {
    }

    Camera(const glm::vec3 &position, const glm::vec3 &lookAt, const glm::vec3 &up, float fov,
          float aspectRatio, int imageWidth, int imageHeight)
        : position(position)
        , fov(fov)
        , aspectRatio(aspectRatio)
        , imageWidth(imageWidth)
        , imageHeight(imageHeight)
    {
        forward = glm::normalize(lookAt - position);
        right = glm::normalize(glm::cross(forward, up));
        this->up = glm::cross(right, forward);
        updateCameraVectors();
    }

    void setPosition(const glm::vec3 &newPosition)
    {
        position = newPosition;
        updateCameraVectors();
    }

    void setLookAt(const glm::vec3 &lookAt)
    {
        forward = glm::normalize(lookAt - position);
        right = glm::normalize(glm::cross(forward, up));
        up = glm::cross(right, forward);
        updateCameraVectors();
    }

    Farlor::Vector3 getPosition() const
    {
        return Farlor::Vector3(position.x, position.y, position.z);
    }

    Farlor::Vector3 getRayDirectionFromUV(float u, float v) const
    {
        const glm::vec3 norm
              = glm::normalize(lowerLeftCorner + u * horizontal + v * vertical - position);
        return Farlor::Vector3(norm.x, norm.y, norm.z);
    }

    Farlor::Vector3 getRayDirectionForPixel(int x, int y) const
    {
        const float u = (x + 0.5f) / static_cast<float>(imageWidth);
        const float v = (y + 0.5f) / static_cast<float>(imageHeight);
        return getRayDirectionFromUV(u, v);
    }

    Farlor::Vector3 getForward() const { return Farlor::Vector3(forward.x, forward.y, forward.z); }

    int getWidth() { return imageWidth; }
    int getHeight() { return imageHeight; }

   private:
    glm::vec3 position;
    glm::vec3 forward;
    glm::vec3 right;
    glm::vec3 up;
    float fov;
    float aspectRatio;
    int imageWidth;
    int imageHeight;

    glm::vec3 lowerLeftCorner;
    glm::vec3 horizontal;
    glm::vec3 vertical;

    void updateCameraVectors()
    {
        float theta = glm::radians(fov);
        float halfHeight = tan(theta / 2);
        float halfWidth = aspectRatio * halfHeight;

        lowerLeftCorner = position - halfWidth * right - halfHeight * up - forward;
        horizontal = 2 * halfWidth * right;
        vertical = 2 * halfHeight * up;
    }
};

struct NoisyCircleParams {
    // Ok, we want to kick off an experiment per pixel.
    float maxArclengthOffset = 1.0f;
    float distanceFromPlane = 1.0f;
};

Camera LoadCameraFromConfig(nlohmann::json &experimentConfig)
{
    glm::vec3 position;
    position.x = experimentConfig.at("experiment").at("camera").at("eyePos").at("x").get<float>();
    position.y = experimentConfig.at("experiment").at("camera").at("eyePos").at("y").get<float>();
    position.z = experimentConfig.at("experiment").at("camera").at("eyePos").at("z").get<float>();

    glm::vec3 lookAt;
    lookAt.x = experimentConfig.at("experiment").at("camera").at("lookAt").at("x").get<float>();
    lookAt.y = experimentConfig.at("experiment").at("camera").at("lookAt").at("y").get<float>();
    lookAt.z = experimentConfig.at("experiment").at("camera").at("lookAt").at("z").get<float>();

    glm::vec3 up;
    up.x = experimentConfig.at("experiment").at("camera").at("up").at("x").get<float>();
    up.y = experimentConfig.at("experiment").at("camera").at("up").at("y").get<float>();
    up.z = experimentConfig.at("experiment").at("camera").at("up").at("z").get<float>();

    float fovDegrees = experimentConfig.at("experiment").at("camera").at("fovDegrees").get<float>();

    int framePixelWidth
          = experimentConfig.at("experiment").at("camera").at("framePixelWidth").get<int>();
    int framePixelHeight
          = experimentConfig.at("experiment").at("camera").at("framePixelHeight").get<int>();

    Camera camera(position, lookAt, up, fovDegrees,
          static_cast<float>(framePixelWidth) / static_cast<float>(framePixelHeight),
          framePixelWidth, framePixelHeight);
    return camera;
}

NoisyCircleParams ParseExperimentSpecificParams(nlohmann::json &experimentConfig)
{
    NoisyCircleParams params;
    try {
        params.maxArclengthOffset = experimentConfig.at("experiment")
                                          .at("noisyCircleAngleIntegration")
                                          .at("maxArclengthOffset")
                                          .get<float>();
        params.distanceFromPlane = experimentConfig.at("experiment")
                                         .at("noisyCircleAngleIntegration")
                                         .at("distanceFromPlane")
                                         .get<float>();
    } catch (const std::exception &ex) {
        std::cout << "Exception: " << ex.what() << std::endl;
    }
    return params;
}

bool SaveEXR(const float *rgb, int width, int height, const char *outfilename)
{
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    // Split RGBRGBRGB... into R, G and B layer
    for (int w = 0; w < width; w++) {
        for (int h = 0; h < height; h++) {
            int invh = height - h - 1;
            images[0][w + h * width] = rgb[3 * (w + invh * width) + 0];
            images[1][w + h * width] = rgb[3 * (w + invh * width) + 1];
            images[2][w + h * width] = rgb[3 * (w + invh * width) + 2];
        }
    }

    float *image_ptr[3];
    image_ptr[0] = &(images[2].at(0));  // B
    image_ptr[1] = &(images[1].at(0));  // G
    image_ptr[2] = &(images[0].at(0));  // R

    image.images = (unsigned char **)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
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
              = TINYEXR_PIXELTYPE_FLOAT;  // pixel type of output image to be stored in
                                          // .EXR
    }

    const char *err = NULL;  // or nullptr in C++11 or later.
    int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
    if (ret != TINYEXR_SUCCESS) {
        fprintf(stderr, "Save EXR err: %s\n", err);
        FreeEXRErrorMessage(err);  // free's buffer for an error message
        std::cout << "could not save" << std::endl;
        return ret;
    }
    printf("Saved exr file. [ %s ] \n", outfilename);

    // free((void*)rgb);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
    return true;
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Call as: %s configFilename\n", argv[0]);
        return 1;
    }

    std::fstream configFile(argv[1]);
    if (!configFile.is_open()) {
        std::cout << "Failed to open: " << argv[1] << std::endl;
        return 1;
    }

    nlohmann::json experimentConfig;
    configFile >> experimentConfig;

    twisty::ExperimentRunner::ExperimentParameters experimentParams
          = twisty::ExperimentRunner::ParseExperimentParamsFromConfig(experimentConfig);

    NoisyCircleParams experimentSpecificParams = ParseExperimentSpecificParams(experimentConfig);

    std::uniform_real_distribution<float> uniformFloat(0.0f, 1.0f);

    Camera camera;
    try {
        camera = LoadCameraFromConfig(experimentConfig);
    } catch (std::exception &ex) {
        std::cout << "ex: " << ex.what() << std::endl;
    }
    const int imageWidth = camera.getWidth();
    const int imageHeight = camera.getHeight();

    std::vector<boost::multiprecision::cpp_dec_float_100> framePixels(imageWidth * imageHeight);

    std::vector<boost::multiprecision::cpp_dec_float_100> combinedPixels(imageWidth * imageHeight);

    // Experiment start time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Vector storing each runs time
    std::vector<uint64_t> runTimes;

    std::filesystem::path outputDirectoryPath = experimentParams.experimentDirPath;
    if (!std::filesystem::exists(outputDirectoryPath)) {
        std::filesystem::create_directories(outputDirectoryPath);
    }

    const Farlor::Vector3 planeNormal = Farlor::Vector3(-1.0f, 0.0f, 0.0f);
    const Farlor::Vector3 planeNormalO1 = Farlor::Vector3(0.0f, 1.0f, 0.0f);
    const Farlor::Vector3 planeNormalO2 = Farlor::Vector3(0.0f, 0.0f, 1.0f);

    // Must call once
    openvdb::initialize();

    // VDBVolume bunny("bunny.vdb", 1.0f / 75.0f);
    VDBVolume bunny(5.0f, 4.0f);
    bunny.PrintMetadata();
    bunny.PrintWorldBB();

    std::mt19937_64 rng(0);

    int64_t totalOpCount = imageWidth * imageHeight;
    int64_t currentOpCount = 0;

    for (int32_t pixelIdxY = 0; pixelIdxY < imageHeight; ++pixelIdxY) {
        for (int32_t pixelIdxX = 0; pixelIdxX < imageWidth; ++pixelIdxX) {
            const uint32_t frameIdx = (pixelIdxY * imageWidth) + pixelIdxX;

            const Farlor::Vector3 recieverPos = camera.getPosition();
            const Farlor::Vector3 recieverDir
                  = camera.getRayDirectionForPixel(pixelIdxX, pixelIdxY) * -1.0f;

            openvdb::math::Ray<double> ray(
                  openvdb::Vec3d(recieverPos.x, recieverPos.y, recieverPos.z),
                  openvdb::Vec3d(recieverDir.x, recieverDir.y, recieverDir.z));

            openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> meshIntersector(
                  *bunny.AccessGrid());


            bool hitbox = meshIntersector.setWorldRay(ray);
            if (!hitbox) {
                continue;
            }

            const float traceStepSize = 0.1f;
            const Farlor::Vector3 stepVec = recieverDir.Normalized() * traceStepSize;

            openvdb::tools::VolumeRayIntersector<openvdb::FloatGrid> intersector(meshIntersector);
            double t0 = 0.0;
            double t1 = 0.0;
            bool intersection = intersector.march(t0, t1);

            if (intersection) {
                framePixels[frameIdx] = 1.0;
            }
        }
    }

    // Experiment end time
    const auto endTime = std::chrono::high_resolution_clock::now();

    // Experiment duration
    const auto duration
          = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Experiment duration: " << duration.count() << "ms" << std::endl;
    // Experiment duration seconds
    const auto durationSeconds
          = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
    std::cout << "Experiment duration: " << durationSeconds.count() << "s" << std::endl;

    // Experiment duration minutes
    const auto durationMinutes
          = std::chrono::duration_cast<std::chrono::minutes>(endTime - startTime);
    std::cout << "Experiment duration: " << durationMinutes.count() << "m" << std::endl;

    // Average runTimes
    uint64_t totalRunTime = 0;
    for (const auto &runTime : runTimes) {
        totalRunTime += runTime;
    }
    const uint64_t averageRunTime = totalRunTime;  // / runTimes.size();

    std::cout << "Average run time: " << averageRunTime << "ms" << std::endl;
    // Avg run time in seconds
    const double averageRunTimeSeconds = averageRunTime / 1000.0;
    std::cout << "Average run time: " << averageRunTimeSeconds << "s" << std::endl;

    // Average run time in minutes
    const double averageRunTimeMinutes = averageRunTimeSeconds / 60.0;
    std::cout << "Average run time: " << averageRunTimeMinutes << "m" << std::endl;

    OutputRawData(outputDirectoryPath, framePixels, imageWidth, imageHeight);

    std::vector<float> maskValues(framePixels.size() * 3);
    for (int i = 0; i < framePixels.size(); i++) {
        maskValues[i * 3 + 0] = framePixels[i].convert_to<float>();
        maskValues[i * 3 + 1] = framePixels[i].convert_to<float>();
        maskValues[i * 3 + 2] = framePixels[i].convert_to<float>();
    }

    std::string outFilename = outputDirectoryPath.string();
    outFilename += "/" + experimentParams.experimentName + ".exr";

    if (!SaveEXR(maskValues.data(), imageWidth, imageHeight, outFilename.c_str())) {
        std::cout << "Failed to export" << std::endl;
        return 1;
    }
    std::cout << "Wrote: " << outFilename << std::endl;

    std::cout << "Experiment done" << std::endl;

    return 0;
}

void OutputRawData(const std::filesystem::path &outputDirectoryPath,
      const std::vector<boost::multiprecision::cpp_dec_float_100> &rawCombined,
      const int32_t imageWidth, const int32_t imageHeight)
{
    std::filesystem::path rawDataFilepath = outputDirectoryPath;
    rawDataFilepath /= "Combined/";
    if (!std::filesystem::exists(rawDataFilepath)) {
        std::filesystem::create_directories(rawDataFilepath);
    }

    rawDataFilepath /= "noisyCircle.dat";
    std::ofstream rawDataOutfile(rawDataFilepath.string());

    std::cout << "Combined raw Data Outfile path: " << rawDataFilepath << std::endl;

    if (!rawDataOutfile.is_open()) {
        std::cout << "Failed to create rawDataOutfile: " << rawDataFilepath.string() << std::endl;
        exit(1);
    }

    // X
    rawDataOutfile << imageWidth << " " << imageHeight << std::endl;

    for (int32_t pixelIdxY = 0; pixelIdxY < imageHeight; ++pixelIdxY) {
        for (int32_t pixelIdxX = 0; pixelIdxX < imageWidth; ++pixelIdxX) {
            const uint32_t frameIdx = (pixelIdxY * imageWidth) + pixelIdxX;
            rawDataOutfile << rawCombined[frameIdx] << " ";
        }
        rawDataOutfile << std::endl;
    }
}