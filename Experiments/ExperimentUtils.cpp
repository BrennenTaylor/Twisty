#include "ExperimentUtils.h"

#pragma once

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

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

bool SaveEXR(std::vector<float> values, int imgWidth, int imgHeight, float scaleFactor,
      const char *outfilename)
{
    std::vector<float> pixelData(imgWidth * imgHeight * 3);
    for (uint32_t y = 0; y < imgHeight; ++y) {
        for (uint32_t x = 0; x < imgWidth; ++x) {
            float dataValue = values[x + (y * imgWidth)];
#if DEBUG_PRINT
            printf("(%d, %d): %f\n", x, y, dataValue);
#endif
            dataValue *= scaleFactor;

            pixelData[y * imgWidth * 3 + x * 3 + 0] = dataValue;
            pixelData[y * imgWidth * 3 + x * 3 + 1] = dataValue;
            pixelData[y * imgWidth * 3 + x * 3 + 2] = dataValue;
        }
    }

    if (!SaveEXR(pixelData.data(), imgWidth, imgHeight, outfilename)) {
        std::cout << "Failed to export" << std::endl;
        return 1;
    }
    return 0;
}