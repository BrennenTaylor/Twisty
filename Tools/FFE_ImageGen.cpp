#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

bool SaveEXR(const float *rgb, int width, int height, const char *outfilename) {

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
  for (int i = 0; i < width * height; i++) {
    // std::cout << "br" << std::endl;
    // std::cout << rgb[3 * i + 0] << std::endl;
    // std::cout << rgb[3 * i + 1] << std::endl;
    // std::cout << rgb[3 * i + 2] << std::endl;

    images[0][i] = rgb[3 * i + 0]; // 255.0f;
    images[1][i] = rgb[3 * i + 1]; // 255.0f;
    images[2][i] = rgb[3 * i + 2]; // 255.0f;
  }

  float *image_ptr[3];
  image_ptr[0] = &(images[2].at(0)); // B
  image_ptr[1] = &(images[1].at(0)); // G
  image_ptr[2] = &(images[0].at(0)); // R

  image.images = (unsigned char **)image_ptr;
  image.width = width;
  image.height = height;

  header.num_channels = 3;
  header.channels =
      (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
  // Must be (A)BGR order, since most of EXR viewers expect this channel order.
  strncpy(header.channels[0].name, "B", 255);
  header.channels[0].name[strlen("B")] = '\0';
  strncpy(header.channels[1].name, "G", 255);
  header.channels[1].name[strlen("G")] = '\0';
  strncpy(header.channels[2].name, "R", 255);
  header.channels[2].name[strlen("R")] = '\0';

  header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
  header.requested_pixel_types =
      (int *)malloc(sizeof(int) * header.num_channels);
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] =
        TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
    header.requested_pixel_types[i] =
        TINYEXR_PIXELTYPE_FLOAT; // pixel type of output image to be stored in
                                 // .EXR
  }

  const char *err = NULL; // or nullptr in C++11 or later.
  int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err); // free's buffer for an error message
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

int main(int argc, char *argv[]) {
  if (argc < 6) {
    printf("Call as: %s imgWidth imgHeight scaleF inFilename outFilename",
           argv[0]);
    return 0;
  }

  uint32_t imgWidth = std::stoi(argv[1]);
  uint32_t imgHeight = std::stoi(argv[2]);
  float imgScaleFactor = std::stof(argv[3]);

  std::string inFilename(argv[4]);
  std::string outFilename(argv[5]);
  outFilename += ".exr";
  std::ifstream inputDataStream(inFilename);

  // Then, we write out the animation file
  // This could be pulled out to a utility function, but keeping it here is fine
  // for now.
  std::vector<float> pixelData(imgWidth * imgHeight * 3);

  for (uint32_t x = 0; x < imgWidth; ++x) {
    for (uint32_t y = 0; y < imgHeight; ++y) {
      float dataValue = 0.0f;
      inputDataStream >> dataValue;

      dataValue *= imgScaleFactor;

      pixelData[y * imgWidth * 3 + x * 3 + 0] = dataValue;
      pixelData[y * imgWidth * 3 + x * 3 + 1] = dataValue;
      pixelData[y * imgWidth * 3 + x * 3 + 2] = dataValue;
    }
  }

  if (!SaveEXR(pixelData.data(), imgWidth, imgHeight, outFilename.c_str())) {
    std::cout << "Failed to export" << std::endl;
    return 1;
  }

  std::cout << "Done" << std::endl;

  return 0;
}