#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define DEBUG_PRINT 0

bool SaveEXR(const float *rgb, int width, int height, const char *outfilename);
bool SaveEXR(std::vector<float> values, int imgWidth, int imgHeight, float scaleFactor,
      const char *outfilename);