#include <ExperimentRunner.h>
#include <PathWeighters.h>

#include "MathConsts.h"
#include "boost/multiprecision/cpp_dec_float.hpp"

#include <FMath/FMath.h>

#include <nlohmann/json.hpp>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Call as: " << argv[0] << " pathsDirectory" << std::endl;
        return 1;
    }

    std::string pathsDirectory(argv[1]);

    std::filesystem::path pathsDirectoryPath = pathsDirectory;
    std::cout << "pathsDirectoryPath: " << pathsDirectoryPath << std::endl;

    // All valid directories must have log 10 path weights, and five segment binary
    // Five Segment value loading
    std::filesystem::path sixSegmentAngleValuesBinaryFilename = pathsDirectoryPath;
    sixSegmentAngleValuesBinaryFilename.append("SixSegment_Binary.bdt");

    if (!std::filesystem::exists(sixSegmentAngleValuesBinaryFilename)) {
        std::cout << sixSegmentAngleValuesBinaryFilename << " file does not exist, provide one"
                  << std::endl;
        return 1;
    }

    std::ifstream sixSegmentAngleValuesFS(sixSegmentAngleValuesBinaryFilename, std::ios::binary);
    if (!sixSegmentAngleValuesFS.is_open()) {
        std::cout << "Failed to open: " << sixSegmentAngleValuesBinaryFilename << std::endl;
        return 1;
    }

    sixSegmentAngleValuesFS.seekg(0, sixSegmentAngleValuesFS.end);
    uint64_t numsixSegmentAngleBytesTotal = sixSegmentAngleValuesFS.tellg();
    sixSegmentAngleValuesFS.seekg(0, sixSegmentAngleValuesFS.beg);

    uint64_t numsixSegmentAngleBytesPerCurve = sizeof(float) * 5;
    assert((numsixSegmentAngleBytesTotal % numsixSegmentAngleBytesPerCurve) == 0
          && "File should exactly fit five values per curve");
    uint64_t numsixSegmentEntries = numsixSegmentAngleBytesTotal / numsixSegmentAngleBytesPerCurve;

    // Log 10 weight loading
    std::filesystem::path sixSegmentLog10PathWeightBinaryFilename = pathsDirectoryPath;
    sixSegmentLog10PathWeightBinaryFilename.append("Log10PathWeights_Binary.bdt");

    if (!std::filesystem::exists(sixSegmentLog10PathWeightBinaryFilename)) {
        std::cout << sixSegmentLog10PathWeightBinaryFilename << " file does not exist, provide one"
                  << std::endl;
        return 1;
    }

    std::ifstream sixSegmentLog10WeightsFS(
          sixSegmentLog10PathWeightBinaryFilename, std::ios::binary);
    if (!sixSegmentLog10WeightsFS.is_open()) {
        std::cout << "Failed to open: " << sixSegmentLog10PathWeightBinaryFilename << std::endl;
        return 1;
    }

    sixSegmentLog10WeightsFS.seekg(0, sixSegmentLog10WeightsFS.end);
    uint64_t numLog10WeightBytesTotal = sixSegmentLog10WeightsFS.tellg();
    sixSegmentLog10WeightsFS.seekg(0, sixSegmentLog10WeightsFS.beg);

    uint64_t numLog10WeightBytesPerCurve = sizeof(double);
    assert((numLog10WeightBytesTotal % numLog10WeightBytesPerCurve) == 0
          && "File should exactly fit three values per curve");
    uint64_t numLog10WeightEntries = numLog10WeightBytesTotal / numLog10WeightBytesPerCurve;

    if (numsixSegmentEntries != numLog10WeightEntries) {
        std::cout << "Different number of entries" << std::endl;
        std::cout << "numsixSegmentEntries: " << numsixSegmentEntries << std::endl;
        std::cout << "numLog10WeightEntries: " << numLog10WeightEntries << std::endl;
        return 1;
    }

    std::cout << "numsixSegmentEntries: " << numsixSegmentEntries << std::endl;
    std::cout << "numLog10WeightEntries: " << numLog10WeightEntries << std::endl;

    // TODO: Maybe make this a stream sort of operation
    std::vector<float> sixSegmentAngleValues(numsixSegmentEntries * 5);
    sixSegmentAngleValuesFS.read(
          (char *)sixSegmentAngleValues.data(), sizeof(float) * 5 * numsixSegmentEntries);

    std::vector<double> log10WeightValues(numLog10WeightEntries);
    sixSegmentLog10WeightsFS.read(
          (char *)log10WeightValues.data(), sizeof(double) * numLog10WeightEntries);

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = twisty::TwistyPi;
    const uint64_t numPhi1Vals = 64;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const uint64_t numTheta1Vals = 64;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    // Polar angle
    const float phi2Min = 0.0f;
    const float phi2Max = twisty::TwistyPi;
    const uint64_t numPhi2Vals = 64;
    const float dPhi2 = (phi2Max - phi2Min) / numPhi2Vals;

    // Azimuthal
    const float theta2Min = -twisty::TwistyPi;
    const float theta2Max = twisty::TwistyPi;
    const uint64_t numTheta2Vals = 64;
    const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

    // Azimuthal
    const float theta3Min = -twisty::TwistyPi;
    const float theta3Max = twisty::TwistyPi;
    const uint64_t numTheta3Vals = 64;
    const float dTheta3 = (theta3Max - theta3Min) / numTheta3Vals;

    std::vector<std::vector<uint64_t>> bucketCounts(numPhi1Vals);
    std::vector<std::vector<boost::multiprecision::cpp_dec_float_100>> bucketAverages(numPhi1Vals);

    const uint64_t newSize = numTheta1Vals * numPhi2Vals * numTheta2Vals * numTheta3Vals;
    for (auto &image : bucketCounts) {
        image.resize(numTheta1Vals * numPhi2Vals * numTheta2Vals * numTheta3Vals, 0);
    }

    for (auto &image : bucketAverages) {
        image.resize(numTheta1Vals * numPhi2Vals * numTheta2Vals * numTheta3Vals, 0);
    }

    uint64_t numActiveBuckets = 0;

    // Hash out the values
    for (size_t idx = 0; idx < sixSegmentAngleValues.size(); idx += 5) {
        const float phi1 = sixSegmentAngleValues[idx + 0];
        const float theta1 = sixSegmentAngleValues[idx + 1];
        const float phi2 = sixSegmentAngleValues[idx + 2];
        const float theta2 = sixSegmentAngleValues[idx + 3];
        const float theta3 = sixSegmentAngleValues[idx + 4];

        const float phi1Dist = (phi1 - phi1Min);
        uint64_t phi1Idx = phi1Dist / dPhi1;
        phi1Idx = std::min(phi1Idx, numPhi1Vals - 1);

        const float theta1Dist = (theta1 - theta1Min);
        uint64_t theta1Idx = theta1Dist / dTheta1;
        theta1Idx = std::min(theta1Idx, numTheta1Vals - 1);

        const float phi2Dist = (phi2 - phi2Min);
        uint64_t phi2Idx = phi2Dist / dPhi2;
        phi2Idx = std::min(phi2Idx, numPhi2Vals - 1);

        const float theta2Dist = (theta2 - theta2Min);
        uint64_t theta2Idx = theta2Dist / dTheta2;
        theta2Idx = std::min(theta2Idx, numTheta2Vals - 1);

        const float theta3Dist = (theta3 - theta3Min);
        uint64_t theta3Idx = theta3Dist / dTheta3;
        theta3Idx = std::min(theta3Idx, numTheta3Vals - 1);

        size_t bucketIdx = theta1Idx * (numPhi2Vals * numTheta2Vals * numTheta3Vals)
              + phi2Idx * (numTheta2Vals * numTheta3Vals) + (theta2Idx * numTheta3Vals) + theta3Idx;

        if (bucketCounts[phi1Idx][bucketIdx] == 0) {
            numActiveBuckets++;
        }

        // Increament Heatmap
        bucketCounts[phi1Idx][bucketIdx]++;
        bucketAverages[phi1Idx][bucketIdx]
              += boost::multiprecision::cpp_dec_float_100(log10WeightValues[idx / 5]);
    }

    const uint64_t numtotalBuckets = numPhi1Vals * numTheta1Vals * numTheta2Vals;

    std::cout << "Num active buckets: " << numActiveBuckets << std::endl;
    std::cout << "Num total buckets: " << numtotalBuckets << std::endl;

    std::cout << "Percent of space covered: " << double(numActiveBuckets) / double(numtotalBuckets)
              << std::endl;

    boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;
    size_t validBucketCount = 0;
    for (size_t idx = 0; idx < bucketCounts.size(); idx++) {
        for (size_t imageIdx = 0; imageIdx < bucketCounts[idx].size(); imageIdx++) {
            if (bucketCounts[idx][imageIdx] > 0) {
                validBucketCount++;
                // bucketAverages[idx][imageIdx] /= bucketCounts[idx][imageIdx];
                finalResult += bucketAverages[idx][imageIdx];
            }
        }
    }

    if (validBucketCount > 0) {
        finalResult /= validBucketCount;
    }
    std::cout << "Final result: " << finalResult << std::endl;

    std::cout << "Done" << std::endl;
}