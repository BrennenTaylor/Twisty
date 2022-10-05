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
    std::filesystem::path fiveSegmentAngleValuesBinaryFilename = pathsDirectoryPath;
    fiveSegmentAngleValuesBinaryFilename.append("FiveSegment_Binary.bdt");

    if (!std::filesystem::exists(fiveSegmentAngleValuesBinaryFilename)) {
        std::cout << fiveSegmentAngleValuesBinaryFilename << " file does not exist, provide one"
                  << std::endl;
        return 1;
    }

    std::ifstream fiveSegmentAngleValuesFS(fiveSegmentAngleValuesBinaryFilename, std::ios::binary);
    if (!fiveSegmentAngleValuesFS.is_open()) {
        std::cout << "Failed to open: " << fiveSegmentAngleValuesBinaryFilename << std::endl;
        return 1;
    }

    fiveSegmentAngleValuesFS.seekg(0, fiveSegmentAngleValuesFS.end);
    uint64_t numFiveSegmentAngleBytesTotal = fiveSegmentAngleValuesFS.tellg();
    fiveSegmentAngleValuesFS.seekg(0, fiveSegmentAngleValuesFS.beg);

    uint64_t numFiveSegmentAngleBytesPerCurve = sizeof(Farlor::Vector3);
    assert((numFiveSegmentAngleBytesTotal % numFiveSegmentAngleBytesPerCurve) == 0
          && "File should exactly fit three values per curve");
    uint64_t numFiveSegmentEntries
          = numFiveSegmentAngleBytesTotal / numFiveSegmentAngleBytesPerCurve;

    // Log 10 weight loading
    std::filesystem::path fiveSegmentLog10PathWeightBinaryFilename = pathsDirectoryPath;
    fiveSegmentLog10PathWeightBinaryFilename.append("Log10PathWeights_Binary.bdt");

    if (!std::filesystem::exists(fiveSegmentLog10PathWeightBinaryFilename)) {
        std::cout << fiveSegmentLog10PathWeightBinaryFilename << " file does not exist, provide one"
                  << std::endl;
        return 1;
    }

    std::ifstream fiveSegmentLog10WeightsFS(
          fiveSegmentLog10PathWeightBinaryFilename, std::ios::binary);
    if (!fiveSegmentLog10WeightsFS.is_open()) {
        std::cout << "Failed to open: " << fiveSegmentLog10PathWeightBinaryFilename << std::endl;
        return 1;
    }

    fiveSegmentLog10WeightsFS.seekg(0, fiveSegmentLog10WeightsFS.end);
    uint64_t numLog10WeightBytesTotal = fiveSegmentLog10WeightsFS.tellg();
    fiveSegmentLog10WeightsFS.seekg(0, fiveSegmentLog10WeightsFS.beg);

    uint64_t numLog10WeightBytesPerCurve = sizeof(double);
    assert((numLog10WeightBytesTotal % numLog10WeightBytesPerCurve) == 0
          && "File should exactly fit three values per curve");
    uint64_t numLog10WeightEntries = numLog10WeightBytesTotal / numLog10WeightBytesPerCurve;

    if (numFiveSegmentEntries != numLog10WeightEntries) {
        std::cout << "Different number of entries" << std::endl;
        std::cout << "numFiveSegmentEntries: " << numFiveSegmentEntries << std::endl;
        std::cout << "numLog10WeightEntries: " << numLog10WeightEntries << std::endl;
        return 1;
    }

    std::cout << "numFiveSegmentEntries: " << numFiveSegmentEntries << std::endl;
    std::cout << "numLog10WeightEntries: " << numLog10WeightEntries << std::endl;

    // TODO: Maybe make this a stream sort of operation
    std::vector<Farlor::Vector3> fiveSegmentAngleValues(numFiveSegmentEntries);
    fiveSegmentAngleValuesFS.read(
          (char *)fiveSegmentAngleValues.data(), sizeof(Farlor::Vector3) * numFiveSegmentEntries);

    std::vector<double> log10WeightValues(numLog10WeightEntries);
    fiveSegmentLog10WeightsFS.read(
          (char *)log10WeightValues.data(), sizeof(double) * numLog10WeightEntries);

    // Polar angle
    const float phi1Min = 0.0f;
    const float phi1Max = twisty::TwistyPi;
    const uint32_t numPhi1Vals = 256;
    const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

    // Azimuthal
    const float theta1Min = -twisty::TwistyPi;
    const float theta1Max = twisty::TwistyPi;
    const uint32_t numTheta1Vals = 256;
    const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

    // Azimuthal
    const float theta2Min = -twisty::TwistyPi;
    const float theta2Max = twisty::TwistyPi;
    const uint32_t numTheta2Vals = 256;
    const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

    std::vector<std::vector<uint64_t>> bucketCounts(numPhi1Vals);
    std::vector<std::vector<boost::multiprecision::cpp_dec_float_100>> bucketAverages(numPhi1Vals);

    for (auto &image : bucketCounts) {
        image.resize(numTheta1Vals * numTheta2Vals, 0);
    }

    for (auto &image : bucketAverages) {
        image.resize(numTheta1Vals * numTheta2Vals, 0);
    }

    uint64_t numActiveBuckets = 0;

    // Hash out the values
    for (size_t idx = 0; idx < fiveSegmentAngleValues.size(); idx++) {
        const Farlor::Vector3 &curveAngles = fiveSegmentAngleValues[idx];
        const float phi1 = curveAngles.x;
        const float theta1 = curveAngles.y;
        const float theta2 = curveAngles.z;

        const float phiDist = (phi1 - phi1Min);
        uint32_t phiIdx = phiDist / dPhi1;
        phiIdx = std::min(phiIdx, numPhi1Vals - 1);

        const float theta1Dist = (theta1 - theta1Min);
        uint32_t theta1Idx = theta1Dist / dTheta1;
        theta1Idx = std::min(theta1Idx, numTheta1Vals - 1);

        const float theta2Dist = (theta2 - theta2Min);
        uint32_t theta2Idx = theta2Dist / dTheta2;
        theta2Idx = std::min(theta2Idx, numTheta2Vals - 1);

        // std::cout << phiIdx << ", " << theta1Idx << ", " << theta2Idx << std::endl;

        if (bucketCounts[phiIdx][theta1Idx * numTheta2Vals + theta2Idx] == 0) {
            numActiveBuckets++;
        }

        // Increament Heatmap
        bucketCounts[phiIdx][theta1Idx * numTheta2Vals + theta2Idx]++;
        bucketAverages[phiIdx][theta1Idx * numTheta2Vals + theta2Idx]
              += boost::multiprecision::cpp_dec_float_100(log10WeightValues[idx]);
    }

    const uint64_t numtotalBuckets = numPhi1Vals * numTheta1Vals * numTheta2Vals;

    std::cout << "Num active buckets: " << numActiveBuckets << std::endl;
    std::cout << "Num total buckets: " << numtotalBuckets << std::endl;

    std::cout << "Percent of space covered: "
              << double(numActiveBuckets) / double(numtotalBuckets) * 100.0f << std::endl;

    boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;
    size_t validBucketCount = 0;
    for (size_t idx = 0; idx < bucketCounts.size(); idx++) {
        for (size_t imageIdx = 0; imageIdx < bucketCounts[idx].size(); imageIdx++) {
            if (bucketCounts[idx][imageIdx] > 0) {
                validBucketCount++;
                bucketAverages[idx][imageIdx]
                      = bucketAverages[idx][imageIdx] / bucketCounts[idx][imageIdx];
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