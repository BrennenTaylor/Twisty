// This test tests the validity of our combined weight values for the given weight lookup table.
// We randomly select curvatures from the table, and combine these values
// 1. With the combined weights values
// 2. With the BigFloat library we use
// 3. With the Sorted BigFloat library from smallest to biggest

// We parameterize to allow exploration over different weight tables, as well as different numbers of segments
// i.e. differet order of magnitude ranges of the final combined values

#include <cmath>

#include "Curve.h"
#include "ExperimentRunner.h"
#include "FMath/Quaternion.h"
#include "FullExperimentRunnerOptimalPerturb.h"
#include "boost/multiprecision/cpp_dec_float.hpp"
#include <random>
#include <stdexcept>

#include "Bootstrapper.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "boost/multiprecision/detail/default_ops.hpp"

#include <FMath/Vector3.h>

#include <nlohmann/json.hpp>

#include <chrono>
#include <iostream>
#include <memory>
#include <filesystem>

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

    std::filesystem::path outputDirectoryPath = std::filesystem::current_path();
    outputDirectoryPath.append(experimentParams.experimentName);
    if (!std::filesystem::exists(outputDirectoryPath)) {
        std::filesystem::create_directory(outputDirectoryPath);
    }

    // Lets test uniform random directions on sampling hemisphere
    uint32_t targetSeed = experimentConfig["experiment"]["stressTestCombinedWeights"]["seed"];
    twisty::ProcessRandomSeed(targetSeed);

    const uint32_t numPaths
          = experimentConfig["experiment"]["stressTestCombinedWeights"]["numPaths"];

    const float arclength
          = experimentConfig["experiment"]["stressTestCombinedWeights"]["arclength"];
    const float ds = arclength / experimentParams.numSegmentsPerCurve;

    std::filesystem::path experimentDirPath
          = std::filesystem::path(experimentParams.experimentDirPath);
    if (!experimentParams.perExperimentDirSubfolder.empty()) {
        experimentDirPath /= experimentParams.perExperimentDirSubfolder;
    }
    if (!std::filesystem::exists(experimentDirPath)) {
        std::filesystem::create_directories(experimentDirPath);
    }


    std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;
    if (experimentParams.weightingParameters.weightingMethod
          == twisty::WeightingMethod::SimplifiedModel) {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(
              experimentParams.weightingParameters, ds);
    } else {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
              experimentParams.weightingParameters, ds);
    }
    lookupEvaluator->ExportValues(experimentDirPath.string());
    assert(lookupEvaluator);
    twisty::PathWeighting::BaseWeightLookupTable &weightingIntegralsRawPointer = (*lookupEvaluator);

    // Minimum Possible Segment Weight
    // Maximum Possible Segment Weight

    std::cout << "Min possible segment weight: "
              << weightingIntegralsRawPointer.GetMinSegmentWeight() << std::endl;
    std::cout << "Max possible segment weight: "
              << weightingIntegralsRawPointer.GetMaxSegmentWeight() << std::endl;

    boost::multiprecision::cpp_dec_float_100 minPossibleCurveWeight
          = weightingIntegralsRawPointer.GetMinSegmentWeight();
    for (int i = 1; i < experimentParams.numSegmentsPerCurve; i++) {
        minPossibleCurveWeight *= weightingIntegralsRawPointer.GetMinSegmentWeight();
    }

    boost::multiprecision::cpp_dec_float_100 maxPossibleCurveWeight
          = weightingIntegralsRawPointer.GetMaxSegmentWeight();
    for (int i = 1; i < experimentParams.numSegmentsPerCurve; i++) {
        maxPossibleCurveWeight *= weightingIntegralsRawPointer.GetMaxSegmentWeight();
    }

    std::cout << "Min possible curve weight: " << minPossibleCurveWeight << std::endl;
    std::cout << "Max possible curve weight: " << maxPossibleCurveWeight << std::endl;

    std::cout << "Largest difference in curve magnitude: "
              << (maxPossibleCurveWeight - minPossibleCurveWeight) << std::endl;

    std::mt19937 randomGenerator(targetSeed);
    std::uniform_int_distribution<int> weightIdxSampleDist(
          0, weightingIntegralsRawPointer.AccessLookupTable().size() - 1);

    boost::multiprecision::cpp_dec_float_100 test1_min_absolute_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test2_min_absolute_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test3_min_absolute_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test4_min_absolute_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test5_min_absolute_error = 100.0;

    boost::multiprecision::cpp_dec_float_100 test1_max_absolute_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test2_max_absolute_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test3_max_absolute_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test4_max_absolute_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test5_max_absolute_error = -100.0;

    boost::multiprecision::cpp_dec_float_100 test1_min_relative_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test2_min_relative_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test3_min_relative_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test4_min_relative_error = 100.0;
    boost::multiprecision::cpp_dec_float_100 test5_min_relative_error = 100.0;

    boost::multiprecision::cpp_dec_float_100 test1_max_relative_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test2_max_relative_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test3_max_relative_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test4_max_relative_error = -100.0;
    boost::multiprecision::cpp_dec_float_100 test5_max_relative_error = -100.0;

    // Ok, we want to generate a number of weighted paths
    // Note, these do NOT need to be valid paths
    std::vector<float> weights;
    weights.resize(experimentParams.numSegmentsPerCurve - 1);
    for (uint32_t pathIdx = 0; pathIdx < numPaths; pathIdx++) {
        // Randomly sample table.
        double pathWeightDoubles = 1.0;
        double pathWeightLog10Doubles = 0.0;
        boost::multiprecision::cpp_dec_float_100 pathWeightBigFloat = 1.0;
        boost::multiprecision::cpp_dec_float_100 pathWeightLog10BigFloat = 0.0;

        boost::multiprecision::cpp_dec_float_50 pathWeightSmallerBigFloat = 1.0;
        boost::multiprecision::cpp_dec_float_50 pathWeightLog10SmallerBigFloat = 0.0;

        for (uint32_t weightIdx = 0; weightIdx < weights.size(); weightIdx++) {
            weights[weightIdx] = weightingIntegralsRawPointer
                                       .AccessLookupTable()[weightIdxSampleDist(randomGenerator)];
            pathWeightDoubles *= static_cast<double>(weights[weightIdx]);
            pathWeightLog10Doubles += static_cast<double>(std::log10(weights[weightIdx]));

            pathWeightBigFloat *= boost::multiprecision::cpp_dec_float_100(weights[weightIdx]);
            pathWeightLog10BigFloat += boost::multiprecision::log10(
                  boost::multiprecision::cpp_dec_float_100(weights[weightIdx]));

            pathWeightSmallerBigFloat
                  *= boost::multiprecision::cpp_dec_float_50(weights[weightIdx]);
            pathWeightLog10SmallerBigFloat += boost::multiprecision::log10(
                  boost::multiprecision::cpp_dec_float_50(weights[weightIdx]));
        }

        // Convert all to big float, decompress, and calculate error
        const boost::multiprecision::cpp_dec_float_100 groundTruth = pathWeightBigFloat;

        const boost::multiprecision::cpp_dec_float_100 test1_pathWeightDoubles
              = boost::multiprecision::cpp_dec_float_100(pathWeightDoubles);
        const boost::multiprecision::cpp_dec_float_100 test2_pathWeightLog10Doubles
              = boost::multiprecision::pow(
                    10.0, boost::multiprecision::cpp_dec_float_100(pathWeightLog10Doubles));

        const boost::multiprecision::cpp_dec_float_100 test3_pathWeightLog10BigFloat
              = boost::multiprecision::pow(10.0, pathWeightLog10BigFloat);

        const boost::multiprecision::cpp_dec_float_100 test4_pathWeightSmallerBigFloat
              = boost::multiprecision::cpp_dec_float_100(pathWeightSmallerBigFloat);

        const boost::multiprecision::cpp_dec_float_100 test5_pathWeightLog10SmallerBigFloat
              = boost::multiprecision::pow(10.0, pathWeightLog10SmallerBigFloat);


        const boost::multiprecision::cpp_dec_float_100 test1_absolute_error
              = boost::multiprecision::abs(groundTruth - test1_pathWeightDoubles);
        const boost::multiprecision::cpp_dec_float_100 test1_relative_error
              = test1_absolute_error / groundTruth;

        const boost::multiprecision::cpp_dec_float_100 test2_absolute_error
              = boost::multiprecision::abs(groundTruth - test2_pathWeightLog10Doubles);
        const boost::multiprecision::cpp_dec_float_100 test2_relative_error
              = test2_absolute_error / groundTruth;

        const boost::multiprecision::cpp_dec_float_100 test3_absolute_error
              = boost::multiprecision::abs(groundTruth - test3_pathWeightLog10BigFloat);
        const boost::multiprecision::cpp_dec_float_100 test3_relative_error
              = test3_absolute_error / groundTruth;

        const boost::multiprecision::cpp_dec_float_100 test4_absolute_error
              = boost::multiprecision::abs(groundTruth - test4_pathWeightSmallerBigFloat);
        const boost::multiprecision::cpp_dec_float_100 test4_relative_error
              = test4_absolute_error / groundTruth;

        const boost::multiprecision::cpp_dec_float_100 test5_absolute_error
              = boost::multiprecision::abs(groundTruth - test5_pathWeightLog10SmallerBigFloat);
        const boost::multiprecision::cpp_dec_float_100 test5_relative_error
              = test5_absolute_error / groundTruth;

        if (test1_absolute_error < test1_min_absolute_error)
            test1_min_absolute_error = test1_absolute_error;

        if (test2_absolute_error < test2_min_absolute_error)
            test2_min_absolute_error = test2_absolute_error;

        if (test3_absolute_error < test3_min_absolute_error)
            test3_min_absolute_error = test3_absolute_error;

        if (test4_absolute_error < test4_min_absolute_error)
            test4_min_absolute_error = test4_absolute_error;

        if (test5_absolute_error < test5_min_absolute_error)
            test5_min_absolute_error = test5_absolute_error;


        if (test1_absolute_error > test1_max_absolute_error)
            test1_max_absolute_error = test1_absolute_error;

        if (test2_absolute_error > test2_max_absolute_error)
            test2_max_absolute_error = test2_absolute_error;

        if (test3_absolute_error > test3_max_absolute_error)
            test3_max_absolute_error = test3_absolute_error;

        if (test4_absolute_error > test4_max_absolute_error)
            test4_max_absolute_error = test4_absolute_error;

        if (test5_absolute_error > test5_max_absolute_error)
            test5_max_absolute_error = test5_absolute_error;


        if (test1_relative_error < test1_min_relative_error)
            test1_min_relative_error = test1_relative_error;

        if (test2_relative_error < test2_min_relative_error)
            test2_min_relative_error = test2_relative_error;

        if (test3_relative_error < test3_min_relative_error)
            test3_min_relative_error = test3_relative_error;

        if (test4_relative_error < test4_min_relative_error)
            test4_min_relative_error = test4_relative_error;

        if (test5_relative_error < test5_min_relative_error)
            test5_min_relative_error = test5_relative_error;


        if (test1_relative_error > test1_max_relative_error)
            test1_max_relative_error = test1_relative_error;

        if (test2_relative_error > test2_max_relative_error)
            test2_max_relative_error = test2_relative_error;

        if (test3_relative_error > test3_max_relative_error)
            test3_max_relative_error = test3_relative_error;

        if (test4_relative_error > test4_max_relative_error)
            test4_max_relative_error = test4_relative_error;

        if (test5_relative_error > test5_max_relative_error)
            test5_max_relative_error = test5_relative_error;

        // std::cout << "Path: " << pathIdx << "\n";
        // std::cout << "\tGround Truth: " << groundTruth << "\n";
        // std::cout << "\tTest 1 Value: " << test1_pathWeightDoubles << "\n";
        // std::cout << "\tTest 1 Absolute Error: " << test1_absolute_error << "\n";
        // std::cout << "\tTest 1 Relative Error: " << test1_relative_error << "\n";
        // std::cout << "\tTest 2 Value: " << test2_pathWeightLog10Doubles << "\n";
        // std::cout << "\tTest 2 Absolute Error: " << test2_absolute_error << "\n";
        // std::cout << "\tTest 2 Relative Error: " << test2_relative_error << "\n";
        // std::cout << "\tTest 3 Value: " << test3_pathWeightLog10BigFloat << "\n";
        // std::cout << "\tTest 3 Absolute Error: " << test3_absolute_error << "\n";
        // std::cout << "\tTest 3 Relative Error: " << test3_relative_error << "\n";

        // std::cout << "\tTest 4 Value: " << test4_pathWeightSmallerBigFloat << "\n";
        // std::cout << "\tTest 4 Absolute Error: " << test4_absolute_error << "\n";
        // std::cout << "\tTest 4 Relative Error: " << test4_relative_error << "\n";

        // std::cout << "\tTest 5 Value: " << test5_pathWeightLog10SmallerBigFloat << "\n";
        // std::cout << "\tTest 5 Absolute Error: " << test5_absolute_error << "\n";
        // std::cout << "\tTest 5 Relative Error: " << test5_relative_error << "\n";
    }

    std::cout << "Double: "
              << "\n";
    std::cout << "Test 1 Min Absolute Error: " << test1_min_absolute_error << "\n";
    std::cout << "Test 1 Max Absolute Error: " << test1_max_absolute_error << "\n";
    std::cout << "Test 1 Min Relative Error: " << test1_min_relative_error << "\n";
    std::cout << "Test 1 Max Relative Error: " << test1_max_relative_error << "\n";

    std::cout << "Log10 Double: "
              << "\n";
    std::cout << "Test 2 Min Absolute Error: " << test2_min_absolute_error << "\n";
    std::cout << "Test 2 Max Absolute Error: " << test2_max_absolute_error << "\n";
    std::cout << "Test 2 Min Relative Error: " << test2_min_relative_error << "\n";
    std::cout << "Test 2 Max Relative Error: " << test2_max_relative_error << "\n";

    std::cout << "Log10 boost::multiprecision::cpp_float_100: "
              << "\n";
    std::cout << "Test 3 Min Absolute Error: " << test3_min_absolute_error << "\n";
    std::cout << "Test 3 Max Absolute Error: " << test3_max_absolute_error << "\n";
    std::cout << "Test 3 Min Relative Error: " << test3_min_relative_error << "\n";
    std::cout << "Test 3 Max Relative Error: " << test3_max_relative_error << "\n";

    std::cout << "boost::multiprecision::cpp_float_50: "
              << "\n";
    std::cout << "Test 4 Min Absolute Error: " << test4_min_absolute_error << "\n";
    std::cout << "Test 4 Max Absolute Error: " << test4_max_absolute_error << "\n";
    std::cout << "Test 4 Min Relative Error: " << test4_min_relative_error << "\n";
    std::cout << "Test 4 Max Relative Error: " << test4_max_relative_error << "\n";

    std::cout << "Log 10 boost::multiprecision::cpp_float_50: "
              << "\n";
    std::cout << "Test 5 Min Absolute Error: " << test5_min_absolute_error << "\n";
    std::cout << "Test 5 Max Absolute Error: " << test5_max_absolute_error << "\n";
    std::cout << "Test 5 Min Relative Error: " << test5_min_relative_error << "\n";
    std::cout << "Test 5 Max Relative Error: " << test5_max_relative_error << "\n";


    std::cout << "Experiment done" << std::endl;

    return 0;
}
