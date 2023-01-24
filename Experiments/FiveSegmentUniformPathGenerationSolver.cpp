#include <ExperimentRunner.h>
#include <PathWeighters.h>
#include "CombinedWeightUtils.h"

#include "MathConsts.h"
#include "boost/multiprecision/detail/default_ops.hpp"

#include <FMath/FMath.h>

#include <omp.h>

#include <nlohmann/json.hpp>
#include <string>
#include <random>

const float PI = 3.14159265358979323846f;

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(
      const nlohmann::json &experimentConfig)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Values loaded from the config file
    experimentParams.numPathsInExperiment
          = experimentConfig["experiment"]["experimentParams"]["pathsToGenerate"];

    experimentParams.numPathsToSkip
          = experimentConfig["experiment"]["experimentParams"]["pathsToSkip"];
    experimentParams.experimentName = experimentConfig["experiment"]["experimentParams"]["name"];
    experimentParams.experimentDirPath
          = experimentConfig["experiment"]["experimentParams"]["experimentDir"];
    experimentParams.experimentDirPath += "/" + experimentParams.experimentName;
    experimentParams.experimentDirPath += "/" + twisty::GetCurrentTimeForFileName() + "/";

    experimentParams.maxPerturbThreads
          = experimentConfig["experiment"]["experimentParams"]["maxPerturbThreads"];
    experimentParams.maxWeightThreads
          = experimentConfig["experiment"]["experimentParams"]["maxWeightThreads"];

    experimentParams.outputBigFloatWeights
          = experimentConfig["experiment"]["experimentParams"]["outputBigFloatWeights"];
    experimentParams.outputPathBatches
          = experimentConfig["experiment"]["experimentParams"]["outputPathBatches"];
    experimentParams.useGpu = experimentConfig["experiment"]["experimentParams"]["useGpu"];

    experimentParams.numSegmentsPerCurve
          = experimentConfig["experiment"]["experimentParams"]["numSegments"];

    // Seeds
    experimentParams.bootstrapSeed
          = experimentConfig["experiment"]["experimentParams"]["random"]["bootstrapSeed"];
    experimentParams.curvePurturbSeed
          = experimentConfig["experiment"]["experimentParams"]["random"]["perturbSeed"];

    // Weighting parameter stuff
    int weightFunction
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["weightFunction"];
    switch (weightFunction) {
        // Radiative Transfer weight function
        case 0: {
            experimentParams.weightingParameters.weightingMethod
                  = twisty::WeightingMethod::RadiativeTransfer;
        } break;

        // Simplified Model
        case 1: {
            experimentParams.weightingParameters.weightingMethod
                  = twisty::WeightingMethod::SimplifiedModel;
        } break;

        // Default to the simplified model
        default: {
            experimentParams.weightingParameters.weightingMethod
                  = twisty::WeightingMethod::RadiativeTransfer;
        } break;
    }

    // Perturb method stuff
    int perturbMethod = experimentConfig["experiment"]["experimentParams"]["perturbMethod"];
    switch (perturbMethod) {
        // Simplified Model
        case 1: {
            experimentParams.perturbMethod
                  = twisty::ExperimentRunner::PerturbMethod::GeometricMinCurvature;
        } break;

        // Simplified Model
        case 2: {
            experimentParams.perturbMethod
                  = twisty::ExperimentRunner::PerturbMethod::GeometricCombined;
        } break;

        // Default to the simplified model
        case 0:
        default: {
            experimentParams.perturbMethod
                  = twisty::ExperimentRunner::PerturbMethod::GeometricRandom;
        } break;
    }

    // Weighting parameter stuff
    experimentParams.weightingParameters.mu
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["mu"];
    experimentParams.weightingParameters.eps
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["eps"];
    experimentParams.weightingParameters.numStepsInt
          = (int)experimentConfig["experiment"]["experimentParams"]["weighting"]["numStepsInt"];
    experimentParams.weightingParameters.numCurvatureSteps = (int)
          experimentConfig["experiment"]["experimentParams"]["weighting"]["numCurvatureSteps"];
    experimentParams.weightingParameters.absorbtion
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["absorbtion"];

    experimentParams.weightingParameters.scatter
          = experimentConfig["experiment"]["experimentParams"]["weighting"]["scatter"];

    // TODO: Should these be configurable in the file?
    experimentParams.weightingParameters.minBound = 0.0f;
    experimentParams.weightingParameters.maxBound
          = 10.0f / experimentParams.weightingParameters.eps;

    return experimentParams;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
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
          = ParseExperimentParamsFromConfig(experimentConfig);
    assert((experimentParams.numSegmentsPerCurve == 5)
          && "Must only target 5 segment curve configurations");

    if (!std::filesystem::exists(experimentParams.experimentDirPath)) {
        std::filesystem::create_directories(experimentParams.experimentDirPath);
    }
    std::cout << "experimentDirPath: " << experimentParams.experimentDirPath << std::endl;
    const std::string experimentCfgCopyFilename = std::string(experimentParams.experimentDirPath)
          + "/" + experimentParams.experimentName + ".json";

    if (!std::filesystem::exists(experimentCfgCopyFilename)) {
        std::filesystem::copy_file(argv[1], experimentCfgCopyFilename,
              std::filesystem::copy_options::overwrite_existing);
    }

    std::ofstream resultsOFS(experimentParams.experimentDirPath + "/Results.txt");
    if (!resultsOFS.is_open()) {
        std::cout << "Failed to open results file: " << experimentParams.experimentDirPath
                  << "/Results.txt" << std::endl;
        return 1;
    }

    twisty::PerturbUtils::BoundaryConditions experimentGeometry;
    {
        float x = experimentConfig["experiment"]["basicExperiment"]["startPos"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["startPos"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["startPos"][2];
        experimentGeometry.m_startPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["basicExperiment"]["startDir"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["startDir"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["startDir"][2];
        experimentGeometry.m_startDir = Farlor::Vector3(x, y, z).Normalized();
    }

    {
        float x = experimentConfig["experiment"]["basicExperiment"]["endPos"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["endPos"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["endPos"][2];
        experimentGeometry.m_endPos = Farlor::Vector3(x, y, z);
    }
    {
        float x = experimentConfig["experiment"]["basicExperiment"]["endDir"][0];
        float y = experimentConfig["experiment"]["basicExperiment"]["endDir"][1];
        float z = experimentConfig["experiment"]["basicExperiment"]["endDir"][2];
        experimentGeometry.m_endDir = Farlor::Vector3(x, y, z).Normalized();
    }
    // Force to a value
    experimentGeometry.arclength = experimentParams.arclength
          = experimentConfig["experiment"]["basicExperiment"]["arclength"];
    std::cout << "Arclength: " << experimentGeometry.arclength << std::endl;

    const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

    const uint64_t numExperimentPaths = experimentParams.numPathsInExperiment;

    std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;
    if (experimentParams.weightingParameters.weightingMethod
          == twisty::WeightingMethod::SimplifiedModel) {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(
              experimentParams.weightingParameters, ds);
    } else {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
              experimentParams.weightingParameters, ds);
    }
    lookupEvaluator->ExportValues(experimentParams.experimentDirPath);
    assert(lookupEvaluator);
    twisty::PathWeighting::BaseWeightLookupTable &weightingIntegralsRawPointer = (*lookupEvaluator);

    const twisty::PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer
          = (experimentParams.weightingParameters.weightingMethod
                  != twisty::WeightingMethod::RadiativeTransfer)
          ? 1.0
          : twisty::PathWeighting::NormalizerStuff::Norm(
                experimentParams.numSegmentsPerCurve, ds, experimentGeometry);
    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    const double pathNormalizerLog10 = (double)boost::multiprecision::log10(pathNormalizer);

    // Ok, generate the curve.
    const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
    const Farlor::Vector3 point1 = point0 + experimentGeometry.m_startDir * ds;

    const Farlor::Vector3 point5 = experimentGeometry.m_endPos;
    const Farlor::Vector3 point4 = point5 - experimentGeometry.m_endDir * ds;

    // Z axis of new corrdinate frame
    const Farlor::Vector3 zAxis = (point4 - point1).Normalized();
    // Generate orthogonal basis vectors x axis and y axis
    Farlor::Vector3 randomVector = Farlor::Vector3(1.0f, 0.0f, 0.0f);
    if (std::abs(zAxis.Dot(randomVector)) > 0.999f) {
        randomVector = Farlor::Vector3(0.0f, 1.0f, 0.0f);
    }
    const Farlor::Vector3 xAxis = zAxis.Cross(randomVector).Normalized();
    const Farlor::Vector3 yAxis = zAxis.Cross(xAxis).Normalized();

    // Generation of curve stuff
    const double d = (point4 - point1).Magnitude();
    const double d2 = d * d;
    const double leftRadius = ds;
    const double leftRadius2 = leftRadius * leftRadius;
    const double rightRadius = 2.0f * ds;
    const double rightRadius2 = rightRadius * rightRadius;

    double phiExtent = 0.0f;

    if ((leftRadius + rightRadius) < d) {
        std::cout << "No intersection, thus no weight" << std::endl;
        return 1;
    } else if (d + std::min(leftRadius, rightRadius) < std::max(leftRadius, rightRadius)) {
        std::cout << "Left sphere fully in right, thus full range of motion" << std::endl;
        phiExtent = PI;
    } else {
        const double h = 0.5 + (leftRadius2 - rightRadius2) / (2.0 * d2);
        const Farlor::Vector3 centerOfIntersection = point1 + (point4 - point1) * h;
        const double a = std::sqrt(leftRadius2 - (h * h * d2));

        phiExtent = (h * d < 0.0f) ? PI - std::asin(a / leftRadius) : std::asin(a / leftRadius);
    }

    const float uniformPhiSamplingMax = 0.5f - std::cos(phiExtent) * 0.5f;

    //     std::uniform_real_distribution<double> phiDist(0, uniformPhiSamplingMax);
    std::uniform_real_distribution<double> phiDist(0, phiExtent);

    std::uniform_real_distribution<double> thetaDist(0.0f, 2.0f * PI);


    auto FromSpherical = [xAxis, yAxis, zAxis](double phi, double theta) -> Farlor::Vector3 {
        //   return Farlor::Vector3(
        //         std::sin(phi) * std::cos(theta), std::sin(phi) * std::sin(theta), std::cos(phi));
        return xAxis * std::sin(phi) * std::cos(theta) + yAxis * std::sin(phi) * std::sin(theta)
              + zAxis * std::cos(phi);
    };

    std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
    combinedWeightValues.reserve(
          (numExperimentPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

    const int maxThreads = 1;  //omp_get_max_threads();
    std::cout << "Max threads: " << maxThreads << '\n';
    std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
    for (int i = 0; i < maxThreads; i++) {
        twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
    }

    // Per thread min values
    std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
    // Per thread max values
    std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());


    std::mt19937_64 rng(0);
    for (uint64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
        const int threadId = 0;  //omp_get_thread_num();

        //   const double phi = std::acos(1.0 - 2.0 * phiDist(rng));
        const double phi = phiDist(rng);

        const double theta = thetaDist(rng);
        const double theta2 = thetaDist(rng);

        const Farlor::Vector3 point2 = point1 + FromSpherical(phi, theta) * ds;

        const Farlor::Vector3 x_p = (point2 + point4) * 0.5;
        const Farlor::Vector3 lineUnitDir = (point4 - point2).Normalized();

        Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
        if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
            otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
        }

        const Farlor::Vector3 normalToLine = lineUnitDir.Cross(otherCrossVec).Normalized();

        // We should have an even number of segments remaining
        const float hypot = ds;
        const float D_2 = (point4 - point2).Magnitude() * 0.5f;
        assert(D_2 < hypot && "This should never be reached due to earlier check.");

        const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
        Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

        // Now rotate randomly theta amount around the axis.
        {
            const float sinRotAngle = std::sinf(theta2 / 2.0f);
            float quaternionRotation[4] = { std::cosf(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
                lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };

            Farlor::Vector3 shiftedPoint = x_t - point2;
            // Rotate and stuff back in shifted point
            twisty::RotateVectorByQuaternion(quaternionRotation, shiftedPoint.m_data.data());
            // Update the point with the rotated version
            x_t = shiftedPoint + point2;
        }
        const Farlor::Vector3 point3 = x_t;

        std::array<Farlor::Vector3, 6> points = { point0, point1, point2, point3, point4, point5 };
        std::array<Farlor::Vector3, 5> tangents;
        std::array<float, 4> curvatures;

        twisty::PerturbUtils::UpdateTangentsFromPos(
              points.data(), tangents.data(), 5, experimentGeometry);
        twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
              tangents.data(), curvatures.data(), 5, experimentGeometry);

        const double scatteringWeightLog10
              = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                      curvatures.data(), 4, weightingIntegralsRawPointer)
              + pathNormalizerLog10;

        // Update the min and max values
        if (scatteringWeightLog10 < minPathWeightPerThread[threadId]) {
            minPathWeightPerThread[threadId] = scatteringWeightLog10;
        }
        if (scatteringWeightLog10 > maxPathWeightPerThread[threadId]) {
            maxPathWeightPerThread[threadId] = scatteringWeightLog10;
        }

        twisty::CombinedWeightValues_C &activeWeightValue = combinedWeightValuesPerThread[threadId];

        if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
            twisty::CombinedWeightValues_C_AddValue(activeWeightValue, scatteringWeightLog10);
        } else {
            // #pragma omp critical
            {
                combinedWeightValues.push_back(activeWeightValue);
            }
            twisty::CombinedWeightValues_C_Reset(activeWeightValue);
            twisty::CombinedWeightValues_C_AddValue(activeWeightValue, scatteringWeightLog10);
        }
        //     numValidPathsPerThread[threadId]++;
        //     numValidPaths++;
    }

    // For each thread, add the last active weight value in
    for (int i = 0; i < maxThreads; i++) {
        twisty::CombinedWeightValues_C &activeWeightValue = combinedWeightValuesPerThread[i];
        if (activeWeightValue.m_numValues > 0) {
            combinedWeightValues.push_back(activeWeightValue);
        }
    }

    const double overallMinPathWeightLog10
          = *std::min_element(minPathWeightPerThread.begin(), minPathWeightPerThread.end());
    const double overallMaxPathWeightLog10
          = *std::max_element(maxPathWeightPerThread.begin(), maxPathWeightPerThread.end());

    boost::multiprecision::cpp_dec_float_100 pathIntegralResult = 0.0;
    uint64_t numValidPaths = 0;
    for (const auto &combinedWeightValue : combinedWeightValues) {
        pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
        numValidPaths += combinedWeightValue.m_numValues;
    }

    boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;

    if (numValidPaths > 0) {
        finalResult = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
    }

    resultsOFS << "Num valid paths: " << numValidPaths << "/" << numExperimentPaths << std::endl;
    resultsOFS << "Percent valid paths: " << (numValidPaths / (float)numExperimentPaths) * 100.0f
               << "%" << std::endl;

    std::cout << "Num valid paths: " << numValidPaths << "/" << numExperimentPaths << std::endl;
    std::cout << "Percent valid paths: " << (numValidPaths / (float)numExperimentPaths) * 100.0f
              << "%" << std::endl;

    resultsOFS << "Converged final weight: " << finalResult << std::endl;
    std::cout << "Converged final weight: " << finalResult << std::endl;


    resultsOFS << "Min path weight log10: " << overallMinPathWeightLog10 << std::endl;
    resultsOFS << "Max path weight log10: " << overallMaxPathWeightLog10 << std::endl;

    std::cout << "Min path weight log10: " << overallMinPathWeightLog10 << std::endl;
    std::cout << "Max path weight log10: " << overallMaxPathWeightLog10 << std::endl;

    // Big float decompressed versions
    const double overallMinPathWeightLog10Decompressed = std::pow(10.0f, overallMinPathWeightLog10);
    const double overallMaxPathWeightLog10Decompressed = std::pow(10.0f, overallMaxPathWeightLog10);

    resultsOFS << "Min path weight: " << overallMinPathWeightLog10Decompressed << std::endl;
    resultsOFS << "Max path weight: " << overallMaxPathWeightLog10Decompressed << std::endl;

    std::cout << "Min path weight: " << overallMinPathWeightLog10Decompressed << std::endl;
    std::cout << "Max path weight: " << overallMaxPathWeightLog10Decompressed << std::endl;

    std::cout << "Calculating histogram" << std::endl;
    //     // Histogram per thread
    //     const uint64_t numBins = 500;
    //     std::vector<std::vector<uint64_t>> histogramPerThread(maxThreads);
    //     for (int i = 0; i < maxThreads; i++) {
    //         histogramPerThread[i].reserve(numBins);
    //         for (int j = 0; j < numBins; j++) {
    //             histogramPerThread[i][j] = 0;
    //         }
    //     }

    // #pragma omp parallel for num_threads(maxThreads) default(none) shared(histogramPerThread)
    //     for (int phi1Idx = 0; phi1Idx < numPhi1Vals; phi1Idx++) {
    //         const int threadId = omp_get_thread_num();

    //         const float phi1 = phi1Min + phi1Idx * dPhi1;

    //         for (int theta1Idx = 0; theta1Idx < numTheta1Vals; theta1Idx++) {
    //             const float theta1 = theta1Min + theta1Idx * dTheta1;

    //             /*
    //                       x = ρsinφcosθ
    //                       y = ρsinφsinθ
    //                       z = ρcosφ
    //                 */

    //             const float sinPhi1 = std::sin(phi1);
    //             const float cosPhi1 = std::cos(phi1);
    //             const float sinTheta1 = std::sin(theta1);
    //             const float cosTheta1 = std::cos(theta1);

    //             // Calculate the first segment position
    //             const Farlor::Vector3 segment1Dir
    //                   = Farlor::Vector3(sinPhi1 * cosTheta1, sinPhi1 * sinTheta1, cosPhi1);
    //             const Farlor::Vector3 point2 = point1 + segment1Dir * ds;

    //             const float remainingDistance2 = (point4 - point2).SqrMagnitude();

    //             if ((4 * ds * ds) < remainingDistance2) {
    //                 continue;
    //             }

    //             // If not, we keep going through the possible combinations
    //             for (int theta2Idx = 0; theta2Idx < numTheta2Vals; theta2Idx++) {
    //                 const float theta2 = theta2Min + theta2Idx * dTheta2;

    //                 const Farlor::Vector3 x_p = (point2 + point4) * 0.5;
    //                 const Farlor::Vector3 lineUnitDir = (point4 - point2).Normalized();

    //                 Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
    //                 if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
    //                     otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
    //                 }

    //                 const Farlor::Vector3 normalToLine = lineUnitDir.Cross(otherCrossVec).Normalized();

    //                 // We should have an even number of segments remaining
    //                 const float hypot = ds;
    //                 const float D_2 = (point4 - point2).Magnitude() * 0.5f;
    //                 assert(D_2 < hypot && "This should never be reached due to earlier check.");

    //                 const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
    //                 Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

    //                 // Now rotate randomly theta amount around the axis.
    //                 {
    //                     const float sinRotAngle = std::sinf(theta2 / 2.0f);
    //                     float quaternionRotation[4]
    //                           = { std::cosf(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
    //                                 lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


    //                     Farlor::Vector3 shiftedPoint = x_t - point2;
    //                     // Rotate and stuff back in shifted point
    //                     twisty::RotateVectorByQuaternion(
    //                           quaternionRotation, shiftedPoint.m_data.data());
    //                     // Update the point with the rotated version
    //                     x_t = shiftedPoint + point2;
    //                 }
    //                 const Farlor::Vector3 point3 = x_t;

    //                 std::array<Farlor::Vector3, 6> points
    //                       = { point0, point1, point2, point3, point4, point5 };
    //                 std::array<Farlor::Vector3, 5> tangents;
    //                 std::array<float, 4> curvatures;

    //                 twisty::PerturbUtils::UpdateTangentsFromPos(
    //                       points.data(), tangents.data(), 5, experimentGeometry);
    //                 twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
    //                       tangents.data(), curvatures.data(), 5, experimentGeometry);

    //                 const double scatteringWeightLog10
    //                       = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
    //                               curvatures.data(), 4, weightingIntegralsRawPointer)
    //                       + pathNormalizerLog10;
    //                 // Decompressed weight big float
    //                 const double scatteringWeightLog10Decompressed
    //                       = std::pow(10.0, scatteringWeightLog10);


    //                 const uint64_t binIdx = (scatteringWeightLog10 - overallMinPathWeightLog10)
    //                       / (overallMaxPathWeightLog10 - overallMinPathWeightLog10) * numBins;
    //                 histogramPerThread[threadId][binIdx]++;
    //             }
    //         }
    //     }

    //     // Combine bins
    //     std::vector<uint64_t> histogram(numBins);
    //     for (int i = 0; i < maxThreads; i++) {
    //         for (int j = 0; j < numBins; j++) {
    //             histogram[j] += histogramPerThread[i][j];
    //         }
    //     }

    //     // Print histogram to file with bucket range and count
    //     resultsOFS << "Histogram" << std::endl;
    //     for (int i = 0; i < numBins; i++) {
    //         // Big float min
    //         const double binMin = overallMinPathWeightLog10Decompressed
    //               + (overallMaxPathWeightLog10Decompressed - overallMinPathWeightLog10Decompressed) * i
    //                     / numBins;
    //         // Big float max
    //         const double binMax = overallMinPathWeightLog10Decompressed
    //               + (overallMaxPathWeightLog10Decompressed - overallMinPathWeightLog10Decompressed)
    //                     * (i + 1) / numBins;

    //         resultsOFS << binMin << " " << binMax << " " << histogram[i] << std::endl;
    //     }

    std::cout << "Done" << std::endl;
}