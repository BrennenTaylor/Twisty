#include "FullExperimentRunnerOptimalPerturb.h"

#include "CombinedWeightUtils.h"
#include "CurvePerturbUtils.h"
#include "CurveUtils.h"
#include "MathConsts.h"
#include "PathWeighters.h"

#include <cmath>
#include <assert.h>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <limits>

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdlib.h>
#include <memory>

#define EnableHistogramCalc 1

#if EnableHistogramCalc
// Per the path angle integration
static double g_minWeight = 4.69653e-16;
static double g_maxWeight = 1.17517e-13;
static const uint64_t g_numHistogramBuckets = 500;
static std::vector<std::vector<double>> g_perThreadHistograms;
static std::vector<double> g_histogram;
static double g_normalizerLog10 = 0.0f;
static std::ofstream g_histogramFile;
#endif

namespace twisty {
FullExperimentRunnerOptimalPerturb::FullExperimentRunnerOptimalPerturb(
      ExperimentRunner::ExperimentParameters &experimentParams, Bootstrapper &bootstrapper)
    : ExperimentRunner(experimentParams, bootstrapper)
{
}

FullExperimentRunnerOptimalPerturb::~FullExperimentRunnerOptimalPerturb() { }

ExperimentRunner::RunnerSpecificResults
FullExperimentRunnerOptimalPerturb::RunnerSpecificRunExperiment()
{
    auto setupTimeStart = std::chrono::high_resolution_clock::now();

    std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;
    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::SimplifiedModel) {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_ds);
    } else {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_ds);
    }
    assert(lookupEvaluator);
    lookupEvaluator->ExportValues(m_experimentDirPath.string());
    twisty::PathWeighting::BaseWeightLookupTable &weightingIntegralsRawPointer = (*lookupEvaluator);

    const twisty::PerturbUtils::BoundaryConditions boundaryConditions
          = m_upInitialCurve->GetBoundaryConditions();
    assert(boundaryConditions.arclength > 0);

    // Constants
    const int64_t numSystemThreads = std::thread::hardware_concurrency();
    int64_t numPurturbThreads = m_experimentParams.maxPerturbThreads;

    if (numSystemThreads < numPurturbThreads) {
        std::cout
              << "Requested more threads than system has, defaulting to number of system threads"
              << std::endl;
        numPurturbThreads = numSystemThreads;
    }
    if (numPurturbThreads == 0) {
        std::cout << "Requested behavior: Use number of system threads" << std::endl;
        numPurturbThreads = numSystemThreads;
    }
    assert(numPurturbThreads > 0);

    std::cout << "Using " << numPurturbThreads << " threads for purturbation." << std::endl;

    std::filesystem::path convergenceWeightsPath = m_experimentDirPath;
    convergenceWeightsPath /= "ConvergenceWeights/";
    if (!std::filesystem::exists(convergenceWeightsPath)) {
        std::filesystem::create_directories(convergenceWeightsPath);
    }

    std::vector<std::ofstream> perThreadConvergenceFiles(numPurturbThreads);
    std::vector<uint64_t> perThreadNumPathsGenerated(numPurturbThreads);
    std::vector<boost::multiprecision::cpp_dec_float_100> perThreadConvergenceWeight(
          numPurturbThreads);
    for (int threadIdx = 0; threadIdx < numPurturbThreads; threadIdx++) {
        perThreadNumPathsGenerated[threadIdx] = 0;
        perThreadConvergenceWeight[threadIdx] = 0.0;
        const std::string threadFilename
              = "ConvergenceWeights_" + std::to_string(threadIdx) + std::string(".txt");
        perThreadConvergenceFiles[threadIdx].open(convergenceWeightsPath.string() + threadFilename);
        if (!perThreadConvergenceFiles[threadIdx].is_open()) {
            throw std::runtime_error("Failed to open " + threadFilename);
        }
    }

    // Setup rng stuff
    // TODO: Set this up more correctly, based on https://www.pcg-random.org/posts/cpp-seeding-surprises.html
    std::vector<std::mt19937_64> perThreadRngGenerators(numPurturbThreads);
    int64_t seed = m_experimentParams.curvePurturbSeed;
    if (seed == 0) {
        seed = time(0);
    }
    for (int64_t i = 0; i < numPurturbThreads; ++i) {
        perThreadRngGenerators[i] = std::mt19937_64(seed + i);
    }

    // Setup data structures
    std::vector<Farlor::Vector3> initialCurvePositions = m_upInitialCurve->m_positions;
    std::vector<Farlor::Vector3> initialCurveTangents(initialCurvePositions.size() - 1);
    std::vector<float> initialCurveCurvatures(initialCurvePositions.size() - 2);

    // Update tangents
    twisty::PerturbUtils::UpdateTangentsFromPos(initialCurvePositions.data(),
          initialCurveTangents.data(), m_upInitialCurve->m_numSegments, boundaryConditions);

    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::RadiativeTransfer) {
        twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
              initialCurveTangents.data(), initialCurveCurvatures.data(),
              m_upInitialCurve->m_numSegments, boundaryConditions);
    } else {
        twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
              initialCurveTangents.data(), initialCurveCurvatures.data(),
              m_upInitialCurve->m_numSegments, boundaryConditions);
    }

    const int64_t NumPosPerCurve = initialCurvePositions.size();
    const int64_t NumTanPerCurve = initialCurveTangents.size();
    const int64_t NumCurvaturePerCurve = initialCurveCurvatures.size();

    std::vector<Farlor::Vector3> perThreadCurvePositions(NumPosPerCurve * numPurturbThreads);
    std::vector<Farlor::Vector3> perThreadCurveTangents(NumTanPerCurve * numPurturbThreads);
    std::vector<float> perThreadCurveCurvatures(NumCurvaturePerCurve * numPurturbThreads);

    for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx) {
        // Copy Pos
        for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++) {
            perThreadCurvePositions[NumPosPerCurve * threadIdx + posIdx]
                  = initialCurvePositions[posIdx];
        }

        // Copy Tan
        for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++) {
            perThreadCurveTangents[NumTanPerCurve * threadIdx + tanIdx]
                  = initialCurveTangents[tanIdx];
        }

        // Copy Curvatures
        for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++) {
            perThreadCurveCurvatures[NumCurvaturePerCurve * threadIdx + curvatureIdx]
                  = initialCurveCurvatures[curvatureIdx];
        }
    }

    std::vector<Farlor::Vector3> perThreadPositionScratchLeft;
    std::vector<Farlor::Vector3> perThreadTangentScratchLeft;
    std::vector<float> perThreadCurvatureScratchLeft;

    std::vector<Farlor::Vector3> perThreadPositionScratchRight;
    std::vector<Farlor::Vector3> perThreadTangentScratchRight;
    std::vector<float> perThreadCurvatureScratchRight;

    // Only in this case do we allocate room
    if (m_experimentParams.perturbMethod != ExperimentRunner::PerturbMethod::GeometricRandom) {
        perThreadPositionScratchLeft = perThreadCurvePositions;
        perThreadTangentScratchLeft = perThreadCurveTangents;
        perThreadCurvatureScratchLeft = perThreadCurveCurvatures;

        perThreadPositionScratchRight = perThreadCurvePositions;
        perThreadTangentScratchRight = perThreadCurveTangents;
        perThreadCurvatureScratchRight = perThreadCurveCurvatures;
    }

    const PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer
          = (m_experimentParams.weightingParameters.weightingMethod
                  != WeightingMethod::RadiativeTransfer)
          ? 1.0
          : PathWeighting::NormalizerStuff::Norm(
                m_upInitialCurve->m_numSegments, m_upInitialCurve->m_ds, boundaryConditions);
    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    const double pathNormalizerLog10
          = static_cast<double>(boost::multiprecision::log10(pathNormalizer));
    std::cout << "PathNormalizer Log 10: " << pathNormalizerLog10 << std::endl;

    auto setupTimeEnd = std::chrono::high_resolution_clock::now();

    long long setupTimeCount
          = std::chrono::duration_cast<std::chrono::milliseconds>(setupTimeEnd - setupTimeStart)
                  .count();
    ;
    /* --------------------- */

    long long perturbTimeCount = 0;
    long long weightCalcTimeCount = 0;

    const uint64_t MaxNumberOfPathsInDispatch = 20000000000;

    const uint64_t numDispatchesRequired
          = (m_experimentParams.numPathsInExperiment + MaxNumberOfPathsInDispatch - 1)
          / MaxNumberOfPathsInDispatch;
    std::cout << "Number of paths in experiment total: " << m_experimentParams.numPathsInExperiment
              << std::endl;
    std::cout << "Num dispatches required: " << numDispatchesRequired << std::endl;

    uint64_t pathsLeftInExperiment = m_experimentParams.numPathsInExperiment;
    boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;
    uint64_t numPathsGenerated = 0;

#if EnableHistogramCalc
    // For each thread, resize the histrogram
    g_perThreadHistograms.resize(numPurturbThreads);
    for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx) {
        g_perThreadHistograms[threadIdx].resize(g_numHistogramBuckets);
        // Initialize to 0
        for (int64_t bucketIdx = 0; bucketIdx < g_numHistogramBuckets; ++bucketIdx) {
            g_perThreadHistograms[threadIdx][bucketIdx] = 0;
        }
    }
    g_normalizerLog10 = pathNormalizerLog10;
#endif

    for (uint64_t dispatchIdx = 0; dispatchIdx < numDispatchesRequired; dispatchIdx++) {
        /* --------------------- */

        const uint64_t pathsInCurrentDispatch
              = std::min(pathsLeftInExperiment, MaxNumberOfPathsInDispatch);
        pathsLeftInExperiment -= pathsInCurrentDispatch;

        const int64_t numCombinedWeightValues
              = (pathsInCurrentDispatch + MaxNumPathsPerCombinedWeight - 1)
              / MaxNumPathsPerCombinedWeight;

        std::cout << "Num paths in current dispatch: " << pathsInCurrentDispatch << std::endl;
        std::cout << "Num combined weight values needed for current dispatch: "
                  << numCombinedWeightValues << std::endl;

        std::vector<CombinedWeightValues_C> perDispatchCombinedWeightValues(
              numCombinedWeightValues);
        int32_t numCombinedWeightValuesPerThread
              = (numCombinedWeightValues + numPurturbThreads - 1) / numPurturbThreads;
        std::cout << "Num combined weight values per thread: " << numCombinedWeightValuesPerThread
                  << std::endl;

        // Single dispatch, with a number of threads
        {
            auto dispatchTimeStart = std::chrono::high_resolution_clock::now();

            auto perturbTimeStart = std::chrono::high_resolution_clock::now();
            {
                int64_t numPathsPerThread
                      = MaxNumPathsPerCombinedWeight * numCombinedWeightValuesPerThread;
                std::vector<std::thread> threads(numPurturbThreads);
                for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx) {
                    switch (m_experimentParams.perturbMethod) {
                        case ExperimentRunner::PerturbMethod::GeometricRandom: {
                            if (m_experimentParams.outputPathBatches) {
                                std::thread newThread(&FullExperimentRunnerOptimalPerturb::
                                                            GeometryRandom_ExportPaths,
                                      this, threadIdx, pathsInCurrentDispatch, numPathsPerThread,
                                      (dispatchIdx == 0) ? m_experimentParams.numPathsToSkip : 0,
                                      m_experimentParams.numSegmentsPerCurve,
                                      std::ref(perThreadRngGenerators),
                                      std::ref(perThreadCurvePositions),
                                      std::ref(perThreadCurveTangents),
                                      std::ref(perThreadCurveCurvatures),
                                      std::ref(perDispatchCombinedWeightValues),
                                      m_upInitialCurve->m_ds,
                                      lookupEvaluator->AccessLookupTable().data(),
                                      lookupEvaluator->AccessLookupTable().size(),
                                      lookupEvaluator->GetDs(), lookupEvaluator->GetMinCurvature(),
                                      lookupEvaluator->GetMaxCurvature(),
                                      lookupEvaluator->GetCurvatureStepSize(), boundaryConditions,
                                      pathNormalizerLog10);
                                threads[threadIdx] = std::move(newThread);
                            } else {
                                std::thread newThread(
                                      &FullExperimentRunnerOptimalPerturb::GeometryRandom, this,
                                      threadIdx, pathsInCurrentDispatch, numPathsPerThread,
                                      (dispatchIdx == 0) ? m_experimentParams.numPathsToSkip : 0,
                                      m_experimentParams.numSegmentsPerCurve,
                                      std::ref(perThreadRngGenerators),
                                      std::ref(perThreadCurvePositions),
                                      std::ref(perThreadCurveTangents),
                                      std::ref(perThreadCurveCurvatures),
                                      std::ref(perDispatchCombinedWeightValues),
                                      m_upInitialCurve->m_ds,
                                      lookupEvaluator->AccessLookupTable().data(),
                                      lookupEvaluator->AccessLookupTable().size(),
                                      lookupEvaluator->GetDs(), lookupEvaluator->GetMinCurvature(),
                                      lookupEvaluator->GetMaxCurvature(),
                                      lookupEvaluator->GetCurvatureStepSize(), boundaryConditions);
                                threads[threadIdx] = std::move(newThread);
                            }
                        } break;

                        case ExperimentRunner::PerturbMethod::GeometricMinCurvature:
                        case ExperimentRunner::PerturbMethod::GeometricCombined:
                        default: {
                            std::cout << "Error: Invalid Perturb Method" << std::endl;
                            exit(1);
                        } break;
                    };
                }

                for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx) {
                    if (threads[threadIdx].joinable()) {
                        threads[threadIdx].join();
                    }
                }
            }

            auto perturbTimeEnd = std::chrono::high_resolution_clock::now();
            perturbTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(
                  perturbTimeEnd - perturbTimeStart)
                                      .count();

            // -------------------
            auto weightingTimeStart = std::chrono::high_resolution_clock::now();

            int threadIdx = 0;
            uint64_t numCombinedWeightsCurrentThread = 0;

            boost::multiprecision::cpp_dec_float_100 totalDispatchWeight = 0.0;
            for (auto &combinedWeightValue : perDispatchCombinedWeightValues) {
                boost::multiprecision::cpp_dec_float_100 extractedDispatchWeight
                      = ExtractFinalValue(combinedWeightValue);
                totalDispatchWeight += extractedDispatchWeight;

                if (m_experimentParams.outputBigFloatWeights) {
                    UpdateConvergenceWeight(combinedWeightValue.m_numValues,
                          extractedDispatchWeight * pathNormalizer);
                }

                perThreadNumPathsGenerated[threadIdx] += combinedWeightValue.m_numValues;
                perThreadConvergenceWeight[threadIdx] += extractedDispatchWeight * pathNormalizer;

                perThreadConvergenceFiles[threadIdx] << perThreadNumPathsGenerated[threadIdx]
                                                     << ", "
                                                     << perThreadConvergenceWeight[threadIdx]
                            / perThreadNumPathsGenerated[threadIdx]
                                                     << std::endl;
                numCombinedWeightsCurrentThread++;
                if (numCombinedWeightsCurrentThread >= numCombinedWeightValuesPerThread) {
                    numCombinedWeightsCurrentThread = 0;
                    threadIdx++;
                }
            }

            totalDispatchWeight *= pathNormalizer;

            std::cout << "Dispatch Weight: " << totalDispatchWeight << std::endl;
            std::cout << "\tAverage Path Weight in Dispatch: "
                      << totalDispatchWeight / pathsInCurrentDispatch << std::endl;
            bigTotalExperimentWeight += totalDispatchWeight;
            numPathsGenerated += pathsInCurrentDispatch;

            auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
            weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(
                  weightingTimeEnd - weightingTimeStart)
                                         .count();
            /* --------------------- */

            auto dispatchTimeEnd = std::chrono::high_resolution_clock::now();
            auto dispatchRunTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                  dispatchTimeEnd - dispatchTimeStart);
            std::cout << "\tDispatch Time: " << dispatchRunTime.count() << "ms" << std::endl;
        }
    }

#if EnableHistogramCalc
    // Combine histograms
    g_histogram.resize(g_numHistogramBuckets);
      for (int64_t binIdx = 0; binIdx < g_numHistogramBuckets; ++binIdx) {
        g_histogram[binIdx] = 0;
      }

    for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx) {
        for (int64_t binIdx = 0; binIdx < g_numHistogramBuckets; ++binIdx) {
            g_histogram[binIdx] += g_perThreadHistograms[threadIdx][binIdx];
        }
    }

    // Output Histogram
    std::cout << "Writing histogram to: " << m_experimentDirPath.string() + "histogram.txt"
              << std::endl;
    g_histogramFile.open(m_experimentDirPath.string() + "histogram.txt");
    g_histogramFile << "Histogram" << '\n';
    for (int i = 0; i < g_numHistogramBuckets; i++) {
        // Big float min
        const double binMin = g_minWeight + (g_maxWeight - g_minWeight) * i / g_numHistogramBuckets;
        // Big float max
        const double binMax
              = g_minWeight + (g_maxWeight - g_minWeight) * (i + 1) / g_numHistogramBuckets;

        g_histogramFile << binMin << " " << binMax << " " << g_histogram[i] << '\n';
    }
    g_histogramFile.close();
#endif

    ExperimentResults results;
    results.experimentWeight = bigTotalExperimentWeight;
    results.totalPathsGenerated = numPathsGenerated;

    ExperimentRunner::RunnerSpecificResults specificResult;
    specificResult.experimentResults = std::make_optional<ExperimentResults>(results);
    specificResult.setupMs = setupTimeCount;
    specificResult.runExperimentMs = perturbTimeCount;
    specificResult.weightingMs = weightCalcTimeCount;

    return specificResult;
}

void FullExperimentRunnerOptimalPerturb::GeometryRandom(int64_t threadIdx,
      uint64_t numExperimentPaths,
      uint64_t numPathsPerThread,
      uint32_t numPathsToSkipPerThread,
      uint32_t numSegmentsPerCurve,
      std::vector<std::mt19937_64> &rngGenerators,
      std::vector<Farlor::Vector3> &globalPos,
      std::vector<Farlor::Vector3> &globalTans,
      std::vector<float> &globalCurvatures,
      std::vector<CombinedWeightValues_C> &combinedWeightValues,
      float segmentLength,
      const float *pWeightLookupTable,
      const uint32_t weightLookupTableSize,
      const float ds,
      const float minCurvature,
      const float maxCurvature,
      const float curvatureStepSize,
      const twisty::PerturbUtils::BoundaryConditions &boundaryConditions)
{
    const uint32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
    const uint32_t NumTanPerCurve = numSegmentsPerCurve;
    const uint32_t NumCurvaturesPerCurve = (numSegmentsPerCurve - 1);

    const uint32_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
    const uint32_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
    const uint32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

    // Now, we can begin the actual algorithm
    {
        int64_t numCurvesInBatch = 0;
        int64_t outputIdx = 0;

        for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread);
              ++pathCount) {
            // Expect to go negative, thus int
            int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount;

            // We can exit once this point is reached as we have generated all the paths necessary for this thread
            if (currentPathIdx >= (numExperimentPaths + numPathsToSkipPerThread)) {
                printf("Exiting early\n");
                // We dont want to continue if we have already generated the correct number of paths.
                break;
            }

            // Do the perturb now

            std::uniform_int_distribution<int> diffDist(
                  2, std::min((int)(numSegmentsPerCurve - 2), 25));  // uniform, unbiased
            uint32_t diff = diffDist(rngGenerators[threadIdx]);

            std::uniform_int_distribution<int> leftPointIndexUniformDist(
                  1, numSegmentsPerCurve - diff - 1);  // uniform, unbiased
            uint32_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);
            uint32_t rightPointIndex = leftPointIndex + diff;

            // We need two frames for each segment to get the new curvature and torsion.
            // we need the frame left of the segment, as well as the frame right of the segment.
            // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
            const Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftPointIndex];
            const Farlor::Vector3 rightPoint
                  = globalPos[CurrentThreadPosStartIdx + rightPointIndex];

            const Farlor::Vector3 N = (rightPoint - leftPoint).Normalized();

            std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi, TwistyPi);
            float randRotationAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);


            // Quaternion Rotation
            {
                const float sinRotAngle = std::sin(randRotationAngle / 2.0f);
                float quaternionRotation[4] = { std::cos(randRotationAngle / 2.0f),
                    N.x * sinRotAngle, N.y * sinRotAngle, N.z * sinRotAngle };

                for (uint32_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
                      ++pointIdx) {
                    Farlor::Vector3 shiftedPoint
                          = (globalPos[CurrentThreadPosStartIdx + pointIdx]) - leftPoint;
                    // Rotate and stuff back in shifted point
                    RotateVectorByQuaternion(quaternionRotation, shiftedPoint.m_data.data());
                    // Update the point with the rotated version
                    globalPos[CurrentThreadPosStartIdx + pointIdx] = (shiftedPoint + leftPoint);
                }

                //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                //We can do a different approach later.
                // Here, we want to do a perturb update call
                twisty::PerturbUtils::UpdateTangentsFromPos(&globalPos[CurrentThreadPosStartIdx],
                      &globalTans[CurrentThreadTanStartIdx], numSegmentsPerCurve,
                      boundaryConditions);


                if (m_experimentParams.weightingParameters.weightingMethod
                      == WeightingMethod::RadiativeTransfer) {
                    twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                          &globalTans[CurrentThreadTanStartIdx],
                          &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
                          boundaryConditions);
                } else {
                    twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
                          &globalTans[CurrentThreadTanStartIdx],
                          &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
                          boundaryConditions);
                }
            }

            // Normalized version
            double scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CudaSafe(
                        &(globalCurvatures[CurrentThreadCurvatureStartIdx]),
                        NumCurvaturesPerCurve,
                        pWeightLookupTable,
                        weightLookupTableSize,
                        ds,
                        minCurvature,
                        maxCurvature,
                        curvatureStepSize,
                        m_experimentParams.weightingParameters.absorption);

            if (pathCount < numPathsToSkipPerThread) {
                // Skip
            } else {
#if EnableHistogramCalc
                const double scatteringWeightLog10Decompressed
                      = std::pow(10.0, (scatteringWeightLog10 + g_normalizerLog10));

                const uint64_t binIdx =
                      [scatteringWeightLog10Decompressed]()->uint64_t {
                    if (scatteringWeightLog10Decompressed < g_minWeight)
                        return 0;
                    if (scatteringWeightLog10Decompressed > g_maxWeight)
                        return (g_numHistogramBuckets - 1);
                    return (scatteringWeightLog10Decompressed - g_minWeight) / (g_maxWeight - g_minWeight)
                          * g_numHistogramBuckets;
                }();
                g_perThreadHistograms[threadIdx][binIdx]++;
#endif


                // Else, contribute to the paths
                int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount;
                assert((currentPathIdx - numPathsToSkipPerThread) >= numPathsPerThread * threadIdx);

                int combinedWeightValueIdx
                      = ((currentPathIdx - numPathsToSkipPerThread) / MaxNumPathsPerCombinedWeight);

                if (((currentPathIdx - numPathsToSkipPerThread) % MaxNumPathsPerCombinedWeight)
                      == 0) {
                    // printf("****************Calling reset: %d\n", combinedWeightValueIdx);
                    CombinedWeightValues_C_Reset(combinedWeightValues[combinedWeightValueIdx]);
                }

                //printf("****************Adding Value: %d, %lf\n", combinedWeightValueIdx, scatteringWeightLog10);
                CombinedWeightValues_C_AddValue(
                      combinedWeightValues[combinedWeightValueIdx], scatteringWeightLog10);
            }
        }
    }
}

void FullExperimentRunnerOptimalPerturb::GeometryRandom_ExportPaths(int64_t threadIdx,
      uint64_t numExperimentPaths,
      uint64_t numPathsPerThread,
      uint32_t numPathsToSkipPerThread,
      uint32_t numSegmentsPerCurve,
      std::vector<std::mt19937_64> &rngGenerators,
      std::vector<Farlor::Vector3> &globalPos,
      std::vector<Farlor::Vector3> &globalTans,
      std::vector<float> &globalCurvatures,
      std::vector<CombinedWeightValues_C> &combinedWeightValues,
      float segmentLength,
      const float *pWeightLookupTable,
      const uint32_t weightLookupTableSize,
      const float ds,
      const float minCurvature,
      const float maxCurvature,
      const float curvatureStepSize,
      const twisty::PerturbUtils::BoundaryConditions &boundaryConditions,
      const double pathNormalizerLog10)
{
    const uint32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
    const uint32_t NumTanPerCurve = numSegmentsPerCurve;
    const uint32_t NumCurvaturesPerCurve = (numSegmentsPerCurve - 1);

    const uint32_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
    const uint32_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
    const uint32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

    const uint32_t ExportPathBatchCacheSize = 1000000;
    std::vector<Farlor::Vector3> pathBatchCache(ExportPathBatchCacheSize * NumPosPerCurve);
    std::vector<double> log10PathWeightCache(ExportPathBatchCacheSize);
    std::vector<Farlor::Vector3> fiveSegmentPathCache(ExportPathBatchCacheSize);

    // Now, we can begin the actual algorithm
    {
        int64_t numCurvesInBatch = 0;
        int64_t outputIdx = 0;

        for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread);
              ++pathCount) {
            // Expect to go negative, thus int
            int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount;

            // We can exit once this point is reached as we have generated all the paths necessary for this thread
            if (currentPathIdx >= (numExperimentPaths + numPathsToSkipPerThread)) {
                printf("Exiting early\n");
                // We dont want to continue if we have already generated the correct number of paths.
                break;
            }

            // Do the perturb now

            std::uniform_int_distribution<int> diffDist(
                  2, std::min((int)(numSegmentsPerCurve - 2), 25));  // uniform, unbiased
            uint32_t diff = diffDist(rngGenerators[threadIdx]);

            std::uniform_int_distribution<int> leftPointIndexUniformDist(
                  1, numSegmentsPerCurve - diff - 1);  // uniform, unbiased
            uint32_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);
            uint32_t rightPointIndex = leftPointIndex + diff;

            // We need two frames for each segment to get the new curvature and torsion.
            // we need the frame left of the segment, as well as the frame right of the segment.
            // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
            const Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftPointIndex];
            const Farlor::Vector3 rightPoint
                  = globalPos[CurrentThreadPosStartIdx + rightPointIndex];

            const Farlor::Vector3 N = (rightPoint - leftPoint).Normalized();

            std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi, TwistyPi);
            float randRotationAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);


            // Quaternion Rotation
            {
                const float sinRotAngle = std::sin(randRotationAngle / 2.0f);
                float quaternionRotation[4] = { std::cos(randRotationAngle / 2.0f),
                    N.x * sinRotAngle, N.y * sinRotAngle, N.z * sinRotAngle };

                for (uint32_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
                      ++pointIdx) {
                    Farlor::Vector3 shiftedPoint
                          = (globalPos[CurrentThreadPosStartIdx + pointIdx]) - leftPoint;
                    // Rotate and stuff back in shifted point
                    RotateVectorByQuaternion(quaternionRotation, shiftedPoint.m_data.data());
                    // Update the point with the rotated version
                    globalPos[CurrentThreadPosStartIdx + pointIdx] = (shiftedPoint + leftPoint);
                }

                //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                //We can do a different approach later.
                // Here, we want to do a perturb update call
                twisty::PerturbUtils::UpdateTangentsFromPos(&globalPos[CurrentThreadPosStartIdx],
                      &globalTans[CurrentThreadTanStartIdx], numSegmentsPerCurve,
                      boundaryConditions);


                if (m_experimentParams.weightingParameters.weightingMethod
                      == WeightingMethod::RadiativeTransfer) {
                    twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                          &globalTans[CurrentThreadTanStartIdx],
                          &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
                          boundaryConditions);
                } else {
                    twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
                          &globalTans[CurrentThreadTanStartIdx],
                          &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
                          boundaryConditions);
                }
            }

            // Normalized version
            double scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CudaSafe(
                        &(globalCurvatures[CurrentThreadCurvatureStartIdx]),
                        NumCurvaturesPerCurve,
                        pWeightLookupTable,
                        weightLookupTableSize,
                        ds,
                        minCurvature,
                        maxCurvature,
                        curvatureStepSize,
                        m_experimentParams.weightingParameters.absorption);

            if (pathCount < numPathsToSkipPerThread) {
                // Skip
            } else {
                // Else, contribute to the paths
                int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount;
                assert((currentPathIdx - numPathsToSkipPerThread) >= numPathsPerThread * threadIdx);

                int combinedWeightValueIdx
                      = ((currentPathIdx - numPathsToSkipPerThread) / MaxNumPathsPerCombinedWeight);

                if (((currentPathIdx - numPathsToSkipPerThread) % MaxNumPathsPerCombinedWeight)
                      == 0) {
                    // printf("****************Calling reset: %d\n", combinedWeightValueIdx);
                    CombinedWeightValues_C_Reset(combinedWeightValues[combinedWeightValueIdx]);
                }

                CombinedWeightValues_C_AddValue(
                      combinedWeightValues[combinedWeightValueIdx], scatteringWeightLog10);

                // Add the path to the path batch
                for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx) {
                    Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
                    pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
                }
                log10PathWeightCache[numCurvesInBatch]
                      = std::pow(10.0, (scatteringWeightLog10 + pathNormalizerLog10));

                if (m_experimentParams.numSegmentsPerCurve == 5) {
                    // TODO: Extract out the three angles for 5 segment paths here
                    // Get the five points
                    const Farlor::Vector3 point0 = globalPos[CurrentThreadPosStartIdx + 0];
                    const Farlor::Vector3 point1 = globalPos[CurrentThreadPosStartIdx + 1];
                    const Farlor::Vector3 point2 = globalPos[CurrentThreadPosStartIdx + 2];
                    const Farlor::Vector3 point3 = globalPos[CurrentThreadPosStartIdx + 3];
                    const Farlor::Vector3 point4 = globalPos[CurrentThreadPosStartIdx + 4];
                    const Farlor::Vector3 point5 = globalPos[CurrentThreadPosStartIdx + 5];

                    const Farlor::Vector3 segment1Dir = (point2 - point1).Normalized();
                    // Extract angles
                    const float phi1 = std::acos(segment1Dir.z);
                    const float x = segment1Dir.Dot(Farlor::Vector3(1.0f, 0.0f, 0.0f));
                    const float y = segment1Dir.Dot(Farlor::Vector3(0.0f, 1.0f, 0.0f));
                    const float theta1 = std::atan2(y, x);


                    // Extract the theta2 value
                    const Farlor::Vector3 theta2_Z = (point4 - point2).Normalized();
                    Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
                    if (abs(theta2_Z.Dot(otherCrossVec)) >= 0.99) {
                        otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
                    }

                    const Farlor::Vector3 theta2_X = theta2_Z.Cross(otherCrossVec).Normalized();
                    const Farlor::Vector3 theta2_Y = theta2_Z.Cross(theta2_X);

                    const Farlor::Vector3 vectorInQuestion = (point3 - point2).Normalized();

                    const float theta2_x = vectorInQuestion.Dot(theta2_X);
                    const float theta2_y = vectorInQuestion.Dot(theta2_Y);

                    const float theta2 = std::atan2(theta2_y, theta2_x);


                    // Save off the values
                    fiveSegmentPathCache[numCurvesInBatch].x = phi1;
                    fiveSegmentPathCache[numCurvesInBatch].y = theta1;
                    fiveSegmentPathCache[numCurvesInBatch].z = theta2;
                }

                numCurvesInBatch++;

                if (numCurvesInBatch == ExportPathBatchCacheSize) {
                    m_exportPathBatchesMutex.lock();

                    m_curvesMetadataFile << threadIdx << " ";
                    m_curvesMetadataFile << outputIdx << " ";
                    m_curvesMetadataFile << numCurvesInBatch << std::endl;

                    m_curvesBinaryFile.write((char *)pathBatchCache.data(),
                          sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);

                    m_log10PathWeightsBinaryFile.write(
                          (char *)log10PathWeightCache.data(), sizeof(double) * numCurvesInBatch);

                    for (int i = 0; i < numCurvesInBatch; i++) {
                        if ((static_cast<uint32_t>(i + threadIdx * (10000.0 / 12.0)) % 10000)
                              == 0) {
                            m_log10PathWeightsTextFile << log10PathWeightCache[i] << std::endl;
                        }
                    }

                    m_fiveSegmentBinaryFile.write((char *)fiveSegmentPathCache.data(),
                          sizeof(Farlor::Vector3) * numCurvesInBatch);


                    numCurvesInBatch = 0;
                    outputIdx++;

                    m_exportPathBatchesMutex.unlock();
                }
            }
        }

        if (numCurvesInBatch > 0) {
            m_exportPathBatchesMutex.lock();

            m_curvesMetadataFile << threadIdx << " ";
            m_curvesMetadataFile << outputIdx << " ";
            m_curvesMetadataFile << numCurvesInBatch << std::endl;

            m_curvesBinaryFile.write((char *)pathBatchCache.data(),
                  sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);

            m_log10PathWeightsBinaryFile.write(
                  (char *)log10PathWeightCache.data(), sizeof(double) * numCurvesInBatch);

            for (int i = 0; i < numCurvesInBatch; i++) {
                if ((static_cast<uint32_t>(i + threadIdx * (10000.0 / 12.0)) % 10000) == 0) {
                    m_log10PathWeightsTextFile << log10PathWeightCache[i] << std::endl;
                }
            }

            m_fiveSegmentBinaryFile.write(
                  (char *)fiveSegmentPathCache.data(), sizeof(Farlor::Vector3) * numCurvesInBatch);

            numCurvesInBatch = 0;
            outputIdx++;

            m_exportPathBatchesMutex.unlock();
        }
    }
}
}