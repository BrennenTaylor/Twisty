#include "FullExperimentRunnerOptimalPerturb.h"

#include "CombinedWeightUtils.h"
#include "CurvePerturbUtils.h"
#include "CurveUtils.h"
#include "MathConsts.h"
#include "PathWeighters.h"

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

    assert(m_experimentParams.weightingParameters.scatterValues.size() > 0);

    // TODO: For now, we simply will support one scattering value
    if (m_experimentParams.weightingParameters.scatterValues.size() > 1) {
        std::cout << "WARNING: Only one scatter value supported, defaulting to first specified "
                     "scatter parameter"
                  << std::endl;
    }
    m_experimentParams.weightingParameters.scatter
          = m_experimentParams.weightingParameters.scatterValues[0];
    std::unique_ptr<twisty::PathWeighting::BaseWeightLookupTable> lookupEvaluator = nullptr;
    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::SimplifiedModel) {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::SimpleWeightLookupTable>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_ds);
    } else {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_ds);
    }
    lookupEvaluator->ExportValues(m_experimentDirPath.string());
    assert(lookupEvaluator);
    twisty::PathWeighting::BaseWeightLookupTable &weightingIntegralsRawPointer = (*lookupEvaluator);


    const twisty::PerturbUtils::BoundaryConditions boundaryConditions
          = m_upInitialCurve->GetBoundaryConditions();
    assert(boundaryConditions.arclength > 0);

    // Constants
    int64_t numSystemThreads = std::thread::hardware_concurrency();
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
                                      lookupEvaluator->GetCurvatureStepSize(), boundaryConditions);
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

    ExperimentResults results;
    results.experimentWeights.push_back(bigTotalExperimentWeight);
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
                const float sinRotAngle = std::sinf(randRotationAngle / 2.0f);
                float quaternionRotation[4] = { std::cosf(randRotationAngle / 2.0f),
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


            // // Matrix Rotation
            // {
            //     float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            //     RotationMatrixAroundAxis_AngleAsIs_CudaSafe(
            //           randRotationAngle, N.m_data.data(), rotationMatrix);

            //     for (uint32_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
            //           ++pointIdx) {
            //         Farlor::Vector3 shiftedPoint
            //               = (globalPos[CurrentThreadPosStartIdx + pointIdx]) - leftPoint;
            //         // Rotate and stuff back in shifted point
            //         RotateVectorByMatrix(rotationMatrix, shiftedPoint.m_data.data());
            //         // Update the point with the rotated version
            //         globalPos[CurrentThreadPosStartIdx + pointIdx] = (shiftedPoint + leftPoint);
            //     }

            //     //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
            //     //We can do a different approach later.
            //     // Here, we want to do a perturb update call
            //     twisty::PerturbUtils::UpdateTangentsFromPos(&globalPos[CurrentThreadPosStartIdx],
            //           &globalTans[CurrentThreadTanStartIdx], numSegmentsPerCurve,
            //           boundaryConditions);


            //     if (m_experimentParams.weightingParameters.weightingMethod
            //           == WeightingMethod::RadiativeTransfer) {
            //         twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
            //               &globalTans[CurrentThreadTanStartIdx],
            //               &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
            //               boundaryConditions);
            //     } else {
            //         twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
            //               &globalTans[CurrentThreadTanStartIdx],
            //               &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
            //               boundaryConditions);
            //     }
            // }


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
                        curvatureStepSize);

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
      const twisty::PerturbUtils::BoundaryConditions &boundaryConditions)
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
                const float sinRotAngle = std::sinf(randRotationAngle / 2.0f);
                float quaternionRotation[4] = { std::cosf(randRotationAngle / 2.0f),
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
                        curvatureStepSize);

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

                //printf("****************Adding Value: %d, %lf\n", combinedWeightValueIdx, scatteringWeightLog10);
                CombinedWeightValues_C_AddValue(
                      combinedWeightValues[combinedWeightValueIdx], scatteringWeightLog10);

                // Add the path to the path batch
                for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx) {
                    Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
                    pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
                }
                log10PathWeightCache[numCurvesInBatch] = scatteringWeightLog10;

                // TODO: Extract out the three angles for 5 segment paths here
                // Get the five points
                const Farlor::Vector3 point0 = globalPos[CurrentThreadPosStartIdx + 0];
                const Farlor::Vector3 point1 = globalPos[CurrentThreadPosStartIdx + 1];
                const Farlor::Vector3 point2 = globalPos[CurrentThreadPosStartIdx + 2];
                const Farlor::Vector3 point3 = globalPos[CurrentThreadPosStartIdx + 3];
                const Farlor::Vector3 point4 = globalPos[CurrentThreadPosStartIdx + 4];
                const Farlor::Vector3 point5 = globalPos[CurrentThreadPosStartIdx + 5];

                const Farlor::Vector3 xDir = (point4 - point1).Normalized();
                const Farlor::Vector3 seg1Dir = (point2 - point1).Normalized();
                const float theta1 = xDir.Dot(seg1Dir);
                const Farlor::Vector3 phi1ReferenceDir
                      = ((point1 - point0).Normalized()).Cross(xDir).Normalized();
                const float phi1 = phi1ReferenceDir.Dot(seg1Dir);

                const Farlor::Vector3 phi2ReferenceDir
                      = ((point5 - point4).Normalized()).Cross(xDir).Normalized();
                const Farlor::Vector3 seg3BackDir = (point3 - point4).Normalized();
                const float phi2 = phi2ReferenceDir.Dot(seg3BackDir);

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

            m_fiveSegmentBinaryFile.write(
                  (char *)fiveSegmentPathCache.data(), sizeof(Farlor::Vector3) * numCurvesInBatch);

            numCurvesInBatch = 0;
            outputIdx++;

            m_exportPathBatchesMutex.unlock();
        }
    }
}


// void FullExperimentRunnerOptimalPerturb::GeometryCombined(int64_t threadIdx,
//       int64_t numExperimentPaths,
//       int64_t numPathsPerThread,
//       int64_t numPathsToSkipPerThread,
//       int64_t numSegmentsPerCurve,
//       std::vector<std::mt19937_64> &rngGenerators,
//       std::vector<Farlor::Vector3> &globalPos,
//       std::vector<Farlor::Vector3> &globalTans,
//       std::vector<float> &globalCurvatures,
//       std::vector<Farlor::Vector3> &scratchPositionSpaceLeft,
//       std::vector<Farlor::Vector3> &scratchTangentSpaceLeft,
//       std::vector<float> &scratchCurvatureSpaceLeft,
//       std::vector<Farlor::Vector3> &scratchPositionSpaceRight,
//       std::vector<Farlor::Vector3> &scratchTangentSpaceRight,
//       std::vector<float> &scratchCurvatureSpaceRight,
//       std::vector<double> &globalPathWeights,
//       // std::vector<double> &cachedSegmentWeights,
//       float segmentLength,
//       twisty::PathWeighting::BaseWeightLookupTable *weightingIntegralPtr,
//       const twisty::PerturbUtils::BoundaryConditions &boundaryConditions)
// {
//     // if (m_experimentParams.exportGeneratedCurves)
//     // {
//     //     // This should be per thread
//     //     int64_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
//     //     std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
//     //     {
//     //         for (int64_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
//     //         {
//     //             for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//     //             {
//     //                 int64_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
//     //                 pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//     //             }
//     //         }
//     //     }
//     // }

//     const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
//     const int64_t NumTanPerCurve = numSegmentsPerCurve;
//     const int64_t NumCurvaturesPerCurve = (numSegmentsPerCurve - 1);

//     const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
//     const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
//     const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

//     // Now, we can begin the actual algorithm
//     {
//         // This is the perturbation piece.
//         // Can we do this in place, most likely
//         // This will modify pCurrentThreadCurve
//         // Remember, the structure of this is:
//         // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M

//         // Start at the thread's first path idx

//         int64_t numCurvesInBatch = 0;
//         int64_t outputIdx = 0;


//         bool useOptimal = false;
//         const uint32_t numRandom = 1000;
//         const uint32_t numOptimal = 5000;
//         uint32_t countCurrentMethod = 0;

//         for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread);
//               ++pathCount) {
//             // Expect to go negative, thus int
//             int64_t currentPathIdx
//                   = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;

//             // We can exit once this point is reached as we have generated all the paths necessary for this thread
//             if (currentPathIdx >= numExperimentPaths) {
//                 // We dont want to continue if we have already generated the correct number of paths.
//                 break;
//             }

//             // Do the perturb now

//             // Each time, we first copy the "old path" to the "scratch space"
//             for (uint32_t segIdx = 0; segIdx <= numSegmentsPerCurve; ++segIdx) {
//                 scratchPositionSpaceLeft[CurrentThreadPosStartIdx + segIdx]
//                       = globalPos[CurrentThreadPosStartIdx + segIdx];
//                 scratchPositionSpaceRight[CurrentThreadPosStartIdx + segIdx]
//                       = globalPos[CurrentThreadPosStartIdx + segIdx];
//             }

//             // Update the tangents and curvatures
//             {
//                 twisty::PerturbUtils::UpdateTangentsFromPos(
//                       &scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
//                       &scratchTangentSpaceLeft[CurrentThreadTanStartIdx], numSegmentsPerCurve,
//                       boundaryConditions);

//                 twisty::PerturbUtils::UpdateCurvaturesFromTangents(
//                       &scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
//                       &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
//                       numSegmentsPerCurve, boundaryConditions,
//                       (int32_t)m_experimentParams.weightingParameters.weightingMethod);
//             }

//             // Update the right tangents and curvatures
//             {
//                 twisty::PerturbUtils::UpdateTangentsFromPos(
//                       &scratchPositionSpaceRight[CurrentThreadPosStartIdx],
//                       &scratchTangentSpaceRight[CurrentThreadTanStartIdx], numSegmentsPerCurve,
//                       boundaryConditions);

//                 twisty::PerturbUtils::UpdateCurvaturesFromTangents(
//                       &scratchTangentSpaceRight[CurrentThreadTanStartIdx],
//                       &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
//                       numSegmentsPerCurve, boundaryConditions,
//                       (int32_t)m_experimentParams.weightingParameters.weightingMethod);
//             }

//             std::uniform_int_distribution<int> diffDist(
//                   2, std::min((int)(numSegmentsPerCurve - 2), 25));  // uniform, unbiased
//             int64_t diff = diffDist(rngGenerators[threadIdx]);

//             std::uniform_int_distribution<int> leftPointIndexUniformDist(
//                   1, numSegmentsPerCurve - diff - 1);  // uniform, unbiased
//             int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);

//             int64_t rightPointIndex = leftPointIndex + diff;

//             assert((rightPointIndex - leftPointIndex) >= diff);
//             assert(leftPointIndex < rightPointIndex);

//             // We need two frames for each segment to get the new curvature and torsion.
//             // we need the frame left of the segment, as well as the frame right of the segment.
//             // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
//             const Farlor::Vector3 leftPoint
//                   = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
//             const Farlor::Vector3 rightPoint
//                   = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];

//             const Farlor::Vector3 N = (rightPoint - leftPoint).Normalized();

//             double leftRotationAngle = 0.0;
//             {
//                 const Farlor::Vector3 Xss1
//                       = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex - 1];
//                 const Farlor::Vector3 Xs
//                       = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
//                 const Farlor::Vector3 Xsp1
//                       = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex + 1];
//                 const Farlor::Vector3 PL = Xs + ((Xsp1 - Xs).Dot(N)) * N;
//                 const Farlor::Vector3 ZL = Xss1 - ((Xss1 - PL).Dot(N)) * N;

//                 // Get side of plane Z is on
//                 const Farlor::Vector3 NL = N.Cross((Xsp1 - PL).Normalized()).Normalized();
//                 const double sideDistL = (ZL - PL).Dot(NL);

//                 const Farlor::Vector3 ZPnorm = (ZL - PL).Normalized();
//                 const Farlor::Vector3 Xsp1PLnorm = (Xsp1 - PL).Normalized();
//                 double cosAngle = ZPnorm.Dot(Xsp1PLnorm);
//                 cosAngle = std::max(-1.0, cosAngle);
//                 cosAngle = std::min(1.0, cosAngle);
//                 const double angle = acos(cosAngle);

//                 const double threshold = 10e-10;

//                 // In the case, we are aligned, we want a Pi rotation
//                 if (abs(angle) <= threshold) {
//                     leftRotationAngle = TwistyPi;
//                 }
//                 // If we are Pi away, we want no rotation
//                 else if (abs(abs(angle) - TwistyPi) < threshold) {
//                     leftRotationAngle = 0.0;
//                 }
//                 // On back side
//                 else if (sideDistL < 0.0) {
//                     leftRotationAngle = TwistyPi - angle;
//                 }
//                 // On front side
//                 else {
//                     leftRotationAngle = -1.0 * (TwistyPi - angle);
//                 }
//             }

//             double rightRotationAngle = 0.0;
//             {
//                 const Farlor::Vector3 Xes1
//                       = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex - 1];
//                 const Farlor::Vector3 Xe
//                       = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];

//                 // Handle case where we are rotation with end point as pivot point
//                 Farlor::Vector3 Xep1(0.0f, 0.0f, 0.0f);
//                 if (rightPointIndex == numSegmentsPerCurve) {
//                     Xep1 = Xe + boundaryConditions.m_endPos
//                           + segmentLength * boundaryConditions.m_endDir.Normalized();
//                 } else {
//                     Xep1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex + 1];
//                 }


//                 //const Farlor::Vector3 Xep1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex + 1];
//                 const Farlor::Vector3 PR = Xe + ((Xes1 - Xe).Dot(N)) * N;
//                 const Farlor::Vector3 ZR = Xep1 - ((Xep1 - PR).Dot(N)) * N;

//                 // Get side of plane Z is on
//                 const Farlor::Vector3 NR = N.Cross((Xes1 - PR).Normalized()).Normalized();
//                 const double sideDistR = (ZR - PR).Dot(NR);

//                 const Farlor::Vector3 ZPnorm = (ZR - PR).Normalized();
//                 const Farlor::Vector3 Xes1PLnorm = (Xes1 - PR).Normalized();
//                 double cosAngle = ZPnorm.Dot(Xes1PLnorm);
//                 cosAngle = std::max(-1.0, cosAngle);
//                 cosAngle = std::min(1.0, cosAngle);
//                 const double angle = acos(cosAngle);

//                 const double threshold = 10e-10;

//                 // In the case, we are aligned, we want a Pi rotation
//                 if (abs(angle) <= threshold) {
//                     rightRotationAngle = TwistyPi;
//                 }
//                 // If we are Pi away, we want no rotation
//                 else if (abs(abs(angle) - TwistyPi) < threshold) {
//                     rightRotationAngle = 0.0;
//                 }
//                 // On back side
//                 else if (sideDistR < 0.0) {
//                     rightRotationAngle = TwistyPi - angle;
//                 }
//                 // On front side
//                 else {
//                     rightRotationAngle = -1.0 * (TwistyPi - angle);
//                 }
//             }

//             // Overwrite angle
//             if (!useOptimal) {
//                 countCurrentMethod++;
//                 if (countCurrentMethod >= numRandom) {
//                     useOptimal = !useOptimal;
//                 }
//                 //double mult = (((pathCount / 2) % 2) == 0) ? 1 : -1;

//                 std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi, TwistyPi);
//                 //rotationAngle = TwistyPi * 0.75 * mult;
//                 double randRotationAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
//                 leftRotationAngle = randRotationAngle;
//                 rightRotationAngle = randRotationAngle;
//                 //std::cout << "Overwritten rotation angle: " << randRotationAngle << std::endl;
//             } else {
//                 countCurrentMethod++;
//                 if (countCurrentMethod >= numOptimal) {
//                     useOptimal = !useOptimal;
//                 }
//             }

//             // Left Rotation
//             {
//                 float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
//                 RotationMatrixAroundAxis(leftRotationAngle, (float *)(&N), rotationMatrix);

//                 for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
//                       ++pointIdx) {
//                     Farlor::Vector3 shiftedPoint
//                           = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx]
//                           - leftPoint;
//                     // Rotate and stuff back in shifted point
//                     RotateVectorByMatrix(rotationMatrix, (float *)(&shiftedPoint));
//                     // Update the point with the rotated version
//                     scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx]
//                           = shiftedPoint + leftPoint;
//                 }

//                 //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
//                 //We can do a different approach later.
//                 // Here, we want to do a perturb update call
//                 twisty::PerturbUtils::UpdateTangentsFromPos(
//                       &scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
//                       &scratchTangentSpaceLeft[CurrentThreadTanStartIdx], numSegmentsPerCurve,
//                       boundaryConditions);

//                 twisty::PerturbUtils::UpdateCurvaturesFromTangents(
//                       &scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
//                       &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
//                       numSegmentsPerCurve, boundaryConditions,
//                       (int32_t)m_experimentParams.weightingParameters.weightingMethod);
//             }

//             // Right Rotation
//             {
//                 float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
//                 RotationMatrixAroundAxis(rightRotationAngle, (float *)(&N), rotationMatrix);

//                 for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
//                       ++pointIdx) {
//                     Farlor::Vector3 shiftedPoint
//                           = scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx]
//                           - leftPoint;
//                     // Rotate and stuff back in shifted point
//                     RotateVectorByMatrix(rotationMatrix, (float *)(&shiftedPoint));
//                     // Update the point with the rotated version
//                     scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx]
//                           = shiftedPoint + leftPoint;
//                 }

//                 //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
//                 //We can do a different approach later.
//                 // Here, we want to do a perturb update call
//                 twisty::PerturbUtils::UpdateTangentsFromPos(
//                       &scratchPositionSpaceRight[CurrentThreadPosStartIdx],
//                       &scratchTangentSpaceRight[CurrentThreadTanStartIdx], numSegmentsPerCurve,
//                       boundaryConditions);

//                 twisty::PerturbUtils::UpdateCurvaturesFromTangents(
//                       &scratchTangentSpaceRight[CurrentThreadTanStartIdx],
//                       &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
//                       numSegmentsPerCurve, boundaryConditions,
//                       (int32_t)m_experimentParams.weightingParameters.weightingMethod);
//             }

//             double leftPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
//                   &(scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx]), numSegmentsPerCurve,
//                   *weightingIntegralPtr);

//             double rightPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
//                   &(scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx]),
//                   numSegmentsPerCurve, *weightingIntegralPtr);

//             bool useLeftRotation = (leftPathWeightLog10 >= rightPathWeightLog10);

//             uint32_t numBetas = (rightPointIndex - 1) - leftPointIndex;
//             std::vector<Farlor::Vector3> oldBetas(numBetas);
//             std::vector<Farlor::Vector3> newBetas(numBetas);
//             for (int64_t tanIdx = 0; tanIdx < numBetas; ++tanIdx) {
//                 oldBetas[tanIdx] = globalTans[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];

//                 if (useLeftRotation) {
//                     newBetas[tanIdx] = scratchTangentSpaceLeft[CurrentThreadTanStartIdx
//                           + leftPointIndex + tanIdx];
//                 } else {
//                     newBetas[tanIdx] = scratchTangentSpaceRight[CurrentThreadTanStartIdx
//                           + leftPointIndex + tanIdx];
//                 }
//             }

//             // Now we have a candidate path
//             // We perform metropolis and see if we want to accept the path, i.e. copy the scratch space values to the actual curve values, or reroll a new curve
//             std::uniform_real_distribution<double> uniformRandomZeroOne(0.0, 1.0);

//             double acceptanceProb = uniformRandomZeroOne(rngGenerators[threadIdx]);

//             // For now, we always accept a path
//             // This is not using the metropolis sampling at all
//             double oldPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
//                   &(globalCurvatures[CurrentThreadCurvatureStartIdx]), numSegmentsPerCurve,
//                   *weightingIntegralPtr);

//             double newPathWeightLog10 = 0.0;

//             if (useLeftRotation) {
//                 newPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
//                       &(scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx]),
//                       numSegmentsPerCurve, *weightingIntegralPtr);
//             } else {
//                 newPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
//                       &(scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx]),
//                       numSegmentsPerCurve, *weightingIntegralPtr);
//             }

//             double lambdaLog10 = newPathWeightLog10 - oldPathWeightLog10;

//             double weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
//             while (weightAcceptance == 0) {
//                 weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
//             }

//             double weightAcceptanceLog10 = std::log10(weightAcceptance);

//             for (uint32_t i = 0; i <= numSegmentsPerCurve; i++) {
//                 if (useLeftRotation) {
//                     globalPos[CurrentThreadPosStartIdx + i]
//                           = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + i];
//                 } else {
//                     globalPos[CurrentThreadPosStartIdx + i]
//                           = scratchPositionSpaceRight[CurrentThreadPosStartIdx + i];
//                 }
//             }

//             twisty::PerturbUtils::UpdateTangentsFromPos(&globalPos[CurrentThreadPosStartIdx],
//                   &globalTans[CurrentThreadTanStartIdx], numSegmentsPerCurve, boundaryConditions);

//             twisty::PerturbUtils::UpdateCurvaturesFromTangents(
//                   &globalTans[CurrentThreadTanStartIdx],
//                   &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
//                   boundaryConditions,
//                   (int32_t)m_experimentParams.weightingParameters.weightingMethod);

//             double scatteringWeight = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
//                   &(globalCurvatures[CurrentThreadCurvatureStartIdx]), numSegmentsPerCurve,
//                   *weightingIntegralPtr);

//             if (pathCount < numPathsToSkipPerThread) {
//                 // Skip
//             } else {
//                 // Else, contribute to the paths
//                 int64_t currentPathIdx
//                       = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
//                 assert(currentPathIdx >= numPathsPerThread * threadIdx);


//                 globalPathWeights[currentPathIdx] = scatteringWeight;

//                 // if (m_experimentParams.outputPathBatches)
//                 // {
//                 //     // Add the path to the path batch
//                 //     for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//                 //     {
//                 //         Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
//                 //         pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
//                 //     }
//                 //     numCurvesInBatch++;

//                 //     if (numCurvesInBatch == ExportPathBatchCacheSize)
//                 //     {
//                 //         m_exportPathBatchesMutex.lock();

//                 //         m_curvesMetadataFile << threadIdx << " ";
//                 //         m_curvesMetadataFile << outputIdx << " ";
//                 //         m_curvesMetadataFile << numCurvesInBatch << std::endl;

//                 //         m_curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//                 //         numCurvesInBatch = 0;
//                 //         outputIdx++;

//                 //         m_exportPathBatchesMutex.unlock();
//                 //     }
//                 // }
//             }
//         }

//         // if(m_experimentParams.outputPathBatches)
//         // {
//         //     if (numCurvesInBatch > 0)
//         //     {
//         //         m_exportPathBatchesMutex.lock();

//         //         m_curvesMetadataFile << threadIdx << " ";
//         //         m_curvesMetadataFile << outputIdx << " ";
//         //         m_curvesMetadataFile << numCurvesInBatch << std::endl;

//         //         m_curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//         //         numCurvesInBatch = 0;
//         //         outputIdx++;

//         //         m_exportPathBatchesMutex.unlock();
//         //     }
//         // }
//     }
// }
}