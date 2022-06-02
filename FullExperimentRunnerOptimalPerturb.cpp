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

const static int64_t ExportPathBatchCacheSize = 100000;

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
              m_experimentParams.weightingParameters, m_upInitialCurve->m_segmentLength);
    } else {
        lookupEvaluator = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(
              m_experimentParams.weightingParameters, m_upInitialCurve->m_segmentLength);
    }

    lookupEvaluator->ExportValues(m_experimentDirPath.string());

    twisty::PerturbUtils::BoundaryConditions boundaryConditions
          = m_upInitialCurve->GetBoundaryConditions();

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
    std::vector<Farlor::Vector3> initialCurveTangents(initialCurvePositions.size());
    std::vector<float> initialCurveCurvatures(initialCurvePositions.size() - 1);

    // Update tangents
    twisty::PerturbUtils::UpdateTangentsFromPos(initialCurvePositions.data(),
          initialCurveTangents.data(), m_upInitialCurve->m_numSegments, boundaryConditions);
    twisty::PerturbUtils::UpdateCurvaturesFromTangents(initialCurveTangents.data(),
          initialCurveCurvatures.data(), m_upInitialCurve->m_numSegments, boundaryConditions,
          (int32_t)m_experimentParams.weightingParameters.weightingMethod);

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

    std::vector<double> cachedSegmentWeights;
    std::vector<double> compressedWeightBuffer;

    // Only in this case do we allocate room
    if (m_experimentParams.perturbMethod != ExperimentRunner::PerturbMethod::GeometricRandom) {
        perThreadPositionScratchLeft = perThreadCurvePositions;
        perThreadTangentScratchLeft = perThreadCurveTangents;
        perThreadCurvatureScratchLeft = perThreadCurveCurvatures;

        perThreadPositionScratchRight = perThreadCurvePositions;
        perThreadTangentScratchRight = perThreadCurveTangents;
        perThreadCurvatureScratchRight = perThreadCurveCurvatures;

        cachedSegmentWeights.resize(m_experimentParams.numSegmentsPerCurve * numPurturbThreads);
        compressedWeightBuffer.resize(ExportPathBatchCacheSize);
    }

    // Cached weights: num segments X num perturb threads
    // Stores the weights to read back from the experiment?
    // Export path batch cache size -> How many compressed paths we can track at once?
    // Per scatter value, which we will focus on one atm.


    std::stringstream fnFilenameSS;
    fnFilenameSS << "SavedFN";
    fnFilenameSS << m_experimentParams.numSegmentsPerCurve;
    fnFilenameSS << ".fnd";
    const std::filesystem::path fnFilePath = std::filesystem::current_path() / fnFilenameSS.str();
    std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> upFN = nullptr;

    // We dont need this actually, so we can just load the default one
    {
        // If we can load the fn data, load it
        if (std::filesystem::exists(fnFilePath)) {
            std::cout << "Using cached fd file at: " << fnFilePath << std::endl;
            std::ifstream inFile(fnFilePath);
            upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(inFile);
            inFile.close();
        }
        // We need to generate it this time and save it off to use next time
        else {
            // This is the max M value
            const int maxorder = m_upInitialCurve->m_numSegments;

            // Generate the fn data table
            const int numZSamples = 5000;
            const int numIntegrationSamples = 5000;

            // Arbitrarily set min and max |r_vec| values.
            // Why this specific max bound, I do not know
            const double rMin = 0.0;
            const double rMax = 200.0;
            upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(
                  numZSamples, numIntegrationSamples, maxorder, rMin, rMax);

            std::ofstream outFile(fnFilePath);
            dynamic_cast<PathWeighting::NormalizerStuff::FN *>(upFN.get())->WriteToFile(outFile);
            outFile.close();
        }
    }
    PathWeighting::NormalizerStuff::BaseNormalizer &fn = (*upFN);

    // Why the 1/(delta s) = (M+2)/s?
    Farlor::Vector3 Z = (boundaryConditions.m_endPos - boundaryConditions.m_startPos)
                * (m_upInitialCurve->m_numSegments + 2) / boundaryConditions.arclength
          - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
    std::cout << "Z: " << Z << std::endl;
    std::cout << "|Z|: " << Z.Magnitude() << std::endl;

    PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = 1.0;
    if (m_experimentParams.weightingParameters.weightingMethod
          == WeightingMethod::RadiativeTransfer) {
        pathNormalizer = PathWeighting::NormalizerStuff::Norm(
              fn, m_upInitialCurve->m_numSegments, Z.Magnitude(), boundaryConditions.arclength);
    }
    const boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10
          = boost::multiprecision::log10(pathNormalizer);

    std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
    std::cout << "PathNormalizerLog10: " << pathNormalizerLog10 << std::endl;

    auto setupTimeEnd = std::chrono::high_resolution_clock::now();

    long long setupTimeCount
          = std::chrono::duration_cast<std::chrono::milliseconds>(setupTimeEnd - setupTimeStart)
                  .count();
    ;
    /* --------------------- */


    /* --------------------- */
    const int64_t numCombinedWeightValues
          = (m_experimentParams.numPathsInExperiment + MaxNumPathsPerCombinedWeight - 1)
          / MaxNumPathsPerCombinedWeight;

    // const int64_t numDispatches = (m_experimentParams.numPathsInExperiment + ExportPathBatchCacheSize - 1) / ExportPathBatchCacheSize;
    std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
    // std::cout << "numPathsPerBatch: " << ExportPathBatchCacheSize << std::endl;
    // std::cout << "Num dispatches required: " << numDispatches << std::endl;
    std::cout << "Num combined weight values needed: " << numCombinedWeightValues << std::endl;

    // We need to calculate the absorbtion/scattering piece
    boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;

    long long perturbTimeCount = 0;
    long long weightCalcTimeCount = 0;

    assert(lookupEvaluator);
    twisty::PathWeighting::BaseWeightLookupTable &weightingIntegralsRawPointer = (*lookupEvaluator);

    std::vector<CombinedWeightValues_C> perDispatchCombinedWeightValues(numCombinedWeightValues);
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
                    case ExperimentRunner::PerturbMethod::GeometricMinCurvature: {
                        std::cout << "Error: GeometricMinCurvature Not implemented" << std::endl;
                        exit(1);
                    } break;

                    case ExperimentRunner::PerturbMethod::GeometricCombined: {
                        std::cout << "Error: GeometricMinCurvature Not Supported currently"
                                  << std::endl;
                        exit(1);
                    } break;

                    case ExperimentRunner::PerturbMethod::GeometricRandom:
                    default: {
                        std::thread newThread(&FullExperimentRunnerOptimalPerturb::GeometryRandom,
                              this, threadIdx, m_experimentParams.numPathsInExperiment,
                              numPathsPerThread, m_experimentParams.numPathsToSkip,
                              m_experimentParams.numSegmentsPerCurve,
                              std::ref(perThreadRngGenerators), std::ref(perThreadCurvePositions),
                              std::ref(perThreadCurveTangents), std::ref(perThreadCurveCurvatures),
                              std::ref(perDispatchCombinedWeightValues),
                              m_upInitialCurve->m_segmentLength,
                              lookupEvaluator->AccessLookupTable().data(),
                              lookupEvaluator->AccessLookupTable().size(), lookupEvaluator->GetDs(),
                              lookupEvaluator->GetMinCurvature(),
                              lookupEvaluator->GetMaxCurvature(),
                              lookupEvaluator->GetCurvatureStepSize(), boundaryConditions,
                              m_experimentParams.weightingParameters.weightingMethod
                                          == twisty::WeightingMethod::RadiativeTransfer
                                    ? pathNormalizerLog10.convert_to<double>()
                                    : 0.0);
                        threads[threadIdx] = std::move(newThread);
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

        boost::multiprecision::cpp_dec_float_100 totalDispatchWeight = 0.0;
        for (auto &combinedWeightValue : perDispatchCombinedWeightValues) {
            boost::multiprecision::cpp_dec_float_100 extractedDispatchWeight
                  = ExtractFinalValue(combinedWeightValue);
            totalDispatchWeight += extractedDispatchWeight;

            if (m_experimentParams.outputBigFloatWeights) {
                UpdateConvergenceWeight(combinedWeightValue.m_numValues, extractedDispatchWeight);
            }
        }

        std::cout << "Dispatch Weight: " << totalDispatchWeight << std::endl;
        std::cout << "\tAverage Path Weight: "
                  << totalDispatchWeight / m_experimentParams.numPathsInExperiment << std::endl;
        bigTotalExperimentWeight += totalDispatchWeight;

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

    ExperimentResults results;
    results.experimentWeights.push_back(bigTotalExperimentWeight);
    results.totalPathsGenerated = m_experimentParams.numPathsInExperiment;
    results.numFailedPaths = 0;

    ExperimentRunner::RunnerSpecificResults specificResult;
    specificResult.experimentResults = std::make_optional<ExperimentResults>(results);
    specificResult.setupMs = setupTimeCount;
    specificResult.runExperimentMs = perturbTimeCount;
    specificResult.weightingMs = weightCalcTimeCount;

    return specificResult;
}

void FullExperimentRunnerOptimalPerturb::WeightCombineThreadKernel(const int64_t threadIdx,
      int64_t numWeights, int64_t numWeightsPerThread, float arclength, int64_t numSegmentsPerCurve,
      const twisty::WeightingMethod weightingMethod, const std::vector<double> &compressedWeights,
      std::vector<boost::multiprecision::cpp_dec_float_100> &bigFloatWeightsLog10,
      std::vector<boost::multiprecision::cpp_dec_float_100> &threadScatterWeights,
      boost::multiprecision::cpp_dec_float_100 pathNormalizer,
      boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10)
{
    for (int64_t i = 0; i < numWeightsPerThread; i++) {
        int64_t idx = threadIdx * numWeightsPerThread + i;
        if (idx >= numWeights) {
            break;
        }

        // std::cout << "Compressed weight: " << compressedWeights[idx] << std::endl;

        boost::multiprecision::cpp_dec_float_100 bigFloatPathWeightLog10 = compressedWeights[idx];
        boost::multiprecision::cpp_dec_float_100 bigfloatPathWeight
              = boost::multiprecision::pow(10.0, bigFloatPathWeightLog10);
        // if (weightingMethod == WeightingMethod::RadiativeTransfer)
        // {
        //     bigFloatPathWeightLog10 += pathNormalizerLog10;
        //     bigfloatPathWeight *= pathNormalizer;
        // }

        // Pulled from Jerry analysis
        bigFloatWeightsLog10[idx] = bigFloatPathWeightLog10;
        threadScatterWeights[threadIdx] += bigfloatPathWeight;
    }
}

void FullExperimentRunnerOptimalPerturb::GeometryRandom(int64_t threadIdx,
      int64_t numExperimentPaths,
      int64_t numPathsPerThread,
      int64_t numPathsToSkipPerThread,
      int64_t numSegmentsPerCurve,
      std::vector<std::mt19937_64> &rngGenerators,
      std::vector<Farlor::Vector3> &globalPos,
      std::vector<Farlor::Vector3> &globalTans,
      std::vector<float> &globalCurvatures,
      std::vector<CombinedWeightValues_C> &combinedWeightValues,
      float segmentLength,
      const double *pWeightLookupTable,
      const int32_t weightLookupTableSize,
      const double ds,
      const double minCurvature,
      const double maxCurvature,
      const double curvatureStepSize,
      const twisty::PerturbUtils::BoundaryConditions &boundaryConditions,
      const double normalizerLog10)
{
    const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
    const int64_t NumTanPerCurve = (numSegmentsPerCurve + 1);
    const int64_t NumCurvaturesPerCurve = numSegmentsPerCurve;

    const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
    const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
    const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

    // Now, we can begin the actual algorithm
    {
        int64_t numCurvesInBatch = 0;
        int64_t outputIdx = 0;

        for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread);
              ++pathCount) {
            // Expect to go negative, thus int
            int64_t currentPathIdx
                  = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;

            // We can exit once this point is reached as we have generated all the paths necessary for this thread
            if (currentPathIdx >= numExperimentPaths) {
                printf("Exiting early\n");
                // We dont want to continue if we have already generated the correct number of paths.
                break;
            }

            // Do the perturb now

            std::uniform_int_distribution<int> diffDist(
                  2, std::min((int)(numSegmentsPerCurve - 2), 25));  // uniform, unbiased
            int64_t diff = diffDist(rngGenerators[threadIdx]);

            std::uniform_int_distribution<int> leftPointIndexUniformDist(
                  1, numSegmentsPerCurve - diff - 1);  // uniform, unbiased
            int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);

            int64_t rightPointIndex = leftPointIndex + diff;

            assert((rightPointIndex - leftPointIndex) >= diff);
            assert(leftPointIndex < rightPointIndex);

            // We need two frames for each segment to get the new curvature and torsion.
            // we need the frame left of the segment, as well as the frame right of the segment.
            // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
            const Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftPointIndex];
            const Farlor::Vector3 rightPoint
                  = globalPos[CurrentThreadPosStartIdx + rightPointIndex];

            const Farlor::Vector3 N = (rightPoint - leftPoint).Normalized();

            std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi, TwistyPi);
            double randRotationAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);

            //  Rotation
            {
                float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                RotationMatrixAroundAxis(randRotationAngle, N.m_data.data(), rotationMatrix);

                for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
                      ++pointIdx) {
                    Farlor::Vector3 shiftedPoint
                          = globalPos[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                    // Rotate and stuff back in shifted point
                    RotateVectorByMatrix(rotationMatrix, shiftedPoint.m_data.data());
                    // Update the point with the rotated version
                    globalPos[CurrentThreadPosStartIdx + pointIdx] = shiftedPoint + leftPoint;
                }

                //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                //We can do a different approach later.
                // Here, we want to do a perturb update call
                twisty::PerturbUtils::UpdateTangentsFromPos(&globalPos[CurrentThreadPosStartIdx],
                      &globalTans[CurrentThreadTanStartIdx], numSegmentsPerCurve,
                      boundaryConditions);

                twisty::PerturbUtils::UpdateCurvaturesFromTangents(
                      &globalTans[CurrentThreadTanStartIdx],
                      &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
                      boundaryConditions,
                      (int32_t)m_experimentParams.weightingParameters.weightingMethod);
            }


            // Normalized version
            // double scatteringWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(globalCurvatures[CurrentThreadCurvatureStartIdx]),
            //         numSegmentsPerCurve, *weightingIntegralPtr);
            double scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10_CudaSafe(
                        &(globalCurvatures[CurrentThreadCurvatureStartIdx]),
                        numSegmentsPerCurve,
                        pWeightLookupTable,
                        weightLookupTableSize,
                        ds,
                        minCurvature,
                        maxCurvature,
                        curvatureStepSize);
            scatteringWeightLog10 += normalizerLog10;

            if (pathCount < numPathsToSkipPerThread) {
                // Skip
            } else {
                // Else, contribute to the paths
                int64_t currentPathIdx
                      = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                assert(currentPathIdx >= numPathsPerThread * threadIdx);

                int combinedWeightValueIdx = (currentPathIdx / MaxNumPathsPerCombinedWeight);

                if ((currentPathIdx % MaxNumPathsPerCombinedWeight) == 0) {
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


void FullExperimentRunnerOptimalPerturb::GeometryCombined(int64_t threadIdx,
      int64_t numExperimentPaths,
      int64_t numPathsPerThread,
      int64_t numPathsToSkipPerThread,
      int64_t numSegmentsPerCurve,
      std::vector<std::mt19937_64> &rngGenerators,
      std::vector<Farlor::Vector3> &globalPos,
      std::vector<Farlor::Vector3> &globalTans,
      std::vector<float> &globalCurvatures,
      std::vector<Farlor::Vector3> &scratchPositionSpaceLeft,
      std::vector<Farlor::Vector3> &scratchTangentSpaceLeft,
      std::vector<float> &scratchCurvatureSpaceLeft,
      std::vector<Farlor::Vector3> &scratchPositionSpaceRight,
      std::vector<Farlor::Vector3> &scratchTangentSpaceRight,
      std::vector<float> &scratchCurvatureSpaceRight,
      std::vector<double> &globalPathWeights,
      std::vector<double> &cachedSegmentWeights,
      float segmentLength,
      twisty::PathWeighting::BaseWeightLookupTable *weightingIntegralPtr,
      const twisty::PerturbUtils::BoundaryConditions &boundaryConditions,
      const PathWeighting::NormalizerStuff::BaseNormalizer &pathNormalizer)
{
    // if (m_experimentParams.exportGeneratedCurves)
    // {
    //     // This should be per thread
    //     int64_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
    //     std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
    //     {
    //         for (int64_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
    //         {
    //             for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
    //             {
    //                 int64_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
    //                 pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
    //             }
    //         }
    //     }
    // }

    const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
    const int64_t NumTanPerCurve = (numSegmentsPerCurve + 1);
    const int64_t NumCurvaturesPerCurve = numSegmentsPerCurve;

    const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
    const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
    const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

    // Now, we can begin the actual algorithm
    {
        // This is the perturbation piece.
        // Can we do this in place, most likely
        // This will modify pCurrentThreadCurve
        // Remember, the structure of this is:
        // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M

        // Start at the thread's first path idx

        int64_t numCurvesInBatch = 0;
        int64_t outputIdx = 0;


        bool useOptimal = false;
        const uint32_t numRandom = 1000;
        const uint32_t numOptimal = 5000;
        uint32_t countCurrentMethod = 0;

        for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread);
              ++pathCount) {
            // Expect to go negative, thus int
            int64_t currentPathIdx
                  = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;

            // We can exit once this point is reached as we have generated all the paths necessary for this thread
            if (currentPathIdx >= numExperimentPaths) {
                // We dont want to continue if we have already generated the correct number of paths.
                break;
            }

            // Do the perturb now

            // Each time, we first copy the "old path" to the "scratch space"
            for (uint32_t segIdx = 0; segIdx <= numSegmentsPerCurve; ++segIdx) {
                scratchPositionSpaceLeft[CurrentThreadPosStartIdx + segIdx]
                      = globalPos[CurrentThreadPosStartIdx + segIdx];
                scratchPositionSpaceRight[CurrentThreadPosStartIdx + segIdx]
                      = globalPos[CurrentThreadPosStartIdx + segIdx];
            }

            // Update the tangents and curvatures
            {
                twisty::PerturbUtils::UpdateTangentsFromPos(
                      &scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
                      &scratchTangentSpaceLeft[CurrentThreadTanStartIdx], numSegmentsPerCurve,
                      boundaryConditions);

                twisty::PerturbUtils::UpdateCurvaturesFromTangents(
                      &scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
                      &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
                      numSegmentsPerCurve, boundaryConditions,
                      (int32_t)m_experimentParams.weightingParameters.weightingMethod);
            }

            // Update the right tangents and curvatures
            {
                twisty::PerturbUtils::UpdateTangentsFromPos(
                      &scratchPositionSpaceRight[CurrentThreadPosStartIdx],
                      &scratchTangentSpaceRight[CurrentThreadTanStartIdx], numSegmentsPerCurve,
                      boundaryConditions);

                twisty::PerturbUtils::UpdateCurvaturesFromTangents(
                      &scratchTangentSpaceRight[CurrentThreadTanStartIdx],
                      &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
                      numSegmentsPerCurve, boundaryConditions,
                      (int32_t)m_experimentParams.weightingParameters.weightingMethod);
            }

            std::uniform_int_distribution<int> diffDist(
                  2, std::min((int)(numSegmentsPerCurve - 2), 25));  // uniform, unbiased
            int64_t diff = diffDist(rngGenerators[threadIdx]);

            std::uniform_int_distribution<int> leftPointIndexUniformDist(
                  1, numSegmentsPerCurve - diff - 1);  // uniform, unbiased
            int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);

            int64_t rightPointIndex = leftPointIndex + diff;

            assert((rightPointIndex - leftPointIndex) >= diff);
            assert(leftPointIndex < rightPointIndex);

            // We need two frames for each segment to get the new curvature and torsion.
            // we need the frame left of the segment, as well as the frame right of the segment.
            // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
            const Farlor::Vector3 leftPoint
                  = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
            const Farlor::Vector3 rightPoint
                  = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];

            const Farlor::Vector3 N = (rightPoint - leftPoint).Normalized();

            double leftRotationAngle = 0.0;
            {
                const Farlor::Vector3 Xss1
                      = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex - 1];
                const Farlor::Vector3 Xs
                      = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
                const Farlor::Vector3 Xsp1
                      = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex + 1];
                const Farlor::Vector3 PL = Xs + ((Xsp1 - Xs).Dot(N)) * N;
                const Farlor::Vector3 ZL = Xss1 - ((Xss1 - PL).Dot(N)) * N;

                // Get side of plane Z is on
                const Farlor::Vector3 NL = N.Cross((Xsp1 - PL).Normalized()).Normalized();
                const double sideDistL = (ZL - PL).Dot(NL);

                const Farlor::Vector3 ZPnorm = (ZL - PL).Normalized();
                const Farlor::Vector3 Xsp1PLnorm = (Xsp1 - PL).Normalized();
                double cosAngle = ZPnorm.Dot(Xsp1PLnorm);
                cosAngle = std::max(-1.0, cosAngle);
                cosAngle = std::min(1.0, cosAngle);
                const double angle = acos(cosAngle);

                const double threshold = 10e-10;

                // In the case, we are aligned, we want a Pi rotation
                if (abs(angle) <= threshold) {
                    leftRotationAngle = TwistyPi;
                }
                // If we are Pi away, we want no rotation
                else if (abs(abs(angle) - TwistyPi) < threshold) {
                    leftRotationAngle = 0.0;
                }
                // On back side
                else if (sideDistL < 0.0) {
                    leftRotationAngle = TwistyPi - angle;
                }
                // On front side
                else {
                    leftRotationAngle = -1.0 * (TwistyPi - angle);
                }
            }


            double rightRotationAngle = 0.0;
            {
                const Farlor::Vector3 Xes1
                      = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex - 1];
                const Farlor::Vector3 Xe
                      = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];

                // Handle case where we are rotation with end point as pivot point
                Farlor::Vector3 Xep1(0.0f, 0.0f, 0.0f);
                if (rightPointIndex == numSegmentsPerCurve) {
                    Xep1 = Xe + boundaryConditions.m_endPos
                          + segmentLength * boundaryConditions.m_endDir.Normalized();
                } else {
                    Xep1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex + 1];
                }


                //const Farlor::Vector3 Xep1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex + 1];
                const Farlor::Vector3 PR = Xe + ((Xes1 - Xe).Dot(N)) * N;
                const Farlor::Vector3 ZR = Xep1 - ((Xep1 - PR).Dot(N)) * N;

                // Get side of plane Z is on
                const Farlor::Vector3 NR = N.Cross((Xes1 - PR).Normalized()).Normalized();
                const double sideDistR = (ZR - PR).Dot(NR);

                const Farlor::Vector3 ZPnorm = (ZR - PR).Normalized();
                const Farlor::Vector3 Xes1PLnorm = (Xes1 - PR).Normalized();
                double cosAngle = ZPnorm.Dot(Xes1PLnorm);
                cosAngle = std::max(-1.0, cosAngle);
                cosAngle = std::min(1.0, cosAngle);
                const double angle = acos(cosAngle);

                const double threshold = 10e-10;

                // In the case, we are aligned, we want a Pi rotation
                if (abs(angle) <= threshold) {
                    rightRotationAngle = TwistyPi;
                }
                // If we are Pi away, we want no rotation
                else if (abs(abs(angle) - TwistyPi) < threshold) {
                    rightRotationAngle = 0.0;
                }
                // On back side
                else if (sideDistR < 0.0) {
                    rightRotationAngle = TwistyPi - angle;
                }
                // On front side
                else {
                    rightRotationAngle = -1.0 * (TwistyPi - angle);
                }
            }

            // Overwrite angle
            if (!useOptimal) {
                countCurrentMethod++;
                if (countCurrentMethod >= numRandom) {
                    useOptimal = !useOptimal;
                }
                //double mult = (((pathCount / 2) % 2) == 0) ? 1 : -1;

                std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi, TwistyPi);
                //rotationAngle = TwistyPi * 0.75 * mult;
                double randRotationAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
                leftRotationAngle = randRotationAngle;
                rightRotationAngle = randRotationAngle;
                //std::cout << "Overwritten rotation angle: " << randRotationAngle << std::endl;
            } else {
                countCurrentMethod++;
                if (countCurrentMethod >= numOptimal) {
                    useOptimal = !useOptimal;
                }
            }

            // Left Rotation
            {
                float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                RotationMatrixAroundAxis(leftRotationAngle, (float *)(&N), rotationMatrix);

                for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
                      ++pointIdx) {
                    Farlor::Vector3 shiftedPoint
                          = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx]
                          - leftPoint;
                    // Rotate and stuff back in shifted point
                    RotateVectorByMatrix(rotationMatrix, (float *)(&shiftedPoint));
                    // Update the point with the rotated version
                    scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx]
                          = shiftedPoint + leftPoint;
                }

                //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                //We can do a different approach later.
                // Here, we want to do a perturb update call
                twisty::PerturbUtils::UpdateTangentsFromPos(
                      &scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
                      &scratchTangentSpaceLeft[CurrentThreadTanStartIdx], numSegmentsPerCurve,
                      boundaryConditions);

                twisty::PerturbUtils::UpdateCurvaturesFromTangents(
                      &scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
                      &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
                      numSegmentsPerCurve, boundaryConditions,
                      (int32_t)m_experimentParams.weightingParameters.weightingMethod);
            }

            // Right Rotation
            {
                float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                RotationMatrixAroundAxis(rightRotationAngle, (float *)(&N), rotationMatrix);

                for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex;
                      ++pointIdx) {
                    Farlor::Vector3 shiftedPoint
                          = scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx]
                          - leftPoint;
                    // Rotate and stuff back in shifted point
                    RotateVectorByMatrix(rotationMatrix, (float *)(&shiftedPoint));
                    // Update the point with the rotated version
                    scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx]
                          = shiftedPoint + leftPoint;
                }

                //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                //We can do a different approach later.
                // Here, we want to do a perturb update call
                twisty::PerturbUtils::UpdateTangentsFromPos(
                      &scratchPositionSpaceRight[CurrentThreadPosStartIdx],
                      &scratchTangentSpaceRight[CurrentThreadTanStartIdx], numSegmentsPerCurve,
                      boundaryConditions);

                twisty::PerturbUtils::UpdateCurvaturesFromTangents(
                      &scratchTangentSpaceRight[CurrentThreadTanStartIdx],
                      &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
                      numSegmentsPerCurve, boundaryConditions,
                      (int32_t)m_experimentParams.weightingParameters.weightingMethod);
            }

            double leftPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                  &(scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx]), numSegmentsPerCurve,
                  *weightingIntegralPtr);

            double rightPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                  &(scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx]),
                  numSegmentsPerCurve, *weightingIntegralPtr);

            bool useLeftRotation = (leftPathWeightLog10 >= rightPathWeightLog10);

            uint32_t numBetas = (rightPointIndex - 1) - leftPointIndex;
            std::vector<Farlor::Vector3> oldBetas(numBetas);
            std::vector<Farlor::Vector3> newBetas(numBetas);
            for (int64_t tanIdx = 0; tanIdx < numBetas; ++tanIdx) {
                oldBetas[tanIdx] = globalTans[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];

                if (useLeftRotation) {
                    newBetas[tanIdx] = scratchTangentSpaceLeft[CurrentThreadTanStartIdx
                          + leftPointIndex + tanIdx];
                } else {
                    newBetas[tanIdx] = scratchTangentSpaceRight[CurrentThreadTanStartIdx
                          + leftPointIndex + tanIdx];
                }
            }

            // Now we have a candidate path
            // We perform metropolis and see if we want to accept the path, i.e. copy the scratch space values to the actual curve values, or reroll a new curve
            std::uniform_real_distribution<double> uniformRandomZeroOne(0.0, 1.0);

            double acceptanceProb = uniformRandomZeroOne(rngGenerators[threadIdx]);

            // For now, we always accept a path
            // This is not using the metropolis sampling at all
            double oldPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                  &(globalCurvatures[CurrentThreadCurvatureStartIdx]), numSegmentsPerCurve,
                  *weightingIntegralPtr);

            double newPathWeightLog10 = 0.0;

            if (useLeftRotation) {
                newPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                      &(scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx]),
                      numSegmentsPerCurve, *weightingIntegralPtr);
            } else {
                newPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                      &(scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx]),
                      numSegmentsPerCurve, *weightingIntegralPtr);
            }

            double lambdaLog10 = newPathWeightLog10 - oldPathWeightLog10;

            double weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
            while (weightAcceptance == 0) {
                weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
            }

            double weightAcceptanceLog10 = std::log10(weightAcceptance);

            for (uint32_t i = 0; i <= numSegmentsPerCurve; i++) {
                if (useLeftRotation) {
                    globalPos[CurrentThreadPosStartIdx + i]
                          = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + i];
                } else {
                    globalPos[CurrentThreadPosStartIdx + i]
                          = scratchPositionSpaceRight[CurrentThreadPosStartIdx + i];
                }
            }

            twisty::PerturbUtils::UpdateTangentsFromPos(&globalPos[CurrentThreadPosStartIdx],
                  &globalTans[CurrentThreadTanStartIdx], numSegmentsPerCurve, boundaryConditions);

            twisty::PerturbUtils::UpdateCurvaturesFromTangents(
                  &globalTans[CurrentThreadTanStartIdx],
                  &globalCurvatures[CurrentThreadCurvatureStartIdx], numSegmentsPerCurve,
                  boundaryConditions,
                  (int32_t)m_experimentParams.weightingParameters.weightingMethod);

            double scatteringWeight = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                  &(globalCurvatures[CurrentThreadCurvatureStartIdx]), numSegmentsPerCurve,
                  *weightingIntegralPtr);

            if (pathCount < numPathsToSkipPerThread) {
                // Skip
            } else {
                // Else, contribute to the paths
                int64_t currentPathIdx
                      = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                assert(currentPathIdx >= numPathsPerThread * threadIdx);


                globalPathWeights[currentPathIdx] = scatteringWeight;

                // if (m_experimentParams.outputPathBatches)
                // {
                //     // Add the path to the path batch
                //     for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                //     {
                //         Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
                //         pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
                //     }
                //     numCurvesInBatch++;

                //     if (numCurvesInBatch == ExportPathBatchCacheSize)
                //     {
                //         m_exportPathBatchesMutex.lock();

                //         m_curvesMetadataFile << threadIdx << " ";
                //         m_curvesMetadataFile << outputIdx << " ";
                //         m_curvesMetadataFile << numCurvesInBatch << std::endl;

                //         m_curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                //         numCurvesInBatch = 0;
                //         outputIdx++;

                //         m_exportPathBatchesMutex.unlock();
                //     }
                // }
            }
        }

        // if(m_experimentParams.outputPathBatches)
        // {
        //     if (numCurvesInBatch > 0)
        //     {
        //         m_exportPathBatchesMutex.lock();

        //         m_curvesMetadataFile << threadIdx << " ";
        //         m_curvesMetadataFile << outputIdx << " ";
        //         m_curvesMetadataFile << numCurvesInBatch << std::endl;

        //         m_curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
        //         numCurvesInBatch = 0;
        //         outputIdx++;

        //         m_exportPathBatchesMutex.unlock();
        //     }
        // }
    }
}
}