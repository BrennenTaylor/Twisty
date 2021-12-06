#include "FullExperimentRunnerOptimalPerturbOptimized.h"

#include "CurvePerturbUtils.h"
#include "CurveUtils.h"
#include "MathConsts.h"

#include <omp.h>

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

//#define HardcodedSegments
//#define HardcodedDifference
//#define HardcodedRotation

// TODO: Make this modifiable
const uint32_t PathsPerScaledWeightValue = 100000;

namespace twisty
{
    static Farlor::Matrix3x3 RotationMatrixAroundAxis(float angle, Farlor::Vector3 axis)
    {
        // Ensure its normalized
        axis.Normalize();

        Farlor::Matrix3x3 rotation(
            Farlor::Vector3(
                cos(angle) + axis.x * axis.x * (1.0f - cos(angle)),
                axis.x * axis.y * (1.0f - cos(angle)) - axis.z * sin(angle),
                axis.x * axis.z * (1.0f - cos(angle)) + axis.y * sin(angle)
            ),
            Farlor::Vector3(
                axis.y * axis.x * (1.0f - cos(angle)) + axis.z * sin(angle),
                cos(angle) + axis.y * axis.y * (1 - cos(angle)),
                axis.y * axis.z * (1 - cos(angle)) - axis.x * sin(angle)
            ),
            Farlor::Vector3(
                axis.z * axis.x * (1 - cos(angle)) - axis.y * sin(angle),
                axis.z * axis.y * (1 - cos(angle)) + axis.x * sin(angle),
                cos(angle) + axis.z * axis.z * (1 - cos(angle))
            )
        );
        return rotation;
    }


    FullExperimentRunnerOptimalPerturbOptimized::FullExperimentRunnerOptimalPerturbOptimized(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper, uint32_t numCombinedValuesForAvg, uint32_t numCombinedValuesForMax)
        : ExperimentRunner(experimentParams, bootstrapper)
        , m_numCombinedValuesForAvg(numCombinedValuesForAvg)
        , m_numCombinedValuesForMax(numCombinedValuesForMax)
    {
    }

    FullExperimentRunnerOptimalPerturbOptimized::~FullExperimentRunnerOptimalPerturbOptimized()
    {
    }

    bool FullExperimentRunnerOptimalPerturbOptimized::Setup()
    {
        // Ask the bootstrapper to generate a discrete curve.
        // If we fail, we want to exit the experiment.
        bool successfulGen = false;
        while (!successfulGen)
        {
            m_upInitialCurve = m_bootstrapper.CreateCurve(m_experimentParams.numSegmentsPerCurve, m_experimentParams.arclength,
                m_experimentParams.bootstrapSeed);
            if (!m_upInitialCurve)
            {
                printf("Failed to create bootstrap curve.\n");
                return false;
            }

            // Lets also get the error of the initial curve, just to know
            const float curveError = CurveUtils::CalculateCurveError(*m_upInitialCurve);
            std::cout << "Seed curve error: " << curveError << std::endl;

            if (curveError < m_experimentParams.maximumBootstrapCurveError)
            {
                successfulGen = true;
            }
        }

        const std::filesystem::path experimentDirPath = m_experimentParams.experimentDirPath;
        if (!std::filesystem::exists(experimentDirPath))
        {
            std::filesystem::create_directories(experimentDirPath);
        }

        return true;
    }

    ExperimentRunner::ExperimentResults FullExperimentRunnerOptimalPerturbOptimized::RunExperiment()
    {
        int64_t numFailures = 0;
        int64_t totalFailures = 0;
        int64_t totalSuccess = 0;

        /* --------------------- */
        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();
        auto setupTimeStart = std::chrono::high_resolution_clock::now();
        /* --------------------- */

        const double ds = m_upInitialCurve->m_arclength / m_experimentParams.numSegmentsPerCurve;
        twisty::PathWeighting::WeightLookupTableIntegral lookupEvaluator(m_experimentParams.weightingParameters, ds);
        
        twisty::PerturbUtils::BoundaryConditions boundaryConditions;
        boundaryConditions.arclength = m_upInitialCurve->m_arclength;
        boundaryConditions.m_startPos = m_upInitialCurve->m_basePos;
        boundaryConditions.m_startDir = m_upInitialCurve->m_baseTangent;
        boundaryConditions.m_endPos = m_upInitialCurve->m_targetPos;
        boundaryConditions.m_endDir = m_upInitialCurve->m_targetTangent;
        
        // Constants
        double minCurvature = 0.0;
        double maxCurvature = 0.0;
        // TODO: Change this to float rather than double
        twisty::PathWeighting::CalcMinMaxCurvature(minCurvature, maxCurvature, ds);
        const float curvatureStepSize = (maxCurvature - minCurvature) / m_experimentParams.weightingParameters.numCurvatureSteps;

        // This is a horible hack to make sure we always purturb a new curve
        Curve curveToBend = *m_upInitialCurve;

        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        // Create threads and dispatch them
        int64_t numPerturbThreads = std::thread::hardware_concurrency();
        std::cout << "We have " << numPerturbThreads << " avalible for purturbation." << std::endl;

        // Setup rng stuff
        std::vector<std::mt19937_64> perThreadRngGenerators(numPerturbThreads);
        int64_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }
        for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
        {
            perThreadRngGenerators[threadIdx] = std::mt19937_64(seed + threadIdx);
        }

        // Setup data structures
        std::vector<Farlor::Vector3> initialCurvePositions(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> initialCurveTangents(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<float> initialCurveCurvatures(m_experimentParams.numSegmentsPerCurve);

        // Positions
        // Hard code the first two positions
        initialCurvePositions[0] = m_upInitialCurve->m_basePos;
        initialCurvePositions[1] = m_upInitialCurve->m_basePos + m_upInitialCurve->m_baseTangent.Normalized() * m_upInitialCurve->m_segmentLength;
        for (int64_t segmentIdx = 2; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
        {
            initialCurvePositions[segmentIdx] = m_upInitialCurve->m_positions[segmentIdx];
        }
        // Hard code the final position
        initialCurvePositions[m_experimentParams.numSegmentsPerCurve] = m_upInitialCurve->m_targetPos;

        twisty::PerturbUtils::UpdateTangentsFromPos(initialCurvePositions.data(), initialCurveTangents.data(),
            m_upInitialCurve->m_numSegments, boundaryConditions);

        twisty::PerturbUtils::UpdateCurvaturesFromTangents(initialCurveTangents.data(), initialCurveCurvatures.data(),
            m_upInitialCurve->m_numSegments, boundaryConditions, m_experimentParams.weightingParameters);

        const int64_t NumPosPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumTanPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumCurvaturePerCurve = (m_upInitialCurve->m_numSegments);

        std::vector<Farlor::Vector3> perThreadCurvePositions(NumPosPerCurve * numPerturbThreads);
        std::vector<Farlor::Vector3> perThreadCurveTangents(NumTanPerCurve * numPerturbThreads);
        std::vector<float> perThreadCurveCurvatures(NumCurvaturePerCurve * numPerturbThreads);

        std::vector<Farlor::Vector3> perThreadPositionScratchLeft(NumPosPerCurve * numPerturbThreads);
        std::vector<Farlor::Vector3> perThreadTangentScratchLeft(NumTanPerCurve * numPerturbThreads);
        std::vector<float> perThreadCurvatureScratchLeft(NumCurvaturePerCurve * numPerturbThreads);

        std::vector<Farlor::Vector3> perThreadPositionScratchRight(NumPosPerCurve * numPerturbThreads);
        std::vector<Farlor::Vector3> perThreadTangentScratchRight(NumTanPerCurve * numPerturbThreads);
        std::vector<float> perThreadCurvatureScratchRight(NumCurvaturePerCurve * numPerturbThreads);

        // TODO: Should this be intermixed somehow for better performance?
        for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
        {
            // Copy Pos
            for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
            {
                perThreadCurvePositions[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
                perThreadPositionScratchLeft[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
                perThreadPositionScratchRight[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
            }

            // Copy Tan
            for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                perThreadCurveTangents[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
                perThreadTangentScratchLeft[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
                perThreadTangentScratchRight[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
            }

            // Copy Curvatures
            for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
            {
                perThreadCurveCurvatures[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
                perThreadCurvatureScratchLeft[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
                perThreadCurvatureScratchRight[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
            }
        }

        // TODO: Is this cache even used?
        std::vector<double> cachedSegmentWeights(m_experimentParams.numSegmentsPerCurve * numPerturbThreads);

        std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> upFN = PathWeighting::NormalizerStuff::GetNormalizer(m_experimentParams.numSegmentsPerCurve);
        PathWeighting::NormalizerStuff::BaseNormalizer& fn = *upFN;

        Farlor::Vector3 Z = (boundaryConditions.m_endPos - boundaryConditions.m_startPos) * (m_upInitialCurve->m_numSegments + 2) / boundaryConditions.arclength
            - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
        std::cout << "Z: " << Z << std::endl;
        std::cout << "|Z|: " << Z.Magnitude() << std::endl;

        PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = PathWeighting::NormalizerStuff::Norm(fn, m_upInitialCurve->m_numSegments,
            Z.Magnitude(), boundaryConditions.arclength);
        std::cout << "PathNormalizer: " << pathNormalizer << std::endl;

        auto setupTimeEnd = std::chrono::high_resolution_clock::now();
        /* --------------------- */


        /* --------------------- */

        long long perturbTimeCount = 0;
        long long weightCalcTimeCount = 0;

        std::cout << "numPathsInExperiment specified: " << m_experimentParams.numPathsInExperiment << std::endl;

        const uint32_t numCombinedWeightValuesTotal = m_numCombinedValuesForAvg * m_numCombinedValuesForMax;
        const uint32_t numCombinedWeightValuesPerThread = (numCombinedWeightValuesTotal + numPerturbThreads - 1) / numPerturbThreads;

        std::cout << "numPathsInExperiment generated: " << numCombinedWeightValuesTotal * twisty::FullExperimentRunnerOptimalPerturbOptimized::CombinedWeightValues::MaxNumberOfPaths << std::endl;

        std::vector<twisty::FullExperimentRunnerOptimalPerturbOptimized::CombinedWeightValues> combinedWeightValues(numCombinedWeightValuesTotal);

        auto perturbTimeStart = std::chrono::high_resolution_clock::now();

        {
            std::vector<std::thread> threads(numPerturbThreads);
            for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
            {
                std::thread newThread(&FullExperimentRunnerOptimalPerturbOptimized::GeometryPerturb, this,
                    threadIdx,
                    numCombinedWeightValuesTotal,
                    numCombinedWeightValuesPerThread,
                    m_experimentParams.numPathsToSkip,
                    m_experimentParams.numSegmentsPerCurve,
                    std::ref(perThreadRngGenerators),
                    std::ref(initialCurvePositions),
                    std::ref(initialCurveTangents),
                    std::ref(initialCurveCurvatures),
                    std::ref(perThreadCurvePositions),
                    std::ref(perThreadCurveTangents),
                    std::ref(perThreadCurveCurvatures),
                    std::ref(perThreadPositionScratchLeft),
                    std::ref(perThreadTangentScratchLeft),
                    std::ref(perThreadCurvatureScratchLeft),
                    std::ref(perThreadPositionScratchRight),
                    std::ref(perThreadTangentScratchRight),
                    std::ref(perThreadCurvatureScratchRight),
                    std::ref(combinedWeightValues),
                    std::ref(cachedSegmentWeights),
                    m_upInitialCurve->m_segmentLength,
                    lookupEvaluator,
                    boundaryConditions,
                    fn
                );

                threads[threadIdx] = std::move(newThread);
            }

            for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
            {
                if (threads[threadIdx].joinable())
                {
                    threads[threadIdx].join();
                }
            }
        }

        auto perturbTimeEnd = std::chrono::high_resolution_clock::now();
        perturbTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(perturbTimeEnd - perturbTimeStart).count();

        // -------------------
        auto weightingTimeStart = std::chrono::high_resolution_clock::now();

        // We need to calculate the absorbtion/scattering piece
        boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;
        //// No, we calculating the weighting
        //for (auto& combinedWeightValue : combinedWeightValues)
        //{
        //    bigTotalExperimentWeight += combinedWeightValue.ExtractFinalValue();
        //}

        for (uint32_t avgIdx = 0; avgIdx < m_numCombinedValuesForAvg; ++avgIdx)
        {
            boost::multiprecision::cpp_dec_float_100 maxResult = 0.0;
            for (uint32_t maxIdx = 0; maxIdx < m_numCombinedValuesForMax; ++maxIdx)
            {
                auto& combinedWeightValue = combinedWeightValues[maxIdx + avgIdx * m_numCombinedValuesForMax];
                auto value = combinedWeightValue.ExtractFinalValue();
                if (value > maxResult)
                {
                    maxResult = value;
                }
            }

            bigTotalExperimentWeight += (maxResult * (1.0 / m_numCombinedValuesForAvg));
        }
        bigTotalExperimentWeight *= pathNormalizer;

        auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
        weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightingTimeEnd - weightingTimeStart).count();
        /* --------------------- */


        auto runExperimentTimeEnd = std::chrono::high_resolution_clock::now();

        std::cout << "Experiment Time Reporting: " << std::endl;
        auto runExperimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(runExperimentTimeEnd - runExperimentTimeStart);
        std::cout << "\tTotal Experiment Time: " << runExperimentTimeMs.count() << "ms" << std::endl;

        {
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupTimeEnd - setupTimeStart);
            std::cout << "\tsetup Time: " << timeMs.count() << "ms - " << ((float)timeMs.count() / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tperturb Time: " << perturbTimeCount << "ms - " << ((float)perturbTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tweighting Time: " << weightCalcTimeCount << "ms - " << ((float)weightCalcTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        ExperimentResults results;
        results.experimentWeight = bigTotalExperimentWeight;
        results.totalPathsGenerated = numCombinedWeightValuesTotal * twisty::FullExperimentRunnerOptimalPerturbOptimized::CombinedWeightValues::MaxNumberOfPaths;
        results.numFailedPaths = 0;
        return results;
    }


    void FullExperimentRunnerOptimalPerturbOptimized::GeometryPerturb(
        int64_t threadIdx,
        int64_t numCombinedWeightValuesTotal,
        int64_t numCombinedWeightValuesPerThread,
        int64_t numPathsToSkipPerThread,
        int64_t numSegmentsPerCurve,
        std::vector<std::mt19937_64>& rngGenerators,
        std::vector<Farlor::Vector3>& initialCurvePositions,
        std::vector<Farlor::Vector3>& initialCurveTangents,
        std::vector<float>& initialCurveCurvatures,
        std::vector<Farlor::Vector3>& globalPos,
        std::vector<Farlor::Vector3>& globalTans,
        std::vector<float>& globalCurvatures,
        std::vector<Farlor::Vector3>& scratchPositionSpaceLeft,
        std::vector<Farlor::Vector3>& scratchTangentSpaceLeft,
        std::vector<float>& scratchCurvatureSpaceLeft,
        std::vector<Farlor::Vector3>& scratchPositionSpaceRight,
        std::vector<Farlor::Vector3>& scratchTangentSpaceRight,
        std::vector<float>& scratchCurvatureSpaceRight,
        std::vector<twisty::FullExperimentRunnerOptimalPerturbOptimized::CombinedWeightValues>& combinedWeightValues,
        std::vector<double>& cachedSegmentWeights,
        float segmentLength,
        const twisty::PathWeighting::WeightLookupTableIntegral& weightingIntegral,
        const twisty::PerturbUtils::BoundaryConditions& boundaryConditions,
        const PathWeighting::NormalizerStuff::BaseNormalizer& fn
    )
    {
        uint32_t numPathsAccepted = 0;
        uint32_t numPathsUnaccepted = 0;
        uint32_t numPathsUnacceptedTangentPdf = 0;
        uint32_t numPathsUnacceptedCurvaturePdf = 0;

        const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
        const int64_t NumTanPerCurve = (numSegmentsPerCurve + 1);
        const int64_t NumCurvaturesPerCurve = numSegmentsPerCurve;

        const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
        const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
        const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

        int64_t numToSkip = numPathsToSkipPerThread;

        // We need a loop over the batches
        for (int64_t combinedWeightValuesIdx = 0; combinedWeightValuesIdx < numCombinedWeightValuesPerThread; combinedWeightValuesIdx++)
        {
            int64_t combinedWeightValuesGlobalIdx = threadIdx * numCombinedWeightValuesPerThread + combinedWeightValuesIdx;
            if (combinedWeightValuesGlobalIdx > numCombinedWeightValuesTotal)
            {
                // We are done, no more batches need generated
                return;
            }


            // We want to reset the global stuff
            // TODO: Should this be intermixed somehow for better performance?

            // Copy Pos
            for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
            {
                globalPos[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
            }

            // Copy Tan
            for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                globalTans[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
            }

            // Copy Curvatures
            for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturesPerCurve; curvatureIdx++)
            {
                globalCurvatures[NumCurvaturesPerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
            }

            // Else, we want to run the perturbation algorithm
            // Now, we can begin the actual algorithm

            FullExperimentRunnerOptimalPerturbOptimized::CombinedWeightValues combinedWeightValue;
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

                for (int64_t pathCount = 0; pathCount < (numToSkip + FullExperimentRunnerOptimalPerturbOptimized::CombinedWeightValues::MaxNumberOfPaths); ++pathCount)
                {
                    // Do the perturb now
                    // Each time, we first copy the "old path" to the "scratch space"
                    for (uint32_t segIdx = 0; segIdx <= numSegmentsPerCurve; ++segIdx)
                    {
                        scratchPositionSpaceLeft[CurrentThreadPosStartIdx + segIdx] = globalPos[CurrentThreadPosStartIdx + segIdx];
                        scratchPositionSpaceRight[CurrentThreadPosStartIdx + segIdx] = globalPos[CurrentThreadPosStartIdx + segIdx];

                    }

                    // Update left
                    {
                        twisty::PerturbUtils::UpdateTangentsFromPos(&scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
                            &scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
                            numSegmentsPerCurve, boundaryConditions);

                        twisty::PerturbUtils::UpdateCurvaturesFromTangents(&scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
                            &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions, m_experimentParams.weightingParameters);
                    }

                    // Update right
                    {
                        twisty::PerturbUtils::UpdateTangentsFromPos(&scratchPositionSpaceRight[CurrentThreadPosStartIdx],
                            &scratchTangentSpaceRight[CurrentThreadTanStartIdx],
                            numSegmentsPerCurve, boundaryConditions);

                        twisty::PerturbUtils::UpdateCurvaturesFromTangents(&scratchTangentSpaceRight[CurrentThreadTanStartIdx],
                            &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions, m_experimentParams.weightingParameters);
                    }

#ifdef HardcodedSegments
                    int64_t leftPointIndex = 25;
                    int64_t rightPointIndex = 75;
#else

#if defined(HardcodedDifference)
                    int64_t diff = 20;
#else
                    // Diff can range from 2 to 10
                    std::uniform_int_distribution<int> diffDist(2, 180); // uniform, unbiased
                    int64_t diff = diffDist(rngGenerators[threadIdx]);
#endif

                    std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - 1 - diff); // uniform, unbiased
                    int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);

                    int64_t rightPointIndex = leftPointIndex + diff;

                    assert((rightPointIndex - leftPointIndex) >= diff);
#endif
                    assert(leftPointIndex < rightPointIndex);

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                    {
                        printf("Left point idx: %d\n", leftPointIndex);
                        printf("Right point idx: %d\n", rightPointIndex);
                    }
#endif
                    // We need two frames for each segment to get the new curvature and torsion.
                    // we need the frame left of the segment, as well as the frame right of the segment.
                    // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                    const Farlor::Vector3 leftPoint = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
                    const Farlor::Vector3 rightPoint = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];

                    const Farlor::Vector3 N = (rightPoint - leftPoint).Normalized();

                    double leftRotationAngle = 0.0;
                    {
#if defined(DetailedPurturb) && defined(SingleThreadMode)
                        printf("Axis before (%.6f, %.6f, %.6f)\n",
                            N[0], N[1], N[2]
                        );
#endif

                        const Farlor::Vector3 Xss1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex - 1];
                        const Farlor::Vector3 Xs = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
                        const Farlor::Vector3 Xsp1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex + 1];
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
                        if (abs(angle) <= threshold)
                        {
                            leftRotationAngle = TwistyPi;
                        }
                        // If we are Pi away, we want no rotation
                        else if (abs(abs(angle) - TwistyPi) < threshold)
                        {
                            leftRotationAngle = 0.0;
                        }
                        // On back side
                        else if (sideDistL < 0.0)
                        {
                            leftRotationAngle = TwistyPi - angle;
                        }
                        // On front side
                        else
                        {
                            leftRotationAngle = -1.0 * (TwistyPi - angle);
                        }

                        //std::cout << "Perturb: " << std::endl;
                        //std::cout << "\tN: " << N << std::endl;
                        //std::cout << "\tXss1: " << Xss1 << std::endl;
                        //std::cout << "\tXs: " << Xs << std::endl;
                        //std::cout << "\tXsp1: " << Xsp1 << std::endl;
                        //std::cout << "\tPL: " << PL << std::endl;
                        //std::cout << "\tZL: " << ZL << std::endl;

                        //std::cout << "\tNL: " << NL << std::endl;
                        //std::cout << "\tXsp1PLnorm: " << Xsp1PLnorm << std::endl;
                        //std::cout << "\tsideDistL: " << sideDistL << std::endl;
                        //std::cout << "\tcosAngle: " << cosAngle << std::endl;
                        //std::cout << "\tangle: " << angle << std::endl;
                        //std::cout << "\tleftRotationAngle: " << leftRotationAngle << std::endl;
                    }


                    double rightRotationAngle = 0.0;
                    {
                        //#if defined(DetailedPurturb) && defined(SingleThreadMode)
                        //                    printf("Axis before (%.6f, %.6f, %.6f)\n",
                        //                        N_R[0], N_R[1], N_R[2]
                        //                    );
                        //#endif

                        const Farlor::Vector3 Xes1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex - 1];
                        const Farlor::Vector3 Xe = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];
                        const Farlor::Vector3 Xep1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex + 1];
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
                        if (abs(angle) <= threshold)
                        {
                            rightRotationAngle = TwistyPi;
                        }
                        // If we are Pi away, we want no rotation
                        else if (abs(abs(angle) - TwistyPi) < threshold)
                        {
                            rightRotationAngle = 0.0;
                        }
                        // On back side
                        else if (sideDistR < 0.0)
                        {
                            rightRotationAngle = TwistyPi - angle;
                        }
                        // On front side
                        else
                        {
                            rightRotationAngle = -1.0 * (TwistyPi - angle);
                        }

                        //std::cout << "Perturb: " << std::endl;
                        //std::cout << "\tN: " << N << std::endl;
                        //std::cout << "\tXes1: " << Xes1 << std::endl;
                        //std::cout << "\tXe: " << Xe << std::endl;
                        //std::cout << "\tXep1: " << Xep1 << std::endl;
                        //std::cout << "\tPR: " << PR << std::endl;
                        //std::cout << "\tZR: " << ZR << std::endl;

                        //std::cout << "\tNR: " << NR << std::endl;
                        //std::cout << "\tXes1PLnorm: " << Xes1PLnorm << std::endl;
                        //std::cout << "\tcosAngle: " << cosAngle << std::endl;
                        //std::cout << "\tangle: " << angle << std::endl;
                        //std::cout << "\tsideDistR: " << sideDistR << std::endl;
                        //std::cout << "\trightRotationAngle: " << rightRotationAngle << std::endl;
                    }

                    // Overwrite angle
                    if (!useOptimal)
                    {
                        countCurrentMethod++;
                        if (countCurrentMethod >= numRandom)
                        {
                            useOptimal = !useOptimal;
                        }
                        //double mult = (((pathCount / 2) % 2) == 0) ? 1 : -1;

                        std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi, TwistyPi);
                        double randRotationAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
                        leftRotationAngle = randRotationAngle;
                        rightRotationAngle = randRotationAngle;
                        //std::cout << "Overwritten rotation angle: " << randRotationAngle << std::endl;
                    }
                    else
                    {
                        countCurrentMethod++;
                        if (countCurrentMethod >= numOptimal)
                        {
                            useOptimal = !useOptimal;
                        }
                    }

                    // Left Rotation
                    {
                        float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                        RotationMatrixAroundAxis(leftRotationAngle, (float*)(&N), rotationMatrix);

                        for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                        {
                            Farlor::Vector3 shiftedPoint = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                            // Rotate and stuff back in shifted point
                            RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));
                            // Update the point with the rotated version
                            scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx] = shiftedPoint + leftPoint;
                        }

                        //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                        //We can do a different approach later.
                        // Here, we want to do a perturb update call
                        twisty::PerturbUtils::UpdateTangentsFromPos(&scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
                            &scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
                            numSegmentsPerCurve, boundaryConditions);

                        twisty::PerturbUtils::UpdateCurvaturesFromTangents(&scratchTangentSpaceLeft[CurrentThreadTanStartIdx],
                            &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions, m_experimentParams.weightingParameters);
                    }

                    // Right Rotation
                    {
                        float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                        RotationMatrixAroundAxis(rightRotationAngle, (float*)(&N), rotationMatrix);

                        for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                        {
                            Farlor::Vector3 shiftedPoint = scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                            // Rotate and stuff back in shifted point
                            RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));
                            // Update the point with the rotated version
                            scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx] = shiftedPoint + leftPoint;
                        }

                        //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                        //We can do a different approach later.
                        // Here, we want to do a perturb update call
                        twisty::PerturbUtils::UpdateTangentsFromPos(&scratchPositionSpaceRight[CurrentThreadPosStartIdx],
                            &scratchTangentSpaceRight[CurrentThreadTanStartIdx],
                            numSegmentsPerCurve, boundaryConditions);

                        twisty::PerturbUtils::UpdateCurvaturesFromTangents(&scratchTangentSpaceRight[CurrentThreadTanStartIdx],
                            &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions, m_experimentParams.weightingParameters);
                    }

                    double leftPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx]),
                        numSegmentsPerCurve, weightingIntegral);

                    double rightPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx]),
                        numSegmentsPerCurve, weightingIntegral);

                    bool useLeftRotation = (leftPathWeightLog10 >= rightPathWeightLog10);

                    uint32_t numBetas = (rightPointIndex - 1) - leftPointIndex;
                    std::vector<Farlor::Vector3> oldBetas(numBetas);
                    std::vector<Farlor::Vector3> newBetas(numBetas);
                    for (int64_t tanIdx = 0; tanIdx < numBetas; ++tanIdx)
                    {
                        oldBetas[tanIdx] = globalTans[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];

                        if (useLeftRotation)
                        {
                            newBetas[tanIdx] = scratchTangentSpaceLeft[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];
                        }
                        else
                        {
                            newBetas[tanIdx] = scratchTangentSpaceRight[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];
                        }
                    }

                    // Now we have a candidate path
                    // We perform metropolis and see if we want to accept the path, i.e. copy the scratch space values to the actual curve values, or reroll a new curve
                    bool accepted = false;
                    {
                        std::uniform_real_distribution<double> uniformRandomZeroOne(0.0, 1.0);

                        double acceptanceProb = uniformRandomZeroOne(rngGenerators[threadIdx]);

                        // Calculate likeness
                        // TODO: We only need to cache the old one
                        twisty::PathWeighting::NormalizerStuff::NormalizerDoubleType likelihood =
                            twisty::PathWeighting::NormalizerStuff::CalculateLikelihood(fn, numSegmentsPerCurve, boundaryConditions,
                                oldBetas, newBetas);


                        double oldPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(globalCurvatures[CurrentThreadCurvatureStartIdx]),
                            numSegmentsPerCurve, weightingIntegral);

                        double newPathWeightLog10 = 0.0;

                        if (useLeftRotation)
                        {
                            newPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx]),
                                numSegmentsPerCurve, weightingIntegral);
                        }
                        else
                        {
                            newPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx]),
                                numSegmentsPerCurve, weightingIntegral);
                        }

                        double lambdaLog10 = newPathWeightLog10 - oldPathWeightLog10;

                        double weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
                        while (weightAcceptance == 0)
                        {
                            weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
                        }

                        double weightAcceptanceLog10 = std::log10(weightAcceptance);

                        accepted = true;
                        for (uint32_t i = 0; i <= numSegmentsPerCurve; i++)
                        {
                            if (useLeftRotation)
                            {
                                globalPos[CurrentThreadPosStartIdx + i] = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + i];
                            }
                            else
                            {
                                globalPos[CurrentThreadPosStartIdx + i] = scratchPositionSpaceRight[CurrentThreadPosStartIdx + i];
                            }
                        }

                        twisty::PerturbUtils::UpdateTangentsFromPos(&globalPos[CurrentThreadPosStartIdx],
                            &globalTans[CurrentThreadTanStartIdx],
                            numSegmentsPerCurve, boundaryConditions);

                        twisty::PerturbUtils::UpdateCurvaturesFromTangents(&globalTans[CurrentThreadTanStartIdx],
                            &globalCurvatures[CurrentThreadCurvatureStartIdx],
                            numSegmentsPerCurve, boundaryConditions, m_experimentParams.weightingParameters);
                    }

                    if (accepted)
                    {
                        numPathsAccepted++;

                        double pathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(globalCurvatures[CurrentThreadCurvatureStartIdx]),
                            numSegmentsPerCurve, weightingIntegral);

                        if (pathCount < numPathsToSkipPerThread)
                        {
                            // Skip
                        }
                        else
                        {
                            // Else, contribute to the paths
                            combinedWeightValue.AddValue(pathWeightLog10);
                        }
                    }
                    else
                    {
                        // Go back one path as we are redoing
                        pathCount--;
                        countCurrentMethod--;

                        numPathsUnaccepted++;
                    }
                }
            }
            combinedWeightValues[combinedWeightValuesGlobalIdx] = combinedWeightValue;
        }


        std::cout << "Num path accepted: " << numPathsAccepted << std::endl;
        std::cout << "Num path unaccepted: " << numPathsUnaccepted << std::endl;
        std::cout << "Num path unaccepted tangents: " << numPathsUnacceptedTangentPdf << std::endl;
        std::cout << "Num path unaccepted curvature: " << numPathsUnacceptedCurvaturePdf << std::endl;
    }

    void FullExperimentRunnerOptimalPerturbOptimized::Shutdown()
    {
    }
}