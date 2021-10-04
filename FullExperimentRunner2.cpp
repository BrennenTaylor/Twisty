#include "FullExperimentRunner2.h"

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

//#define DetailedPurturb
//#define SingleThreadMode

const double AmountOfFullRotation = 0.001;

const double MaxDoubleLog10 = 300;
const double PathBatchOffsetConstant = 6.0;

//#define HardcodedSegments
//#define HardcodedRotation
//#define HardcodednumPerturbThreadsFhybrid


#define SINGLE_THREAD_PERTURB_MODE
//#define OutputBigFloatPathWeights
//#define ExportPathBatches

#if defined(OutputBigFloatPathWeights)
#define SINGLE_THREAD_PERTURB_MODE
#endif

#if defined(ExportPathBatches)
#define SINGLE_THREAD_PERTURB_MODE
#endif

#if defined(ExportPathBatches)

const int64_t ExportPathBatchCacheSize = 1000000;

std::mutex ExportPathBatchesMutex;
std::ofstream curvesBinaryFile;
std::ofstream curvesMetadataFile;

#endif

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

namespace twisty
{
    FullExperimentRunner2::FullExperimentRunner2(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper)
        : ExperimentRunner(experimentParams, bootstrapper)
    {
    }

    FullExperimentRunner2::~FullExperimentRunner2()
    {
    }

    bool FullExperimentRunner2::Setup()
    {
        // Ask the bootstrapper to generate a discrete curve.
        // If we fail, we want to exit the experiment.

        bool successfulGen = false;
        while (!successfulGen)
        {
            m_upInitialCurve = m_bootstrapper.CreateCurve(m_experimentParams.numSegmentsPerCurve);
            if (!m_upInitialCurve)
            {
                printf("Failed to create bootstrap curve.\n");
                return false;
            }

            // Lets also get the error of the initial curve, just to know
            float curveError = CurveUtils::CalculateCurveError(*m_upInitialCurve);
            std::cout << "Seed curve error: " << curveError << std::endl;

            if (curveError < m_experimentParams.maximumBootstrapCurveError)
            {
                successfulGen = true;
            }
        }

        std::filesystem::path experimentDirectoryPath = m_experimentParams.experimentDirPath;
        if (!std::filesystem::exists(experimentDirectoryPath))
        {
            std::filesystem::create_directories(experimentDirectoryPath);
        }

        return true;
    }

    void FullExperimentRunner2::LogWeightThreadFunction(
        int64_t threadIdx,
        int64_t dispatchIdx,
        int64_t numPerturbThreads,
        int64_t numExperimentPaths,
        int64_t numPathsPerThread,
        int64_t numPathsToSkipPerThread,
        int64_t numSegmentsPerCurve,
        std::vector<std::mt19937_64>& rngGenerators,
        std::vector<Farlor::Vector3>& globalPos,
        std::vector<Farlor::Vector3>& globalTans,
        std::vector<float>& globalCurvatures,
        std::vector<double>& cachedSegmentWeights,
        std::vector<double>& finalThreadWeight,
        std::vector<double>& finalThreadDifference,
        std::vector<double>& runningPathWeightsLog10,
        float segmentLength,
        float scattering,
        float absorbtion,
        const std::vector<double>& lookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize
    )
    {
        // If we are outputing path batches, we want to generate a cache for the paths
#if defined(ExportPathBatches)

            // This should be per thread
            int64_t numPosPerCurve = (numSegmentsPerCurve + 1);
            int64_t numPosInCache = numPosPerCurve * ExportPathBatchCacheSize;
            std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
            {
                for (int64_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
                {
                    for (int64_t pointIdx = 0; pointIdx < numPosPerCurve; ++pointIdx)
                    {
                        int64_t pointEntryIdx = numPosPerCurve * cacheEntryIdx + pointIdx;
                        pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                    }
                }
            }
#endif

            // First, we discard random numbers to match the previous amount
            //rngGenerators[threadIdx].discard(numPathsPerThread * threadIdx);
            
            const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
            const int64_t NumTanPerCurve = (numSegmentsPerCurve + 1);
            const int64_t NumCurvaturesPerCurve = numSegmentsPerCurve;

            const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
            const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
            const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

            const double c = scattering + absorbtion;
            const double absorbtionConst = std::exp(-c * segmentLength) / (2.0 * TwistyPi * TwistyPi);
            const double absorbtionConstLog10 = std::log10(absorbtionConst);

            // We start by keeping a running path weight.
            // Lets make this a double actually...
            // Though this is a ln compressed version, so it shouuuuld be ok.

            double currentMaxWeightLog10 = 0.0;
            double currentMaxPossibleFinalWeightLog10 = 0.0;
            double currentDifference = 0.0;
            double runningTotal = 0.0;

            // This is a single path weight that is cached and updated each new path
            double runningSinglePathWeightLog10 = 0.0;
            {
                // Lets precache all the segment weights
                {
                    for (int64_t segIdx = 0; segIdx < numSegmentsPerCurve; ++segIdx)
                    {
                        // Extract curvature
                        float curvature = globalCurvatures[CurrentThreadCurvatureStartIdx + segIdx];

                        float distance = curvature - minCurvature;
                        float realIdx = distance / curvatureStepSize;
                        int64_t leftIdx = floor(realIdx);
                        int64_t rightIdx = leftIdx + 1;

                        double leftLookup = lookupTable[leftIdx];
                        double rightLookup = lookupTable[rightIdx];

                        float leftDist = distance - (leftIdx * curvatureStepSize);

                        double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                        // Take the natural log of the interpolated results
                        double interpolatedResultLog10 = std::log10(interpolatedResult);
                        // Lets do weights as doubles for now
                        double segmentWeightLog10 = interpolatedResultLog10 + absorbtionConstLog10;

                        // Update the running path weight. We also want to cache the segment weights
                        runningSinglePathWeightLog10 += segmentWeightLog10;
                        cachedSegmentWeights[numSegmentsPerCurve * threadIdx + segIdx] = segmentWeightLog10;
                    }
                }
            }

            // At this point, we have cached our weights for each initial curve segment
            {
                // Now, we can begin the actual algorithm
                {

                    // This is the perturbation piece.
                    // Can we do this in place, most likely
                    // This will modify pCurrentThreadCurve
                    // Remember, the structure of this is:
                    // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M

                    // Start at the thread's first path idx

#if defined(ExportPathBatches)
                    int64_t numCurvesInBatch = 0;
                    int64_t outputIdx = 0;
#endif

                    // Cache is per hardware thread
                    int64_t cacheStartPathIdx = numPathsPerThread * threadIdx;

                    // Path count represents the actually path generated in order, including the skipped paths
                    for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
                    {
                        // Expect to go negative, thus int
                        int64_t currentPathIdx = numPathsPerThread * threadIdx + (pathCount - numPathsToSkipPerThread);
                       
                        // We can exit once this point is reached as we have generated all the paths necessary for this thread
                        if (currentPathIdx >= numExperimentPaths)
                        {
#if defined(ExportPathBatches)
                            if (numCurvesInBatch > 0)
                            {
                                //std::cout << "Done before number of possible paths for thread, exporting: " << numCurvesInBatch << " paths" << std::endl;
                                //std::cout << "Path count: " << pathCount << std::endl;
                                //std::cout << "Current path idx: " << currentPathIdx << std::endl;
                                //std::cout << "NumExperimentPaths: " << numExperimentPaths << std::endl;

                                ExportPathBatchesMutex.lock();

                                curvesMetadataFile << threadIdx << " ";
                                curvesMetadataFile << outputIdx << " ";
                                curvesMetadataFile << numCurvesInBatch << std::endl;

                                curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                                
                                // Reset the number of curves and increase the output index
                                numCurvesInBatch = 0;
                                outputIdx++;

                                ExportPathBatchesMutex.unlock();
                            }
#endif


                            // We dont want to continue if we have already generated the correct number of paths.
                            // TODO: Will we need to do weights stuff...?
                            break;
                        }

                        // Do the perturb now
                        {
#ifdef HardcodedSegments
                            const int64_t leftPointIndex = 25;
                            const int64_t rightPointIndex = 75;
#else
                            std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - 3); // uniform, unbiased
                            const int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);
                            std::uniform_int_distribution<int> rightPointIndexUniformDist(leftPointIndex + 2, numSegmentsPerCurve - 1); // uniform, unbiased
                            const int64_t rightPointIndex = rightPointIndexUniformDist(rngGenerators[threadIdx]);
#endif

                            assert(leftPointIndex < rightPointIndex);
                            assert((rightPointIndex - leftPointIndex) >= 2);

                            // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                            const Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftPointIndex];
                            const Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightPointIndex];
                            const Farlor::Vector3 axisOfRotation = (rightPoint - leftPoint).Normalized();

#ifdef HardcodedRotation
                            const float randomAngle = 1.38f;
#else
                            std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * AmountOfFullRotation, TwistyPi * AmountOfFullRotation);
                            const float randomAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
#endif               
                            float rotationMatrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                            RotationMatrixAroundAxis(randomAngle, (float*)(&axisOfRotation), rotationMatrix);


                            // Actually rotate points around rotation axis
                            for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                            {
                                Farlor::Vector3 shiftedPoint = globalPos[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                                // Rotate and stuff back in shifted point
                                RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));

                                // Update the stored point with the rotated version
                                globalPos[CurrentThreadPosStartIdx + pointIdx] = shiftedPoint + leftPoint;
                            }

                            //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                            //We can do a different approach later.

                            // Left side
                            {
                                Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftPointIndex];
                                Farlor::Vector3 rightPos = globalPos[CurrentThreadPosStartIdx + leftPointIndex + 1];

                                globalTans[CurrentThreadTanStartIdx + leftPointIndex] = (rightPoint - leftPoint).Normalized();
                            }

                            // Right side
                            {
                                Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + rightPointIndex - 1];
                                Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightPointIndex];

                                globalTans[CurrentThreadTanStartIdx + rightPointIndex - 1] = (rightPoint - leftPoint).Normalized();
                            }


                            // Update left curvature
                            {
                                const Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + (leftPointIndex - 1)];
                                const Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + (leftPointIndex)];
                                const Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);
                                const float curvature = temp.Magnitude();
                                globalCurvatures[CurrentThreadCurvatureStartIdx + (leftPointIndex - 1)] = curvature;

                                // Also, cache the weight of that changed segment
                                const float distance = curvature - minCurvature;
                                const float realIdx = distance / curvatureStepSize;
                                const int64_t leftIdx = floor(realIdx);
                                const int64_t rightIdx = leftIdx + 1;

                                const double leftLookup = lookupTable[leftIdx];
                                const double rightLookup = lookupTable[rightIdx];

                                const float leftDist = distance - (leftIdx * curvatureStepSize);

                                const double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                                const double interpolatedResultLog10 = std::log10(interpolatedResult);
                                const double segmentWeightLog10 = interpolatedResultLog10 + absorbtionConstLog10;

                                // Udpate running weight and cache
                                runningSinglePathWeightLog10 -= cachedSegmentWeights[(leftPointIndex - 1) + (numSegmentsPerCurve * threadIdx)];
                                runningSinglePathWeightLog10 += segmentWeightLog10;
                                cachedSegmentWeights[(leftPointIndex - 1) + (numSegmentsPerCurve * threadIdx)] = segmentWeightLog10;
                            }

                            // Update right curvature
                            {
                                const Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + (rightPointIndex - 1)];
                                const Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + rightPointIndex];
                                const Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);
                                const float curvature = temp.Magnitude();
                                globalCurvatures[CurrentThreadCurvatureStartIdx + (rightPointIndex - 1)] = curvature;

                                // Also, cache the weight of that changed segment
                                const float distance = curvature - minCurvature;
                                const float realIdx = distance / curvatureStepSize;
                                const int64_t leftIdx = floor(realIdx);
                                const int64_t rightIdx = leftIdx + 1;

                                const double leftLookup = lookupTable[leftIdx];
                                const double rightLookup = lookupTable[rightIdx];
                                const float leftDist = distance - (leftIdx * curvatureStepSize);

                                const double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                                const double interpolatedResultLog10 = std::log10(interpolatedResult);
                                const double segmentWeightLog10 = interpolatedResultLog10 + absorbtionConstLog10;

                                // Remove old segmentWeight
                                runningSinglePathWeightLog10 -= cachedSegmentWeights[(rightPointIndex - 1) + (numSegmentsPerCurve * threadIdx)];
                                runningSinglePathWeightLog10 += segmentWeightLog10;
                                cachedSegmentWeights[(rightPointIndex - 1) + (numSegmentsPerCurve * threadIdx)] = segmentWeightLog10;
                            }
                        }

                        if (pathCount < numPathsToSkipPerThread)
                        {
                            // Skip
                            // We done want to output any weights or paths based on this
                            //std::cout << "Skip" << std::endl;
                        }
                        else
                        {
                            //std::cout << "No Skip" << std::endl;
                            // Else, contribute to the paths
                            // This is the index of the current path for all threads in this dispatch 
                            int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                            assert(currentPathIdx >= numPathsPerThread * threadIdx);

                            int64_t pathNumber = pathCount - numPathsToSkipPerThread;

                            // Log only the weights that we want to count
                            // We want this to store all the weights for this dispatch
                            runningPathWeightsLog10[currentPathIdx] = runningSinglePathWeightLog10;
                            
                            if (pathNumber == 0)
                            {
                                currentMaxWeightLog10 = runningSinglePathWeightLog10;
                                currentMaxPossibleFinalWeightLog10 = runningSinglePathWeightLog10 + PathBatchOffsetConstant;
                                currentDifference = MaxDoubleLog10 - currentMaxPossibleFinalWeightLog10;
                                runningTotal += std::pow(10, runningSinglePathWeightLog10 + currentDifference);
                            }
                            else
                            {
                                double newMaxWeightLog10 = runningSinglePathWeightLog10;
                                // If this checks out, we have the same maximum and thus can just adjust things up
                                if (currentMaxWeightLog10 > newMaxWeightLog10)
                                {
                                    runningTotal += std::pow(10, runningSinglePathWeightLog10 + currentDifference);
                                }
                                else
                                {
                                    // If we are past, then we have a new maximum and need to adjust
                                    // New difference
                                    double newMaxPossibleFinalWeightLog10 = newMaxWeightLog10 + PathBatchOffsetConstant;
                                    double newDifference = MaxDoubleLog10 - newMaxPossibleFinalWeightLog10;

                                    double differenceDelta = newDifference - currentDifference;
                                    double log10RunningTotal = std::log10(runningTotal);
                                    double adjustedLog10RunningTotal = log10RunningTotal + differenceDelta;
                                    runningTotal = std::pow(10.0, adjustedLog10RunningTotal);

                                    // Update
                                    currentMaxWeightLog10 = newMaxWeightLog10;
                                    currentMaxPossibleFinalWeightLog10 = newMaxPossibleFinalWeightLog10;
                                    currentDifference = newDifference;

                                    runningTotal += std::pow(10, (runningSinglePathWeightLog10 + currentDifference));
                                }
                            }

                            finalThreadWeight[threadIdx] = runningTotal;
                            finalThreadDifference[threadIdx] = currentDifference;

#if defined(ExportPathBatches)
                            // Add the path to the path batch
                            for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                            {
                                Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
                                pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
                            }
                            numCurvesInBatch++;
                            //std::cout << "Numcurvesinbatch udpate: " << numCurvesInBatch << std::endl;
#endif

#if defined(ExportPathBatches)
                            if (numCurvesInBatch == ExportPathBatchCacheSize)
                            {
                                ExportPathBatchesMutex.lock();

                                curvesMetadataFile << threadIdx << " ";
                                curvesMetadataFile << outputIdx << " ";
                                curvesMetadataFile << numCurvesInBatch << std::endl;

                                curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                                numCurvesInBatch = 0;
                                outputIdx++;

                                ExportPathBatchesMutex.unlock();
                            }
#endif
                        }
                    }

#if defined(ExportPathBatches)
                    if (numCurvesInBatch > 0)
                    {
                        std::cout << "Final cleanup, exporting: " << numCurvesInBatch << " paths" << std::endl;
                        ExportPathBatchesMutex.lock();

                        curvesMetadataFile << threadIdx << " ";
                        curvesMetadataFile << outputIdx << " ";
                        curvesMetadataFile << numCurvesInBatch << std::endl;

                        curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                        numCurvesInBatch = 0;
                        outputIdx++;

                        ExportPathBatchesMutex.unlock();
                    }
#endif
                }
            }
    }

    ExperimentRunner::ExperimentResults FullExperimentRunner2::RunExperiment()
    {
        return RunExperimentLogWeightTable();
    }

    ExperimentRunner::ExperimentResults FullExperimentRunner2::RunExperimentLogWeightTable()
    {
        int64_t numFailures = 0;
        int64_t totalFailures = 0;
        int64_t totalSuccess = 0;

        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();

        /* --------------------- */
        auto setupTimeStart = std::chrono::high_resolution_clock::now();

#if defined(ExportPathBatches)
        {
            BeginPathBatchOutput();

            std::filesystem::path currentPath = std::filesystem::current_path();
            currentPath.append(m_experimentParams.experimentDir);

            if (!std::filesystem::exists(currentPath))
            {
                std::filesystem::create_directories(currentPath);
            }

            std::stringstream pathBinaryFilenameSS;
            pathBinaryFilenameSS << m_experimentParams.pathBatchPrepend;
            pathBinaryFilenameSS << "Paths_Binary" << ".pbd";

            std::filesystem::path binaryFilePath = currentPath;
            binaryFilePath.append(pathBinaryFilenameSS.str());
            curvesBinaryFile.open(binaryFilePath, std::ios::binary);

            std::stringstream pathMetadataFilenameSS;
            pathMetadataFilenameSS << m_experimentParams.pathBatchPrepend;
            pathMetadataFilenameSS << "Paths_Metadata" << ".pmd";

            std::filesystem::path metadataFilePath = currentPath;
            metadataFilePath.append(pathMetadataFilenameSS.str());
            curvesMetadataFile.open(metadataFilePath);
        }
#endif

        // Say that we will start outputing the path batch output
        const double ds = m_upInitialCurve->m_arclength / m_experimentParams.numSegmentsPerCurve;
        twisty::PathWeighting::WeightLookupTableIntegral lookupEvaluator(m_experimentParams.weightingParameters, ds);

        // Constants
        double minCurvature = 0.0;
        double maxCurvature = 0.0;
        twisty::PathWeighting::CalcMinMaxCurvature(minCurvature, maxCurvature, ds);
        const float curvatureStepSize = (maxCurvature - minCurvature) / m_experimentParams.weightingParameters.numCurvatureSteps;

        // This is a horible hack to make sure we always purturb a new curve
        Curve curveToBend = *m_upInitialCurve;

        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        const int64_t NumPathsPerThread = 1000000;

        // Create threads and dispatch them
#ifdef SINGLE_THREAD_PERTURB_MODE
        int64_t numPerturbThreads = 1;
#else
#ifdef HardcodednumPerturbThreads
        int64_t numPerturbThreads = 3;
#else
        int64_t numPerturbThreads = std::thread::hardware_concurrency();
#endif
#endif
        std::cout << "We have " << numPerturbThreads << " avalible for purturbation." << std::endl;

        // Setup rng stuff
        std::vector<std::mt19937_64> perThreadRngGenerators(numPerturbThreads);
        int64_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }
        for (int64_t i = 0; i < numPerturbThreads; ++i)
        {
            perThreadRngGenerators[i] = std::mt19937_64(seed + i);
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

        // Tangents
        // Hardcode intial tangent
        initialCurveTangents[0] = m_upInitialCurve->m_baseTangent;
        for (int64_t tanIdx = 1; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
        {
            Farlor::Vector3 leftPos = initialCurvePositions[tanIdx];
            Farlor::Vector3 rightPos = initialCurvePositions[tanIdx + 1];

            initialCurveTangents[tanIdx] = (rightPos - leftPos).Normalized();
        }
        // Final Tangents
        initialCurveTangents[m_experimentParams.numSegmentsPerCurve] = m_upInitialCurve->m_targetTangent;

        // Curvatures
        float segLength = m_upInitialCurve->m_arclength / m_upInitialCurve->m_numSegments;
        for (int64_t curvatureIdx = 0; curvatureIdx < m_experimentParams.numSegmentsPerCurve; ++curvatureIdx)
        {
            Farlor::Vector3 tanLeft = initialCurveTangents[curvatureIdx];
            Farlor::Vector3 tanRight = initialCurveTangents[curvatureIdx + 1];

            Farlor::Vector3 curvatureVec = (tanRight - tanLeft) * (1.0f / segLength);
            float curvature = curvatureVec.Magnitude();
            initialCurveCurvatures[curvatureIdx] = curvature;
        }

        const int64_t NumPosPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumTanPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumCurvaturePerCurve = (m_upInitialCurve->m_numSegments);

        std::vector<Farlor::Vector3> perThreadCurvePositions(NumPosPerCurve * numPerturbThreads);
        std::vector<Farlor::Vector3> perThreadCurveTangents(NumTanPerCurve * numPerturbThreads);
        std::vector<float> perThreadCurveCurvatures(NumCurvaturePerCurve * numPerturbThreads);

        for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
        {
            // Copy Pos
            for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
            {
                perThreadCurvePositions[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
            }

            // Copy Tan
            for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                perThreadCurveTangents[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
            }

            // Copy Curvatures
            for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
            {
                perThreadCurveCurvatures[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
            }
        }

        std::vector<double> cachedSegmentWeights(m_experimentParams.numSegmentsPerCurve * numPerturbThreads);
        std::vector<double> compressedWeightBuffer(m_experimentParams.numPathsPerBatch);

        auto setupTimeEnd = std::chrono::high_resolution_clock::now();
        /* --------------------- */


        /* --------------------- */

        std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
        std::cout << "numPathsPerBatch: " << m_experimentParams.numPathsPerBatch << std::endl;
        std::cout << "Num perturbThreads: " << numPerturbThreads << std::endl;

        // We need to calculate the absorbtion/scattering piece
        boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;

        long long perturbTimeCount = 0;
        long long weightCalcTimeCount = 0;

        // We need a number of dispatches
        boost::multiprecision::cpp_dec_float_100 minimumPathWeight = 0.0;
        boost::multiprecision::cpp_dec_float_100 maximumPathWeight = 0.0;

        const int64_t maxPathsPerDispatch = numPerturbThreads * NumPathsPerThread;
        const int64_t numDispatches = (m_experimentParams.numPathsInExperiment + maxPathsPerDispatch - 1) / maxPathsPerDispatch;

        int64_t numPathsLeft = m_experimentParams.numPathsInExperiment;
        int64_t numPathsGenerated = 0;

        for (int64_t dispatchIdx = 0; dispatchIdx < numDispatches; ++dispatchIdx)
        {
            int64_t pathsInDispatch = std::min(maxPathsPerDispatch, numPathsLeft);
            numPathsLeft -= pathsInDispatch;
            numPathsGenerated += pathsInDispatch;

            // We want to reinitialize starting paths here...
            {
                for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
                {
                    // Copy Pos
                    for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
                    {
                        perThreadCurvePositions[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
                    }

                    // Copy Tan
                    for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
                    {
                        perThreadCurveTangents[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
                    }

                    // Copy Curvatures
                    for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
                    {
                        perThreadCurveCurvatures[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
                    }
            }
        }

            auto perturbTimeStart = std::chrono::high_resolution_clock::now();


            std::vector<double> finalThreadWeights(numPerturbThreads);
            std::vector<double> finalThreadDifferences(numPerturbThreads);
        
            std::vector<double> runningPathWeightsLog10(numPathsGenerated);
            {
                std::vector<std::thread> threads(numPerturbThreads);

                for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
                {
                    std::thread newThread(&FullExperimentRunner2::LogWeightThreadFunction, this,
                        threadIdx,
                        dispatchIdx,
                        numPerturbThreads,
                        m_experimentParams.numPathsInExperiment,
                        NumPathsPerThread,
                        m_experimentParams.numPathsToSkip,
                        m_experimentParams.numSegmentsPerCurve,
                        std::ref(perThreadRngGenerators),
                        std::ref(perThreadCurvePositions),
                        std::ref(perThreadCurveTangents),
                        std::ref(perThreadCurveCurvatures),
                        std::ref(cachedSegmentWeights),
                        std::ref(finalThreadWeights),
                        std::ref(finalThreadDifferences),
                        std::ref(runningPathWeightsLog10),
                        m_upInitialCurve->m_segmentLength,
                        m_experimentParams.weightingParameters.scatter,
                        m_experimentParams.weightingParameters.absorbtion,
                        std::ref(lookupEvaluator.AccessLookupTable()),
                        minCurvature,
                        maxCurvature,
                        curvatureStepSize
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

            boost::multiprecision::cpp_dec_float_100 totalDispatchWeight = 0.0;
            {
                // We do the recombination of batches here
                // This is what we shift by for batch recombination stuff
                const double MaxBatchConstant = std::log10(numPerturbThreads);

                double maxBatchUnshiftedTotalLog10 = std::log10(finalThreadWeights[0]) - finalThreadDifferences[0];
                double actingDifference = MaxDoubleLog10 - (maxBatchUnshiftedTotalLog10 + MaxBatchConstant);
                double runningBatchTotal = std::pow(10.0, (maxBatchUnshiftedTotalLog10 + actingDifference));
                {
                    boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
                }

                for (int64_t batchIdx = 1; batchIdx < numPerturbThreads; ++batchIdx)
                {
                    double currentShiftedBatchWeightLog10 = std::log10(finalThreadWeights[batchIdx]);
                    double currentUnshiftedBatchWeightLog10 = currentShiftedBatchWeightLog10 - finalThreadDifferences[batchIdx];
                    // If this checks out, we have the same maximum and thus can just adjust things up
                    // TODO: Make this equal
                    if (maxBatchUnshiftedTotalLog10 >= currentUnshiftedBatchWeightLog10)
                    {
                        runningBatchTotal += std::pow(10, (currentUnshiftedBatchWeightLog10 + actingDifference));
                        {
                            boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
                        }

                        continue;
                    }


                    // If we are past, then we have a new maximum and need to adjust
                    // New difference

                    double adjustedLog10RunningTotal = std::log10(runningBatchTotal) - actingDifference;
                    actingDifference = MaxDoubleLog10 - (currentUnshiftedBatchWeightLog10 + MaxBatchConstant);
                    runningBatchTotal = std::pow(10.0, (adjustedLog10RunningTotal + actingDifference));

                    // Update
                    maxBatchUnshiftedTotalLog10 = currentUnshiftedBatchWeightLog10;

                    runningBatchTotal += std::pow(10, (maxBatchUnshiftedTotalLog10 + actingDifference));
                }
                
                // Convert combined batch total into big float and decompress
                boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
                totalDispatchWeight = boost::multiprecision::pow(10.0, (boost::multiprecision::log10(bigFloatRunningBatchTotal) - actingDifference));

                // Full path normalization term
                // Hardcoded value from Jerry analysis.
                boost::multiprecision::cpp_dec_float_100 singleSegmentNormalizer = 2.0 * TwistyPi * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(0.625));
                boost::multiprecision::cpp_dec_float_100 segmentNormalizer = 1.0;
                for (int64_t segIdx = 0; segIdx < (m_experimentParams.numSegmentsPerCurve - 1); ++segIdx)
                {
                    segmentNormalizer = segmentNormalizer * singleSegmentNormalizer;
                }

                boost::multiprecision::cpp_dec_float_100 pathNormalizer = 1.0;
                pathNormalizer = pathNormalizer * boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(static_cast<float>(m_experimentParams.numSegmentsPerCurve) / m_upInitialCurve->m_arclength), 3.0);
                pathNormalizer = pathNormalizer * segmentNormalizer;
                pathNormalizer = pathNormalizer * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(-0.325));
                
                totalDispatchWeight *= pathNormalizer;
            }

            bigTotalExperimentWeight += totalDispatchWeight;
            std::cout << "Weight: " << bigTotalExperimentWeight << std::endl;

            auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
            weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightingTimeEnd - weightingTimeStart).count();
            /* --------------------- */

#ifdef OutputBigFloatPathWeights
            std::filesystem::path currentPath = std::filesystem::current_path();
            currentPath.append(m_experimentParams.experimentDir);
            std::string bigfloatOutputFile = "BigFloatWeights.txt";
            currentPath.append(bigfloatOutputFile);

            std::ofstream bigfloatOFS;
            if (dispatchIdx == 0)
            {
                bigfloatOFS.open(currentPath.c_str());
                bigfloatOFS << m_experimentParams.numPathsInExperiment << std::endl;
            }
            else
            {
                bigfloatOFS.open(currentPath.c_str(), std::ios::app);
            }
            

            boost::multiprecision::cpp_dec_float_100 singleSegmentNormalizer = 2.0 * TwistyPi * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(0.625));
            boost::multiprecision::cpp_dec_float_100 segmentNormalizer = 1.0;
            for (int64_t segIdx = 0; segIdx < (m_experimentParams.numSegmentsPerCurve - 1); ++segIdx)
            {
                segmentNormalizer = segmentNormalizer * singleSegmentNormalizer;
            }

            boost::multiprecision::cpp_dec_float_100 pathNormalizer = 1.0;
            pathNormalizer = pathNormalizer * boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(static_cast<float>(m_experimentParams.numSegmentsPerCurve) / m_upInitialCurve->m_arclength), 3.0);
            pathNormalizer = pathNormalizer * segmentNormalizer;
            pathNormalizer = pathNormalizer * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(-0.325));


            boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10 = boost::multiprecision::log10(pathNormalizer);

            for (int64_t i = 0; i < pathsInDispatch; ++i)
            {
                boost::multiprecision::cpp_dec_float_100 weightLog10 = runningPathWeightsLog10[i];              
                bigfloatOFS << (weightLog10 + pathNormalizerLog10) << std::endl;
            }
#endif
        }

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

#if defined(ExportPathBatches)
        {
        EndPathBatchOutput();

        curvesBinaryFile.close();
        curvesMetadataFile.close();
        }
#endif

        ExperimentResults results;
        results.experimentWeight = bigTotalExperimentWeight;
        results.totalPathsGenerated = numPathsGenerated;
        results.numFailedPaths = 0;
        return results;
    }

    void FullExperimentRunner2::Shutdown()
    {
    }
}