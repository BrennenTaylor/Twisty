#include "FullExperimentRunnerOldMethodBridge.h"

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

const double AmountOfFullRotation = 1.0;

/*
    0 - Maps to Geometry
    1 - Maps to Spring
*/

enum class MethodType : int64_t
{
    Geometry = 0,
    Spring,
    Count
};

const int64_t Method_Geometry = 0;
const int64_t Method_Spring = 1;

const int64_t HybridRunCounts[static_cast<int64_t>(MethodType::Count)] =
{
    1000,
    1000
};
const MethodType StartingHybridMethod = MethodType::Geometry;

//#define EnforceSpringLengthDistance


const double MaxDoubleLog10 = 300;
const double PathBatchOffsetConstant = 6.0;

#if defined(EnforceSpringLengthDistance)
const double pathLengthThreshold = 0.01;
#endif
//#define HardcodedSegments
//#define HardcodedRotation
//#define HardcodednumPerturbThreadsFhybrid

// NOTE: Must be in single thread mode to work!!!
#define SINGLE_THREAD_PERTURB_MODE
//#define OutputCurvatures
//#define OutputBigFloatPathWeights
//#define SerialMultithread
//#define BlockingMultithread
//#define BlockingOutputThread

//#define ExportPathBatches

#ifdef BlockingOutputThread
std::mutex outputThreadMutex;
#endif


#ifdef BlockingMultithread
std::mutex blockingMultithreadMutex;
#endif

#ifdef SerialMultithread
//std::mutex serialMultithreadMutex;
//std::condition_variable serialMultithreadCV;
//int64_t activeThreadIdx = 0;

std::atomic<int64_t> activeThreadIdx = 0;

#endif

#if defined(ExportPathBatches)

const int64_t ExportPathBatchCacheSize = 30000;

std::mutex ExportPathBatchesMutex;
std::ofstream curvesBinaryFile;
std::ofstream curvesMetadataFile;

#endif

// static Farlor::Matrix3x3 RotationMatrixAroundAxis(float angle, Farlor::Vector3 axis)
// {
//     // Ensure its normalized
//     axis.Normalize();

//     Farlor::Matrix3x3 rotation(
//         Farlor::Vector3(
//             cos(angle) + axis.x * axis.x * (1.0f - cos(angle)),
//             axis.x * axis.y * (1.0f - cos(angle)) - axis.z * sin(angle),
//             axis.x * axis.z * (1.0f - cos(angle)) + axis.y * sin(angle)
//         ),
//         Farlor::Vector3(
//             axis.y * axis.x * (1.0f - cos(angle)) + axis.z * sin(angle),
//             cos(angle) + axis.y * axis.y * (1 - cos(angle)),
//             axis.y * axis.z * (1 - cos(angle)) - axis.x * sin(angle)
//         ),
//         Farlor::Vector3(
//             axis.z * axis.x * (1 - cos(angle)) - axis.y * sin(angle),
//             axis.z * axis.y * (1 - cos(angle)) + axis.x * sin(angle),
//             cos(angle) + axis.z * axis.z * (1 - cos(angle))
//         )
//     );
//     return rotation;
// }

namespace twisty
{
    FullExperimentRunnerOldMethodBridge::FullExperimentRunnerOldMethodBridge(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper)
        : ExperimentRunner(experimentParams, bootstrapper)
    {
    }

    FullExperimentRunnerOldMethodBridge::~FullExperimentRunnerOldMethodBridge()
    {
    }

    bool FullExperimentRunnerOldMethodBridge::Setup()
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

    void FullExperimentRunnerOldMethodBridge::LogWeightThreadFunction(
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
        float segmentLength,
        float scattering,
        float absorbtion,
        const std::vector<double>& lookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize,
        std::string pathToRawBinary,
        std::vector<double>& cachedWeights
    )
    {

#ifdef SerialMultithread
            while (activeThreadIdx.load() != threadIdx)
            {
            };
#endif

#ifdef BlockingOutputThread
            {
                std::scoped_lock<std::mutex> lock(outputThreadMutex);
                std::cout << "On thread: " << threadIdx << std::endl;
            }
#endif

#if defined(ExportPathBatches)

            // This should be per thread
            int64_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
            std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
            {
                for (int64_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
                {
                    for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                    {
                        int64_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
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

            double c = scattering + absorbtion;
            double absorbtionConst = std::exp(-c * segmentLength) / (2.0 * TwistyPi * TwistyPi);
            double absorbtionConstLog10 = std::log10(absorbtionConst);

            // We start by keeping a running path weight.
            // Lets make this a double actually...
            // Though this is a ln compressed version, so it shouuuuld be ok.

            double currentMaxWeightLog10 = 0.0;
            double currentMaxPossibleFinalWeightLog10 = 0.0;
            double currentDifference = 0.0;
            double runningTotal = 0.0;


            std::filesystem::path indexPath = std::filesystem::current_path();
            std::filesystem::path rawBinaryFullPath = indexPath.append(pathToRawBinary);
            std::ifstream rawBinaryFile(rawBinaryFullPath.c_str(), std::ios::binary);

            {
#ifdef BlockingMultithread
                std::scoped_lock<std::mutex> lock(blockingMultithreadMutex);
#endif
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

                    // Path count is in total
                    double runningSinglePathWeight = 0.0;
                    for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
                    {
                        // Expect to go negative, thus int
                        int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                        // We can exit once this point is reached as we have generated all the paths necessary for this thread
                        if (currentPathIdx >= numExperimentPaths)
                        {
#ifdef BlockingOutputThread
                            {
                                std::scoped_lock<std::mutex> lock(outputThreadMutex);
                                std::cout << "Returning, all paths complete" << std::endl;
                            }
#endif

#if defined(ExportPathBatches)
                            if (numCurvesInBatch > 0)
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


                            // We dont want to continue if we have already generated the correct number of paths.
                            break;
                        }

                        // Do the perturb now
                        {

                            const uint64_t bytesPerCurve = sizeof(Farlor::Vector3) * (numSegmentsPerCurve + 1);
                            uint64_t numIdxBytes = (uint64_t)(currentPathIdx) * (uint64_t)(bytesPerCurve);

                            rawBinaryFile.seekg(numIdxBytes, std::ios::beg);
                            rawBinaryFile.read((char*)(&(globalPos[CurrentThreadPosStartIdx])), bytesPerCurve);

                            // Read in from file and write to global pos

                            //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                            //We can do a different approach later.

                            // Tangents update
                            for (uint32_t pointIdx = 0; pointIdx < numSegmentsPerCurve; ++pointIdx)
                            {
                                Farlor::Vector3 leftPos = globalPos[CurrentThreadPosStartIdx + pointIdx];
                                Farlor::Vector3 rightPos = globalPos[CurrentThreadPosStartIdx + pointIdx + 1];

                                globalTans[CurrentThreadTanStartIdx + pointIdx] = (rightPos - leftPos).Normalized();
                            }

                            // Update curvatures
                            for (uint32_t segIdx = 0; segIdx < numSegmentsPerCurve; ++segIdx)
                            {
                                Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + segIdx];
                                Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + (segIdx + 1)];
                                Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);
                                float curvature = temp.Magnitude();
                                globalCurvatures[CurrentThreadCurvatureStartIdx + (segIdx)] = curvature;

                                
                            }

                            // Calculate weight
                            runningSinglePathWeight = 0.0;
                            for (uint32_t segIdx = 0; segIdx < numSegmentsPerCurve; ++segIdx)
                            {
                                double curvature = globalCurvatures[CurrentThreadCurvatureStartIdx + segIdx];

                                // Also, cache the weight of that changed segment
                                float distance = curvature - minCurvature;
                                float realIdx = distance / curvatureStepSize;
                                int64_t leftIdx = floor(realIdx);
                                int64_t rightIdx = leftIdx + 1;

                                double leftLookup = lookupTable[leftIdx];
                                double rightLookup = lookupTable[rightIdx];

                                float leftDist = distance - (leftIdx * curvatureStepSize);

                                double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                                double interpolatedResultLog10 = std::log10(interpolatedResult);

                                double segmentWeightLog10 = interpolatedResultLog10;

                                // Take natural log of this constant
                                segmentWeightLog10 += absorbtionConstLog10;

                                runningSinglePathWeight += segmentWeightLog10;
                            }

                        }

                        int64_t globalPathIdx = numPathsPerThread * numPerturbThreads * dispatchIdx + numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;

                        if (globalPathIdx >= numExperimentPaths)
                        {
                            // Skip
                        }
                        else if (pathCount < numPathsToSkipPerThread)
                        {
                            // Skip
                        }
                        else
                        {
                            // Else, contribute to the paths
                            int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                            assert(currentPathIdx >= numPathsPerThread * threadIdx);

                            int64_t pathNumber = pathCount - numPathsToSkipPerThread;

                            if (pathNumber == 0)
                            {
                                currentMaxWeightLog10 = runningSinglePathWeight;
                                currentMaxPossibleFinalWeightLog10 = runningSinglePathWeight + PathBatchOffsetConstant;
                                currentDifference = MaxDoubleLog10 - currentMaxPossibleFinalWeightLog10;
                                runningTotal += std::pow(10, runningSinglePathWeight + currentDifference);

                                // The log10 version of the weight
                                cachedWeights[currentPathIdx] = runningSinglePathWeight;
                            }
                            else
                            {
                                double newMaxWeightLog10 = runningSinglePathWeight;
                                // If this checks out, we have the same maximum and thus can just adjust things up
                                if (currentMaxWeightLog10 > newMaxWeightLog10)
                                {
                                    runningTotal += std::pow(10, runningSinglePathWeight + currentDifference);

                                    // The log10 version of the weight
                                    cachedWeights[currentPathIdx] = runningSinglePathWeight;
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

                                    runningTotal += std::pow(10, runningSinglePathWeight + currentDifference);

                                    // The log10 version of the weight
                                    cachedWeights[currentPathIdx] = runningSinglePathWeight;
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

            rawBinaryFile.close();
#ifdef SerialMultithread
        activeThreadIdx--;
#endif
    }

    ExperimentRunner::ExperimentResults FullExperimentRunnerOldMethodBridge::RunExperiment()
    {
        return RunExperimentLogWeightTable();
    }

    ExperimentRunner::ExperimentResults FullExperimentRunnerOldMethodBridge::RunExperimentLogWeightTable()
    {
        int64_t numFailures = 0;
        int64_t totalFailures = 0;
        int64_t totalSuccess = 0;

        std::string pathToRawBinary = "";
        std::cout << "Input: path to raw binary" << std::endl;
        std::cin >> pathToRawBinary;
        std::cout << "Path to Raw Binary: " << pathToRawBinary << std::endl;

        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();

        /* --------------------- */
        auto setupTimeStart = std::chrono::high_resolution_clock::now();

#if defined(ExportPathBatches)
        {
            BeginPathBatchOutput();

            std::filesystem::path experimentDirPath = m_experimentParams.experimentDirPath;
            if (!std::filesystem::exists(experimentDirPath))
            {
                std::filesystem::create_directories(experimentDirPath);
            }

            std::stringstream pathBinaryFilenameSS;
            pathBinaryFilenameSS << m_experimentParams.pathBatchPrepend;
            pathBinaryFilenameSS << "Paths_Binary" << ".pbd";

            std::filesystem::path binaryFilePath = experimentDirPath;
            binaryFilePath.append(pathBinaryFilenameSS.str());
            curvesBinaryFile.open(binaryFilePath, std::ios::binary);

            std::stringstream pathMetadataFilenameSS;
            pathMetadataFilenameSS << m_experimentParams.pathBatchPrepend;
            pathMetadataFilenameSS << "Paths_Metadata" << ".pmd";

            std::filesystem::path metadataFilePath = experimentDirPath;
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
//#ifdef SINGLE_THREAD_PERTURB_MODE
        // NOTE: Only works in single thread mode
        int64_t numPerturbThreads = 1;
//#else
//#ifdef HardcodednumPerturbThreads
//        int64_t numPerturbThreads = 3;
//#else
//        int64_t numPerturbThreads = std::thread::hardware_concurrency();
//#endif
//#endif
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

#if defined(DetailedPurturb)
        {
            std::cout << "Positions" << std::endl;
            for (int64_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurvePositions[segmentIdx] << std::endl;
            }
        }
#endif

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

#if defined(DetailedPurturb)
        {
            std::cout << "Tangents" << std::endl;
            for (int64_t tanIdx = 0; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
            {
                std::cout << "\t" << initialCurveTangents[tanIdx] << std::endl;
            }
        }
#endif

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

#if defined(DetailedPurturb)
        {
            std::cout << "Curvatures" << std::endl;
            for (int64_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurveCurvatures[segmentIdx] << std::endl;
            }
        }
#endif

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

#ifdef SerialMultithread
        activeThreadIdx.store(numPerturbThreads - 1);
#endif


        //const int64_t numDispatches = (m_experimentParams.numPathsInExperiment + m_experimentParams.numPathsPerBatch - 1) / m_experimentParams.numPathsPerBatch;
        std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
        std::cout << "numPathsPerBatch: " << m_experimentParams.numPathsPerBatch << std::endl;
        std::cout << "Num perturbThreads: " << numPerturbThreads << std::endl;
        //std::cout << "Num dispatches required: " << numDispatches << std::endl;

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

        std::vector<double> cachedWeights(maxPathsPerDispatch);

        for (int64_t dispatchIdx = 0; dispatchIdx < numDispatches; ++dispatchIdx)
        {

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

            numPathsGenerated += NumPathsPerThread * numPerturbThreads;

            auto perturbTimeStart = std::chrono::high_resolution_clock::now();

            //std::cout << "Paths in dispatch " << dispatchIdx << ": " << maxPathsPerDispatch << std::endl;


            std::vector<double> finalThreadWeights(numPerturbThreads);
            std::vector<double> finalThreadDifferences(numPerturbThreads);
            {
                std::vector<std::thread> threads(numPerturbThreads);

                for (int64_t threadIdx = 0; threadIdx < numPerturbThreads; ++threadIdx)
                {
                    std::thread newThread(&FullExperimentRunnerOldMethodBridge::LogWeightThreadFunction, this,
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
                        m_upInitialCurve->m_segmentLength,
                        m_experimentParams.weightingParameters.scatter,
                        m_experimentParams.weightingParameters.absorbtion,
                        std::ref(lookupEvaluator.AccessLookupTable()),
                        minCurvature,
                        maxCurvature,
                        curvatureStepSize,
                        pathToRawBinary,
                        std::ref(cachedWeights)
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
                //std::cout << "\n\nBatch recombination section: " << std::endl;

                //for (int64_t i = 0; i < numPerturbThreads; ++i)
                //{
                //    std::cout << "Weight: " << finalThreadWeights[i] << " - Difference: " << finalThreadDifferences[i] << std::endl;
                //}


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

                    //{
                    //    // Convert combined batch total into big float and decompress
                    //    boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
                    //    boost::multiprecision::cpp_dec_float_100 temp = boost::multiprecision::pow(10.0, (boost::multiprecision::log10(bigFloatRunningBatchTotal) - actingDifference));

                    //    // Full path normalization term
                    //    // Hardcoded value from Jerry analysis.
                    //    boost::multiprecision::cpp_dec_float_100 singleSegmentNormalizer = 2.0 * TwistyPi * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(0.625));
                    //    boost::multiprecision::cpp_dec_float_100 segmentNormalizer = 1.0;
                    //    for (int64_t segIdx = 0; segIdx < (m_experimentParams.numSegmentsPerCurve - 1); ++segIdx)
                    //    {
                    //        segmentNormalizer = segmentNormalizer * singleSegmentNormalizer;
                    //    }

                    //    boost::multiprecision::cpp_dec_float_100 pathNormalizer = 1.0;
                    //    pathNormalizer = pathNormalizer * boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(static_cast<float>(m_experimentParams.numSegmentsPerCurve) / m_upInitialCurve->m_arclength), 3.0);
                    //    pathNormalizer = pathNormalizer * segmentNormalizer;
                    //    pathNormalizer = pathNormalizer * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(-0.325));

                    //    temp *= pathNormalizer;
                    //    std::cout << "Weight: " << temp << std::endl;
                    //}
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


#if defined(DetailedPurturb)
                {
                    std::cout << "Thread weights: " << std::endl;
                    for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                    {
                        std::cout << threadWeights[threadIdx] << std::endl;
                    }
                }
#endif
            }

            bigTotalExperimentWeight += totalDispatchWeight;
            std::cout << "Weight: " << bigTotalExperimentWeight << std::endl;

            auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
            weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightingTimeEnd - weightingTimeStart).count();
            /* --------------------- */

#ifdef OutputBigFloatPathWeights

            uint32_t numWeights = std::min(numPathsLeft, numPathsGenerated);
            std::cout << "Writing weights: " << numWeights << std::endl;

            std::filesystem::path experimentDirectoryPath = m_experimentParams.experimentDirPath;
            if (!std::filesystem::exists(experimentDirectoryPath))
            {
                std::filesystem::create_directories(experimentDirectoryPath);
            }

            std::string log10WeightOutputFile = "PathWeightsLog10.txt";
            experimentDirectoryPath.append(log10WeightOutputFile);

            std::ofstream weightOFS;
            if (dispatchIdx == 0)
            {
                weightOFS.open(experimentDirectoryPath.c_str());
                weightOFS << m_experimentParams.numPathsInExperiment << std::endl;
            }
            else
            {
                weightOFS.open(experimentDirectoryPath.c_str(), std::ios::app);
            }
            
            for (int64_t i = 0; i < numWeights; ++i)
            {
                weightOFS << cachedWeights[i] << std::endl;
            }
#endif

            numPathsLeft -= numPathsGenerated;
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

    void FullExperimentRunnerOldMethodBridge::Shutdown()
    {
    }
}