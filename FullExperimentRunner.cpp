#include "FullExperimentRunner.h"

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

enum class MethodType : uint32_t
{
    Geometry = 0,
    Spring,
    Count
};

const uint32_t Method_Geometry = 0;
const uint32_t Method_Spring = 1;

const uint32_t HybridRunCounts[static_cast<uint32_t>(MethodType::Count)] =
{
    1000,
    1000
};
const MethodType StartingHybridMethod = MethodType::Geometry;

//#define EnforceSpringLengthDistance

#if defined(EnforceSpringLengthDistance)
const double pathLengthThreshold = 0.01;
#endif
//#define HardcodedSegments
//#define HardcodedRotation
//#define HardcodedNumPurturbThreadsFhybrid


//#define DelayedAbsorbtionContribution
//#define SINGLE_THREAD_PERTURB_MODE
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
//uint32_t activeThreadIdx = 0;

std::atomic<uint32_t> activeThreadIdx = 0;

#endif

#if defined(ExportPathBatches)

const uint32_t ExportPathBatchCacheSize = 30000;

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
    FullExperimentRunner::FullExperimentRunner(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper, Range kdsRange, Range tdsRange)
        : ExperimentRunner(experimentParams, bootstrapper)
    {
    }

    FullExperimentRunner::~FullExperimentRunner()
    {
    }

    bool FullExperimentRunner::Setup()
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
        return true;
    }

    void FullExperimentRunner::LogWeightThreadFunction(
        uint32_t threadIdx,
        int32_t numExperimentPaths,
        int32_t numPathsPerThread,
        int32_t numPathsToSkipPerThread,
        int32_t numSegmentsPerCurve,
        std::vector<std::mt19937>& rngGenerators,
        std::vector<Farlor::Vector3>& globalPos,
        std::vector<Farlor::Vector3>& globalTans,
        std::vector<float>& globalCurvatures,
        std::vector<double>& globalPathWeights,
        std::vector<double>& cachedSegmentWeights,
        float segmentLength,
        float scattering,
        float absorbtion,
        const std::vector<double>& lookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize
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
            uint32_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
            std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
            {
                for (uint32_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
                {
                    for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                    {
                        uint32_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
                        pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                    }
                }
            }
#endif

            // First, we discard random numbers to match the previous amount
            rngGenerators[threadIdx].discard(numPathsPerThread * threadIdx);
            
            const uint32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
            const uint32_t NumTanPerCurve = (numSegmentsPerCurve + 1);
            const uint32_t NumCurvaturesPerCurve = numSegmentsPerCurve;

            const uint32_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
            const uint32_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
            const uint32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

            float c = scattering + absorbtion;
            float absorbtionConst = std::exp(-c * segmentLength) / (2.0 * TwistyPi * TwistyPi);

            float lnAbsorbtionConst = log(absorbtionConst);

            // We start by keeping a running path weight.
            // Lets make this a double actually...
            // Though this is a ln compressed version, so it shouuuuld be ok.

            double runningPathWeight = 0.0;
            {
#ifdef BlockingMultithread
                std::scoped_lock<std::mutex> lock(blockingMultithreadMutex);
#endif
                // Lets precache all the segment weights
                {
                    for (int32_t segIdx = 0; segIdx < numSegmentsPerCurve; ++segIdx)
                    {
                        // Extract curvature
                        float curvature = globalCurvatures[CurrentThreadCurvatureStartIdx + segIdx];

                        float distance = curvature - minCurvature;
                        float realIdx = distance / curvatureStepSize;
                        int32_t leftIdx = floor(realIdx);
                        int32_t rightIdx = leftIdx + 1;

                        double leftLookup = lookupTable[leftIdx];
                        double rightLookup = lookupTable[rightIdx];

                        float leftDist = distance - (leftIdx * curvatureStepSize);

                        double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                        // Take the natural log of the interpolated results
                        double interpolatedResultLog = log(interpolatedResult);
                        // Lets do weights as doubles for now
                        double segmentWeight = interpolatedResultLog;

#ifdef DelayedAbsorbtionContribution
                        // Do nothing, we dont add it in here
#else
                        // Take natural log of this constant
                        segmentWeight += lnAbsorbtionConst;
#endif

                        // Update the running path weight. We also want to cache the segment weights
                        runningPathWeight += segmentWeight;
                        cachedSegmentWeights[segIdx + (numSegmentsPerCurve * threadIdx)] = segmentWeight;
                    }
                }

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                {
                    printf("Cached Weights:\n");

                    for (uint32_t segIdx = 0; segIdx < numSegmentsPerCurve; segIdx++)
                    {
                        printf("\tCached Weight: <%0.6f>\n", cachedSegmentWeights[segIdx + (numSegmentsPerCurve * threadIdx)]);
                    }
                }
#endif
            }


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

                    uint32_t numCurvesInBatch = 0;
                    uint32_t outputIdx = 0;

                    int32_t cacheStartPathIdx = numPathsPerThread * threadIdx;
                    for (int32_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
                    {
                        //std::cout << "Current path idx: " << perThreadPathCount << std::endl;
                        // Expect to go negative, thus int
                        int32_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
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

                                if (threadIdx == 11)
                                {
                                    std::cout << "Should be exporting thread 12" << std::endl;
                                }


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
#ifdef HardcodedSegments
                            int32_t leftPointIndex = 17;
                            int32_t rightPointIndex = 39;
#else
                            std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - 3); // uniform, unbiased
                            int32_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);
                            std::uniform_int_distribution<int> rightPointIndexUniformDist(leftPointIndex + 2, numSegmentsPerCurve - 1); // uniform, unbiased
                            int32_t rightPointIndex = rightPointIndexUniformDist(rngGenerators[threadIdx]);
#endif

                            assert(leftPointIndex < rightPointIndex);
                            assert((rightPointIndex - leftPointIndex) >= 2);

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                            {
                                printf("Left point idx: %d\n", leftPointIndex);
                                printf("Right point idx: %d\n", rightPointIndex);
                            }
#endif

                            assert(leftPointIndex < rightPointIndex);
                            assert((rightPointIndex - leftPointIndex) >= 2);

                            // We need two frames for each segment to get the new curvature and torsion.
                            // we need the frame left of the segment, as well as the frame right of the segment.

                            // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                            Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftPointIndex];
                            Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightPointIndex];

                            Farlor::Vector3 axisOfRotation = (rightPoint - leftPoint).Normalized();

#ifdef HardcodedRotation
                            float randomAngle = 1.38f;
#else
                            std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * AmountOfFullRotation, TwistyPi * AmountOfFullRotation);
                            float randomAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
#endif               
                            //float randomAngle = 0.0f;

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                            {
                                printf("randomAngle: %.6f\n", randomAngle);
                            }
#endif

                            float rotationMatrix[9];
                            RotationMatrixAroundAxis(randomAngle, (float*)(&axisOfRotation), rotationMatrix);
#if defined(DetailedPurturb) && defined(SingleThreadMode)
                            {
                                printf("Rotation Matrix\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n",
                                    rotationMatrix[0], rotationMatrix[1], rotationMatrix[2],
                                    rotationMatrix[3], rotationMatrix[4], rotationMatrix[5],
                                    rotationMatrix[6], rotationMatrix[7], rotationMatrix[8]
                                );
                            }
#endif

                            int32_t numChanged = 0;
                            for (int32_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                            {
                                numChanged++;

                                Farlor::Vector3 shiftedPoint = globalPos[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                                // Rotate and stuff back in shifted point
                                RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));

                                // Update the point with the rotated version
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
                                Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + (leftPointIndex - 1)];
                                Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + (leftPointIndex)];
                                Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);
                                float curvature = temp.Magnitude();
                                globalCurvatures[CurrentThreadCurvatureStartIdx + (leftPointIndex - 1)] = curvature;

                                // Also, cache the weight of that changed segment
                                float distance = curvature - minCurvature;
                                float realIdx = distance / curvatureStepSize;
                                int32_t leftIdx = floor(realIdx);
                                int32_t rightIdx = leftIdx + 1;

                                double leftLookup = lookupTable[leftIdx];
                                double rightLookup = lookupTable[rightIdx];

                                float leftDist = distance - (leftIdx * curvatureStepSize);

                                double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                                //std::cout << "Interpolated result: " << interpolatedResult << std::endl;
                                double interpolatedResultLog = log(interpolatedResult);

                                double segmentWeight = interpolatedResultLog;

#ifdef DelayedAbsorbtionContribution
                                // Do nothing, we dont add it in here
#else
                                // Take natural log of this constant
                                segmentWeight += lnAbsorbtionConst;
#endif

                                // Remove old segmentWeight
                                //runningPathWeight -= pCachedSegmentWeights[leftPointIndex - 1];
                                runningPathWeight -= cachedSegmentWeights[(leftPointIndex - 1) + (numSegmentsPerCurve * threadIdx)];
                                // Cache new weight
                                cachedSegmentWeights[(leftPointIndex - 1) + (numSegmentsPerCurve * threadIdx)] = segmentWeight;
                                // Add segment weighting into running path weight
                                runningPathWeight += segmentWeight;
                            }

                            // Update right curvature
                            {
                                Farlor::Vector3 leftTan = globalPos[CurrentThreadTanStartIdx + (rightPointIndex - 1)];
                                Farlor::Vector3 rightTan = globalPos[CurrentThreadTanStartIdx + rightPointIndex];

                                Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);
                                float curvature = temp.Magnitude();
                                globalCurvatures[CurrentThreadCurvatureStartIdx + (rightPointIndex - 1)] = curvature;

                                // Also, cache the weight of that changed segment
                                float distance = curvature - minCurvature;
                                float realIdx = distance / curvatureStepSize;
                                int32_t leftIdx = floor(realIdx);
                                int32_t rightIdx = leftIdx + 1;

                                double leftLookup = lookupTable[leftIdx];
                                double rightLookup = lookupTable[rightIdx];

                                float leftDist = distance - (leftIdx * curvatureStepSize);

                                // TODO: Whats the point even of doing logs if this is necessary
                                double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                                //std::cout << "Interpolated result: " << interpolatedResult << std::endl;
                                double interpolatedResultLog = log(interpolatedResult);

                                double segmentWeight = interpolatedResultLog;
#ifdef DelayedAbsorbtionContribution
                                // Do nothing, we dont add it in here
#else
                                // Take natural log of this constant
                                segmentWeight += lnAbsorbtionConst;
#endif

                                // Remove old segmentWeight
                                runningPathWeight -= cachedSegmentWeights[(rightPointIndex - 1) + (numSegmentsPerCurve * threadIdx)];
                                runningPathWeight += segmentWeight;
                                cachedSegmentWeights[(rightPointIndex - 1) + (numSegmentsPerCurve * threadIdx)] = segmentWeight;
                            }
                        }

                        if (pathCount < numPathsToSkipPerThread)
                        {
                            // Skip
                        }
                        else
                        {
                            // Else, contribute to the paths
                            int32_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                            assert(currentPathIdx >= numPathsPerThread * threadIdx);
                            //std::cout << "----- Path: " << currentPathIdx << std::endl;
                            globalPathWeights[currentPathIdx] = runningPathWeight;

#if defined(ExportPathBatches)
                            // Add the path to the path batch
                            for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
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
#ifdef SerialMultithread
        activeThreadIdx--;
#endif
    }

    // U and V are uniform random between 0 and 1
    Farlor::Vector3 UniformDirection(const float u, const float v)
    {
        float theta = u * TwistyPi * 2;
        float z = (v * 2.0f) - 1.0f;

        return Farlor::Vector3(sqrt(1.0f - z*z)*std::cos(theta), sqrt(1.0f - z * z) * std::sin(theta), z).Normalized();
    }


    Farlor::Vector3 SpringForceAonB(Farlor::Vector3 a, Farlor::Vector3 b, float springConst, float desiredLength)
    {
        Farlor::Vector3 springVector = b - a;
        float mag = springVector.Magnitude();

        if (mag == 0)
        {
            return Farlor::Vector3(0.0f, 0.0f, 0.0f);
        }

        Farlor::Vector3 force = -1.0f * (springVector) * (1.0f / mag) * (mag - desiredLength) * springConst;
        return force;
    }

    void FullExperimentRunner::SpringBasedPerturb(
        uint32_t threadIdx,
        int32_t numExperimentPaths,
        int32_t numPathsPerThread,
        int32_t numPathsToSkipPerThread,
        int32_t numSegmentsPerCurve,
        std::vector<std::mt19937>& rngGenerators,
        std::vector<Farlor::Vector3>& globalPos,
        std::vector<Farlor::Vector3>& globalTans,
        std::vector<float>& globalCurvatures,
        std::vector<double>& globalPathWeights,
        std::vector<double>& cachedSegmentWeights,
        float segmentLength,
        float scattering,
        float absorbtion,
        const std::vector<double>& lookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize
    )
    {
        if (threadIdx == 11)
        {
            std::cout << "Thread 12 hit" << std::endl;
        }

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
        uint32_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
        std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
        {
            for (uint32_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
            {
                for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                {
                    uint32_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
                    pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                }
            }
        }
#endif

        // First, we discard random numbers to match the previous 
        rngGenerators[threadIdx].discard(numPathsPerThread * threadIdx);

        const uint32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
        const uint32_t NumTanPerCurve = (numSegmentsPerCurve + 1);
        const uint32_t NumCurvaturesPerCurve = numSegmentsPerCurve;

        const uint32_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
        const uint32_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
        const uint32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

        // We store forces for all points.
        // Movable points have forces updated.
        // This excluedes the first two points, as the first segment is locked
        // This also excludes the final point. They still have springs attached, however they remain at zero
        std::vector<Farlor::Vector3> netForcePerPoint(numSegmentsPerCurve + 1);
        for (auto& force : netForcePerPoint)
        {
            // Zero out the force
            force = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        }
        
        std::uniform_real_distribution<float> zeroToOneUniformDist(0.0f, 1.0f);

        std::vector<Farlor::Vector3> oldPoints(numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> prevPoints(numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> newPoints(numSegmentsPerCurve + 1);

        for (uint32_t ptIdx = 0; ptIdx < netForcePerPoint.size(); ptIdx++)
        {
            // The left point is the first point of the first movable segment
            Farlor::Vector3 point = globalPos[CurrentThreadPosStartIdx + ptIdx];

            oldPoints[ptIdx] = point;
            prevPoints[ptIdx] = point;
            newPoints[ptIdx] = point;
        }

        double initialPathArclength = 0.0;
        for (uint32_t ptIdx = 0; ptIdx < numSegmentsPerCurve; ptIdx++)
        {
            auto& leftPt = newPoints[ptIdx];
            auto& rightPt = newPoints[ptIdx + 1];
            initialPathArclength += (rightPt - leftPt).Magnitude();
        }

        // Assume we have a mass of 1
        float pointMass = 0.1f;
        float segmentStiffness = 10000.0f;
        float jointStiffness = 1000.0f;
        float desiredSegmentLengthSpring = segmentLength * 1.0f;
        float desiredJointLengthSpring = segmentLength * 2.0f;
        uint32_t gravityRate = 10000;

        Farlor::Vector3 gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]),
            zeroToOneUniformDist(rngGenerators[threadIdx]));

        // Update "15 times a second"
        float timestep = 1.0f / 1000;

        // Caclulate the first global path index this thread will start on
        int32_t threadStartingPathIdx = numPathsPerThread * threadIdx;
        float c = scattering + absorbtion;
        float absorbtionConst = std::exp(-c * segmentLength) / (2.0 * TwistyPi * TwistyPi);
        float lnAbsorbtionConst = log(absorbtionConst);

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

                uint32_t numCurvesInBatch = 0;
                uint32_t outputIdx = 0;

                int32_t cacheStartPathIdx = numPathsPerThread * threadIdx;
                for (int32_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
                {
                    // Expect to go negative, thus int
                    int32_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;

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

                            if (threadIdx == 11)
                            {
                                std::cout << "Should be exporting thread 12" << std::endl;
                            }


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
                    double pathWeight = 0.0;
                    {
                        {
                            // Reset the force vector
                            // All points
                            for (auto& force : netForcePerPoint)
                            {
                                // Zero out the force
                                force = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                            }

                            // Segment Springs
                            {
                                for (uint32_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 1); ptIdx++)
                                {
                                    uint32_t leftIdx = ptIdx;
                                    uint32_t rightIdx = ptIdx + 1;
                                    // The left point is the first point of the first movable segment
                                    Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
                                    // The right point is one to the right of that
                                    Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];

                                    Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, segmentStiffness, desiredSegmentLengthSpring);
                                    // Only apply to left in this case
                                    netForcePerPoint[leftIdx] -= forceAonB;
                                    netForcePerPoint[rightIdx] += forceAonB;
                                }
                            }

                            // Joint Springs
                            {
                                for (uint32_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 2); ptIdx++)
                                {
                                    uint32_t leftIdx = ptIdx;
                                    uint32_t rightIdx = ptIdx + 2;
                                    // The left point is the first point of the first movable segment
                                    Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
                                    // The right point is one to the right of that
                                    Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];

                                    Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, jointStiffness, desiredJointLengthSpring);
                                    // Only apply to left in this case
                                    netForcePerPoint[leftIdx] -= forceAonB;
                                    netForcePerPoint[rightIdx] += forceAonB;
                                }
                            }


                            // Add in gravity cause why not
                            for (uint32_t ptIdx = 0; ptIdx < netForcePerPoint.size(); ptIdx++)
                            {
                                netForcePerPoint[ptIdx] += gravityForce * pointMass;
                            }

                            // Force the three set points to have no force
                            {
                                netForcePerPoint[0] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                                netForcePerPoint[1] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                                netForcePerPoint[numSegmentsPerCurve] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                            }

                            // Dont update point 0, 1 and M
                            for (int32_t pointIdx = 2; pointIdx < numSegmentsPerCurve; ++pointIdx)
                            {
                                Farlor::Vector3 acc = netForcePerPoint[pointIdx] * (1.0f / pointMass);
                                newPoints[pointIdx] = 2.0f * prevPoints[pointIdx] - oldPoints[pointIdx] + acc * timestep * timestep;
                            }

                            // Assert points are at the start and end correctly
                            if (newPoints[0] != m_upInitialCurve->m_basePos)
                            {
                                std::cout << "Path perturb failed as start pos moved" << std::endl;
                                std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
                            }

                            if (newPoints[numSegmentsPerCurve] != m_upInitialCurve->m_targetPos)
                            {
                                std::cout << "Path perturb failed as end pos moved" << std::endl;
                                std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
                            }

                            assert(newPoints[0] == m_upInitialCurve->m_basePos);
                            assert(newPoints[numSegmentsPerCurve] == m_upInitialCurve->m_targetPos);
                        }

                        // Update points from current buffer
                        for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                        {
                            globalPos[CurrentThreadPosStartIdx + pointIdx] = newPoints[pointIdx];
                        }

                        // Store the older versions of the points
                        oldPoints = prevPoints;
                        prevPoints = newPoints;

                        // Update all tangents

                        for (int32_t tanIdx = 0; tanIdx < numSegmentsPerCurve; ++tanIdx)
                        {
                            Farlor::Vector3 leftPos = globalPos[CurrentThreadPosStartIdx + tanIdx];
                            Farlor::Vector3 rightPos = globalPos[CurrentThreadPosStartIdx + (tanIdx + 1)];

                            globalTans[CurrentThreadTanStartIdx + tanIdx] = (rightPos - leftPos).Normalized();
                        }

                        // Update curvature values
                        for (int32_t curvatureIdx = 0; curvatureIdx < numSegmentsPerCurve; ++curvatureIdx)
                        {
                            Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + curvatureIdx];
                            Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + (curvatureIdx + 1)];

                            Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);

                            const float curvature = temp.Magnitude();
                            globalCurvatures[CurrentThreadCurvatureStartIdx + curvatureIdx] = curvature;

                            // Also, cache the weight of that changed segment
                            float distance = curvature - minCurvature;
                            float realIdx = distance / curvatureStepSize;
                            int32_t leftIdx = floor(realIdx);
                            int32_t rightIdx = leftIdx + 1;

                            double leftLookup = lookupTable[leftIdx];
                            double rightLookup = lookupTable[rightIdx];

                            float leftDist = distance - (leftIdx * curvatureStepSize);

                            double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                            //std::cout << "Interpolated result: " << interpolatedResult << std::endl;
                            double interpolatedResultLog = log(interpolatedResult);

                            double segmentWeight = interpolatedResultLog;

#ifdef DelayedAbsorbtionContribution
                            // Do nothing, we dont add it in here
#else
                            // Take natural log of this constant
                            segmentWeight += lnAbsorbtionConst;
#endif
                            // Add segment weighting into running path weight
                            pathWeight += segmentWeight;
                        }
                    }

                    if (pathCount < numPathsToSkipPerThread)
                    {
                        // Skip
                    }
                    else
                    {
                        // Else, contribute to the paths
                        int32_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                        assert(currentPathIdx >= numPathsPerThread * threadIdx);

                        // Select new gravity
                        if (currentPathIdx % gravityRate == 0)
                        {
                            gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]), zeroToOneUniformDist(rngGenerators[threadIdx]));
                        }

                        //std::cout << "----- Path: " << currentPathIdx << std::endl;
                        globalPathWeights[currentPathIdx] = pathWeight;


#if defined(ExportPathBatches)
                        // Add the path to the path batch
                        for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
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
#ifdef SerialMultithread
        activeThreadIdx--;
#endif
    }




    void FullExperimentRunner::HybridMethod(
        uint32_t threadIdx,
        int32_t numExperimentPaths,
        int32_t numPathsPerThread,
        int32_t numPathsToSkipPerThread,
        int32_t numSegmentsPerCurve,
        std::vector<std::mt19937>& rngGenerators,
        std::vector<Farlor::Vector3>& globalPos,
        std::vector<Farlor::Vector3>& globalTans,
        std::vector<float>& globalCurvatures,
        std::vector<double>& globalPathWeights,
        std::vector<double>& cachedSegmentWeights,
        float segmentLength,
        float scattering,
        float absorbtion,
        const std::vector<double>& lookupTable,
        float minCurvature,
        float maxCurvature,
        float curvatureStepSize
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
        uint32_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
        std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
        {
            for (uint32_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
            {
                for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                {
                    uint32_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
                    pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                }
            }
        }
#endif


        // Both method variables
        MethodType currentMethod = StartingHybridMethod;
        uint32_t currentMethodCount = 0;

        const uint32_t NumPosPerCurve = (numSegmentsPerCurve + 1);
        const uint32_t NumTanPerCurve = (numSegmentsPerCurve + 1);
        const uint32_t NumCurvaturesPerCurve = numSegmentsPerCurve;

        const uint32_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
        const uint32_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
        const uint32_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

        // First, we discard random numbers to match the previous 
        rngGenerators[threadIdx].discard(numPathsPerThread * threadIdx);
        
        std::uniform_real_distribution<float> zeroToOneUniformDist(0.0f, 1.0f);
        std::vector<Farlor::Vector3> newPoints(numSegmentsPerCurve + 1);
        for (uint32_t ptIdx = 0; ptIdx < (numSegmentsPerCurve + 1); ptIdx++)
        {
            // The left point is the first point of the first movable segment
            Farlor::Vector3 point = globalPos[CurrentThreadPosStartIdx + ptIdx];
            newPoints[ptIdx] = point;
        }


        // Geometry method specific stuff


        // Spring force specific stuff
        std::vector<Farlor::Vector3> oldPoints(numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> prevPoints(numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> netForcePerPoint(numSegmentsPerCurve + 1);

        // Assume we have a mass of 1
        const float pointMass = 0.1f;
        const float segmentStiffness = 10000.0f;
        const float jointStiffness = 1000.0f;
        const float desiredSegmentLengthSpring = segmentLength * 1.0f;
        const float desiredJointLengthSpring = segmentLength * 2.0f;
        const uint32_t gravityRate = 10000;

        Farlor::Vector3 gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]),
            zeroToOneUniformDist(rngGenerators[threadIdx]));

        // Update "15 times a second"
        float timestep = 1.0f / 1000;

        // End Spring Specific Initialization

        //// Initialization code
        //bool done = false;
        //while (!done)
        //{
        //    // If using the geometry method
        //    if (currentMethod == 0)
        //    {

        //    }
        //    // Spring method
        //    else if (currentMethod == 1)
        //    {

        //    }
        //    else
        //    {
        //        assert(false);
        //        // Method not supported
        //    }
        //}



        // Caclulate the first global path index this thread will start on
        int32_t threadStartingPathIdx = numPathsPerThread * threadIdx;
        float c = scattering + absorbtion;
        float absorbtionConst = std::exp(-c * segmentLength) / (2.0 * TwistyPi * TwistyPi);
        float lnAbsorbtionConst = log(absorbtionConst);

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

                uint32_t numCurvesInBatch = 0;
                uint32_t outputIdx = 0;

                bool justSwitchedPerturbMethod = true;

                int32_t cacheStartPathIdx = numPathsPerThread * threadIdx;
                for (int32_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
                {
                    // Expect to go negative, thus int
                    int32_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;

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

                            if (threadIdx == 11)
                            {
                                std::cout << "Should be exporting thread 12" << std::endl;
                            }


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
                    double pathWeight = 0.0;
                    {
                        if (currentMethod == MethodType::Geometry)
                        {



                            // Do the perturb now
                            {
#ifdef HardcodedSegments
                                int32_t leftPointIndex = 17;
                                int32_t rightPointIndex = 39;
#else
                                std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - 3); // uniform, unbiased
                                int32_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);
                                std::uniform_int_distribution<int> rightPointIndexUniformDist(leftPointIndex + 2, numSegmentsPerCurve - 1); // uniform, unbiased
                                int32_t rightPointIndex = rightPointIndexUniformDist(rngGenerators[threadIdx]);
#endif

                                assert(leftPointIndex < rightPointIndex);
                                assert((rightPointIndex - leftPointIndex) >= 2);

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                                {
                                    printf("Left point idx: %d\n", leftPointIndex);
                                    printf("Right point idx: %d\n", rightPointIndex);
                                }
#endif

                                assert(leftPointIndex < rightPointIndex);
                                assert((rightPointIndex - leftPointIndex) >= 2);

                                // We need two frames for each segment to get the new curvature and torsion.
                                // we need the frame left of the segment, as well as the frame right of the segment.

                                // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                                Farlor::Vector3 leftPoint = newPoints[leftPointIndex];
                                Farlor::Vector3 rightPoint = newPoints[rightPointIndex];

                                Farlor::Vector3 axisOfRotation = (rightPoint - leftPoint).Normalized();

#ifdef HardcodedRotation
                                float randomAngle = 1.38f;
#else
                                std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * AmountOfFullRotation, TwistyPi * AmountOfFullRotation);
                                float randomAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
#endif               

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                                {
                                    printf("randomAngle: %.6f\n", randomAngle);
                                }
#endif

                                float rotationMatrix[9];
                                RotationMatrixAroundAxis(randomAngle, (float*)(&axisOfRotation), rotationMatrix);

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                                {
                                    printf("Rotation Matrix\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n",
                                        rotationMatrix[0], rotationMatrix[1], rotationMatrix[2],
                                        rotationMatrix[3], rotationMatrix[4], rotationMatrix[5],
                                        rotationMatrix[6], rotationMatrix[7], rotationMatrix[8]
                                    );
                                }
#endif

                                int32_t numChanged = 0;
                                for (int32_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                                {
                                    numChanged++;

                                    Farlor::Vector3 shiftedPoint = newPoints[pointIdx] - leftPoint;
                                    // Rotate and stuff back in shifted point
                                    RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));

                                    // Update the point with the rotated version
                                    newPoints[pointIdx] = shiftedPoint + leftPoint;
                                }
                            }
                        }
                        else if (currentMethod == MethodType::Spring)
                        {
                            if (justSwitchedPerturbMethod)
                            {
                                justSwitchedPerturbMethod = false;

                                for (uint32_t ptIdx = 0; ptIdx < (numSegmentsPerCurve + 1); ptIdx++)
                                {
                                    // The left point is the first point of the first movable segment
                                    oldPoints[ptIdx] = newPoints[ptIdx];
                                    prevPoints[ptIdx] = newPoints[ptIdx];
                                }

                                gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]), zeroToOneUniformDist(rngGenerators[threadIdx]));
                            }

                            // Spring
                            {
                                // Reset the force vector
                                // All points
                                for (auto& force : netForcePerPoint)
                                {
                                    // Zero out the force
                                    force = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                                }

                                // Segment Springs
                                {
                                    for (uint32_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 1); ptIdx++)
                                    {
                                        uint32_t leftIdx = ptIdx;
                                        uint32_t rightIdx = ptIdx + 1;
                                        // The left point is the first point of the first movable segment
                                        Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
                                        // The right point is one to the right of that
                                        Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];

                                        Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, segmentStiffness, desiredSegmentLengthSpring);
                                        // Only apply to left in this case
                                        netForcePerPoint[leftIdx] -= forceAonB;
                                        netForcePerPoint[rightIdx] += forceAonB;
                                    }
                                }

                                // Joint Springs
                                {
                                    for (uint32_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 2); ptIdx++)
                                    {
                                        uint32_t leftIdx = ptIdx;
                                        uint32_t rightIdx = ptIdx + 2;
                                        // The left point is the first point of the first movable segment
                                        Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
                                        // The right point is one to the right of that
                                        Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];

                                        Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, jointStiffness, desiredJointLengthSpring);
                                        // Only apply to left in this case
                                        netForcePerPoint[leftIdx] -= forceAonB;
                                        netForcePerPoint[rightIdx] += forceAonB;
                                    }
                                }


                                // Add in gravity cause why not
                                for (uint32_t ptIdx = 0; ptIdx < netForcePerPoint.size(); ptIdx++)
                                {
                                    netForcePerPoint[ptIdx] += gravityForce * pointMass;
                                }

                                // Force the three set points to have no force
                                {
                                    netForcePerPoint[0] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                                    netForcePerPoint[1] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                                    netForcePerPoint[numSegmentsPerCurve] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                                }

                                // Dont update point 0, 1 and M
                                for (int32_t pointIdx = 2; pointIdx < numSegmentsPerCurve; ++pointIdx)
                                {
                                    Farlor::Vector3 acc = netForcePerPoint[pointIdx] * (1.0f / pointMass);
                                    newPoints[pointIdx] = 2.0f * prevPoints[pointIdx] - oldPoints[pointIdx] + acc * timestep * timestep;
                                }

                                // Assert points are at the start and end correctly
                                if (newPoints[0] != m_upInitialCurve->m_basePos)
                                {
                                    std::cout << "Path perturb failed as start pos moved" << std::endl;
                                    std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
                                }

                                if (newPoints[numSegmentsPerCurve] != m_upInitialCurve->m_targetPos)
                                {
                                    std::cout << "Path perturb failed as end pos moved" << std::endl;
                                    std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
                                }

                                assert(newPoints[0] == m_upInitialCurve->m_basePos);
                                assert(newPoints[numSegmentsPerCurve] == m_upInitialCurve->m_targetPos);
                            }

                            // Store the older versions of the points
                            oldPoints = prevPoints;
                            prevPoints = newPoints;
                        }
                        else
                        {
                            // Method not implemented
                            assert(false);
                        }

                        // Update points from current buffer
                        for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                        {
                            globalPos[CurrentThreadPosStartIdx + pointIdx] = newPoints[pointIdx];
                        }

                        // Update all tangents
                        for (int32_t tanIdx = 0; tanIdx < numSegmentsPerCurve; ++tanIdx)
                        {
                            Farlor::Vector3 leftPos = globalPos[CurrentThreadPosStartIdx + tanIdx];
                            Farlor::Vector3 rightPos = globalPos[CurrentThreadPosStartIdx + (tanIdx + 1)];

                            globalTans[CurrentThreadTanStartIdx + tanIdx] = (rightPos - leftPos).Normalized();
                        }

                        // Update curvature values
                        for (int32_t curvatureIdx = 0; curvatureIdx < numSegmentsPerCurve; ++curvatureIdx)
                        {
                            Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + curvatureIdx];
                            Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + (curvatureIdx + 1)];

                            Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);

                            const float curvature = temp.Magnitude();
                            globalCurvatures[CurrentThreadCurvatureStartIdx + curvatureIdx] = curvature;

                            // Also, cache the weight of that changed segment
                            float distance = curvature - minCurvature;
                            float realIdx = distance / curvatureStepSize;
                            int32_t leftIdx = floor(realIdx);
                            int32_t rightIdx = leftIdx + 1;

                            double leftLookup = lookupTable[leftIdx];
                            double rightLookup = lookupTable[rightIdx];

                            float leftDist = distance - (leftIdx * curvatureStepSize);

                            double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                            //std::cout << "Interpolated result: " << interpolatedResult << std::endl;
                            double interpolatedResultLog = log(interpolatedResult);

                            double segmentWeight = interpolatedResultLog;

#ifdef DelayedAbsorbtionContribution
                            // Do nothing, we dont add it in here
#else
                            // Take natural log of this constant
                            segmentWeight += lnAbsorbtionConst;
#endif

                            // Add segment weighting into running path weight
                            pathWeight += segmentWeight;
                        }
                    }

                    if (pathCount < numPathsToSkipPerThread)
                    {
                        // Skip
                    }
                    else
                    {
                        currentMethodCount++;
                        if (currentMethodCount >= HybridRunCounts[static_cast<uint32_t>(currentMethod)])
                        {
                            // Reset the current method count
                            currentMethodCount = 0;

                            // Switch method
                            currentMethod = static_cast<MethodType>(static_cast<uint32_t>(currentMethod) + 1);
                            if (static_cast<uint32_t>(currentMethod) >= static_cast<uint32_t>(MethodType::Count))
                            {
                                currentMethod = static_cast<MethodType>(0);
                            }

                            justSwitchedPerturbMethod = true;
                        }

                        // Else, contribute to the paths
                        int32_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                        assert(currentPathIdx >= numPathsPerThread * threadIdx);

                        //std::cout << "----- Path: " << currentPathIdx << std::endl;
                        globalPathWeights[currentPathIdx] = pathWeight;


#if defined(ExportPathBatches)
                        // Add the path to the path batch
                        for (int32_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
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
#ifdef SerialMultithread
        activeThreadIdx--;
#endif
    }

    ExperimentRunner::ExperimentResults FullExperimentRunner::RunExperiment()
    {
        return RunExperimentLogWeightTable();
    }

    ExperimentRunner::ExperimentResults FullExperimentRunner::RunExperimentLogWeightTable()
    {
        uint32_t numFailures = 0;
        uint32_t totalFailures = 0;
        uint32_t totalSuccess = 0;

        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();

        /* --------------------- */
        auto setupTimeStart = std::chrono::high_resolution_clock::now();

#if defined(ExportPathBatches)
        {
            BeginPathBatchOutput();

            std::filesystem::path currentPath = std::filesystem::current_path();
            currentPath.append(m_experimentParams.pathBatchDirectory);

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
        float ds = m_upInitialCurve->m_arclength / m_experimentParams.numSegmentsPerCurve;

        // Constants
        double minCurvature = 0.0;
        double maxCurvature = (3.47 / ds) * 2.0;


        twisty::PathSpaceUtils::WeightLookupTableIntegral lookupEvaluator(ds, m_experimentParams.weightingParameters.mu, m_experimentParams.weightingParameters.numStepsInt,
            m_experimentParams.weightingParameters.minBound, m_experimentParams.weightingParameters.maxBound, m_experimentParams.weightingParameters.eps,
            minCurvature, maxCurvature, m_experimentParams.weightingParameters.numCurvatureSteps, m_experimentParams.weightingParameters.scatter);
        const float curvatureStepSize = (maxCurvature - minCurvature) / m_experimentParams.weightingParameters.numCurvatureSteps;

        // This is a horible hack to make sure we always purturb a new curve
        Curve curveToBend = *m_upInitialCurve;

        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        // Create threads and dispatch them
#ifdef SINGLE_THREAD_PERTURB_MODE
        uint32_t numPurturbThreads = 1;
#else
#ifdef HardcodedNumPurturbThreads
        uint32_t numPurturbThreads = 3;
#else
        uint32_t numPurturbThreads = std::thread::hardware_concurrency();
#endif
#endif
        std::cout << "We have " << numPurturbThreads << " avalible for purturbation." << std::endl;

        // Setup rng stuff
        std::vector<std::mt19937> perThreadRngGenerators(numPurturbThreads);
        uint32_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }
        for (uint32_t i = 0; i < numPurturbThreads; ++i)
        {
            perThreadRngGenerators[i] = std::mt19937(seed);
        }

        // Setup data structures
        std::vector<Farlor::Vector3> initialCurvePositions(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> initialCurveTangents(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<float> initialCurveCurvatures(m_experimentParams.numSegmentsPerCurve);

        // Positions
        // Hard code the first two positions
        initialCurvePositions[0] = m_upInitialCurve->m_basePos;
        initialCurvePositions[1] = m_upInitialCurve->m_basePos + m_upInitialCurve->m_baseTangent.Normalized() * m_upInitialCurve->m_segmentLength;
        for (uint32_t segmentIdx = 2; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
        {
            initialCurvePositions[segmentIdx] = m_upInitialCurve->m_positions[segmentIdx];
        }
        // Hard code the final position
        initialCurvePositions[m_experimentParams.numSegmentsPerCurve] = m_upInitialCurve->m_targetPos;

#if defined(DetailedPurturb)
        {
            std::cout << "Positions" << std::endl;
            for (uint32_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurvePositions[segmentIdx] << std::endl;
            }
        }
#endif

        // Tangents
        // Hardcode intial tangent
        initialCurveTangents[0] = m_upInitialCurve->m_baseTangent;
        for (uint32_t tanIdx = 1; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
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
            for (uint32_t tanIdx = 0; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
            {
                std::cout << "\t" << initialCurveTangents[tanIdx] << std::endl;
            }
        }
#endif

        // Curvatures
        float segLength = m_upInitialCurve->m_arclength / m_upInitialCurve->m_numSegments;
        for (uint32_t curvatureIdx = 0; curvatureIdx < m_experimentParams.numSegmentsPerCurve; ++curvatureIdx)
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
            for (uint32_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurveCurvatures[segmentIdx] << std::endl;
            }
        }
#endif

        const uint32_t NumPosPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const uint32_t NumTanPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const uint32_t NumCurvaturePerCurve = (m_upInitialCurve->m_numSegments);

        std::vector<Farlor::Vector3> perThreadCurvePositions(NumPosPerCurve * numPurturbThreads);
        std::vector<Farlor::Vector3> perThreadCurveTangents(NumTanPerCurve * numPurturbThreads);
        std::vector<float> perThreadCurveCurvatures(NumCurvaturePerCurve * numPurturbThreads);

        for (uint32_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
        {
            // Copy Pos
            for (uint32_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
            {
                perThreadCurvePositions[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
            }

            // Copy Tan
            for (uint32_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                perThreadCurveTangents[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
            }

            // Copy Curvatures
            for (uint32_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
            {
                perThreadCurveCurvatures[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
            }
        }

        std::vector<double> cachedSegmentWeights(m_experimentParams.numSegmentsPerCurve * numPurturbThreads);
        std::vector<double> compressedWeightBuffer(m_experimentParams.numPathsPerBatch);

        auto setupTimeEnd = std::chrono::high_resolution_clock::now();
        /* --------------------- */


        /* --------------------- */
        auto perturbTimeStart = std::chrono::high_resolution_clock::now();

#ifdef SerialMultithread
        activeThreadIdx.store(numPurturbThreads - 1);
#endif


        const uint32_t numDispatches = (m_experimentParams.numPathsInExperiment + m_experimentParams.numPathsPerBatch - 1) / m_experimentParams.numPathsPerBatch;
        std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
        std::cout << "numPathsPerBatch: " << m_experimentParams.numPathsPerBatch << std::endl;
        std::cout << "Num dispatches required: " << numDispatches << std::endl;
        uint32_t numPathsLeft = m_experimentParams.numPathsInExperiment;
        uint32_t numPathsGenerated = 0;

        // We need to calculate the absorbtion/scattering piece
        boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;

        long long perturbTimeCount = 0;
        long long weightCalcTimeCount = 0;

        // We need a number of dispatches
        for (uint32_t dispatchIdx = 0; dispatchIdx < numDispatches; ++dispatchIdx)
        {
            uint32_t pathsInDispatch = std::min(m_experimentParams.numPathsPerBatch, numPathsLeft);
            std::cout << "Paths in dispatch " << dispatchIdx << ": " << pathsInDispatch << std::endl;
            {
                uint32_t numPathsPerThread = (pathsInDispatch + numPurturbThreads - 1) / numPurturbThreads;

                std::vector<std::thread> threads(numPurturbThreads);

                for (uint32_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
                {
                    std::thread newThread(&FullExperimentRunner::LogWeightThreadFunction, this,
                        threadIdx,
                        pathsInDispatch,
                        numPathsPerThread,
                        m_experimentParams.numPathsToSkip,
                        m_experimentParams.numSegmentsPerCurve,
                        std::ref(perThreadRngGenerators),
                        std::ref(perThreadCurvePositions),
                        std::ref(perThreadCurveTangents),
                        std::ref(perThreadCurveCurvatures),
                        std::ref(compressedWeightBuffer),
                        std::ref(cachedSegmentWeights),
                        m_upInitialCurve->m_segmentLength,
                        m_experimentParams.weightingParameters.scatter,
                        m_experimentParams.weightingParameters.absorbtion,
                        std::ref(lookupEvaluator.AccessLookupTable()),
                        minCurvature,
                        maxCurvature,
                        curvatureStepSize
                    );

                    //std::thread newThread(&FullExperimentRunner::SpringBasedPerturb, this,
                    //    threadIdx,
                    //    pathsInDispatch,
                    //    numPathsPerThread,
                    //    m_experimentParams.numPathsToSkip,
                    //    m_experimentParams.numSegmentsPerCurve,
                    //    std::ref(perThreadRngGenerators),
                    //    std::ref(perThreadCurvePositions),
                    //    std::ref(perThreadCurveTangents),
                    //    std::ref(perThreadCurveCurvatures),
                    //    std::ref(compressedWeightBuffer),
                    //    std::ref(cachedSegmentWeights),
                    //    m_upInitialCurve->m_segmentLength,
                    //    m_experimentParams.weightingParameters.scatter,
                    //    m_experimentParams.weightingParameters.absorbtion,
                    //    std::ref(lookupEvaluator.AccessLookupTable()),
                    //    minCurvature,
                    //    maxCurvature,
                    //    curvatureStepSize
                    //);

                    //std::thread newThread(&FullExperimentRunner::HybridMethod, this,
                    //    threadIdx,
                    //    pathsInDispatch,
                    //    numPathsPerThread,
                    //    m_experimentParams.numPathsToSkip,
                    //    m_experimentParams.numSegmentsPerCurve,
                    //    std::ref(perThreadRngGenerators),
                    //    std::ref(perThreadCurvePositions),
                    //    std::ref(perThreadCurveTangents),
                    //    std::ref(perThreadCurveCurvatures),
                    //    std::ref(compressedWeightBuffer),
                    //    std::ref(cachedSegmentWeights),
                    //    m_upInitialCurve->m_segmentLength,
                    //    m_experimentParams.weightingParameters.scatter,
                    //    m_experimentParams.weightingParameters.absorbtion,
                    //    std::ref(lookupEvaluator.AccessLookupTable()),
                    //    minCurvature,
                    //    maxCurvature,
                    //    curvatureStepSize
                    //);

                    threads[threadIdx] = std::move(newThread);
                }

                for (uint32_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
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

            std::vector<boost::multiprecision::cpp_dec_float_100> bigFloatWeights(pathsInDispatch);
            uint32_t numWeightingThreads = std::thread::hardware_concurrency();

            twisty::BigFloat totalDispatchWeight = 0.0f;
            uint32_t numWeightsPerThread = (pathsInDispatch + numWeightingThreads - 1) / numWeightingThreads;
            {
                std::vector<std::thread> threads(numWeightingThreads);
                std::vector<twisty::BigFloat> threadWeights(numWeightingThreads);
                for (uint32_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    threadWeights[threadIdx] = 0.0;
                    std::thread newThread(&FullExperimentRunner::WeightCombineThreadKernel, this, threadIdx, pathsInDispatch, numWeightsPerThread, m_upInitialCurve->m_arclength,
                        m_upInitialCurve->m_numSegments, std::ref(compressedWeightBuffer), std::ref(bigFloatWeights), std::ref(threadWeights[threadIdx]));
                    threads[threadIdx] = std::move(newThread);
                }

                for (uint32_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    if (threads[threadIdx].joinable())
                    {
                        threads[threadIdx].join();
                    }
                }

#if defined(DetailedPurturb)
                {
                    std::cout << "Thread weights: " << std::endl;
                    for (uint32_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                    {
                        std::cout << threadWeights[threadIdx] << std::endl;
                    }
                }
#endif

                for (uint32_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    totalDispatchWeight += threadWeights[threadIdx];
                }
            }

            std::cout << "Dispatch " << dispatchIdx << " Weight: " << totalDispatchWeight << std::endl;
            bigTotalExperimentWeight += totalDispatchWeight;

            auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
            weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightingTimeEnd - weightingTimeStart).count();
            /* --------------------- */

#ifdef OutputBigFloatPathWeights
            std::string bigfloatOutputFile = m_experimentParams.experimentName;
            bigfloatOutputFile += "_BigFloatWeights.txt";
            std::ofstream bigfloatOFS(bigfloatOutputFile);

            bigfloatOFS << bigFloatWeights.size() << std::endl;
            for (uint32_t i = 0; i < bigFloatWeights.size(); ++i)
            {
                bigfloatOFS << bigFloatWeights[i] << std::endl;
            }
#endif



            numPathsLeft -= pathsInDispatch;
            numPathsGenerated += pathsInDispatch;
        }


        std::cout << "Big total weight before: " << bigTotalExperimentWeight << std::endl;

#ifdef DelayedAbsorbtionContribution
        double c = scatter + m_experimentParams.weightingParameters.absorbtion;
        double constant = std::exp(-c * m_upInitialCurve->m_segmentLength) / (2.0 * TwistyPi * TwistyPi);

        std::cout << "Absorbtion const delayed: " << constant << std::endl;

        for (uint32_t i = 0; i < m_experimentParams.numSegmentsPerCurve; i++)
        {
            bigTotalExperimentWeight *= constant;
        }
#endif


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

    void FullExperimentRunner::WeightCombineThreadKernel(const uint32_t threadIdx, uint32_t numWeights, uint32_t numWeightsPerThread, float arclength, uint32_t numSegmentsPerCurve,
        const std::vector<double>& compressedWeights, std::vector<boost::multiprecision::cpp_dec_float_100>& bigFloatWeights, twisty::BigFloat& threadWeight)
    {
        // Hardcoded value from Jerry analysis.
        boost::multiprecision::cpp_dec_float_100 singleSegmentNormalizer = 2.0 * TwistyPi * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(0.625));
        boost::multiprecision::cpp_dec_float_100 segmentNormalizer = 1.0;
        for (uint32_t segIdx = 0; segIdx < (numSegmentsPerCurve - 1); ++segIdx)
        {
            segmentNormalizer = segmentNormalizer * singleSegmentNormalizer;
        }

        // Full path normalization term
        boost::multiprecision::cpp_dec_float_100 pathNormalizer = 1.0;
        pathNormalizer = pathNormalizer * boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(static_cast<float>(numSegmentsPerCurve) / arclength), 3.0);
        pathNormalizer = pathNormalizer * segmentNormalizer;
        pathNormalizer = pathNormalizer * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(-0.325));

        for (uint32_t i = 0; i < numWeightsPerThread; i++)
        {
            uint32_t idx = threadIdx * numWeightsPerThread + i;
            if (idx >= numWeights)
            {
                break;
            }

            twisty::BigFloat bigfloatCompressed = compressedWeights[idx];
#ifdef BigFloatMultiprecision
            boost::multiprecision::cpp_dec_float_100 decompressed = boost::multiprecision::exp(bigfloatCompressed);
            // Pulled from Jerry analysis
            decompressed = decompressed * pathNormalizer;
            bigFloatWeights[idx] = decompressed;
            threadWeight += decompressed;
#else
            //threadWeight += std::exp(bigfloatCompressed);
#endif
        }
    }

    void FullExperimentRunner::Shutdown()
    {
    }
}