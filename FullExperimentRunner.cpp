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

#if defined(EnforceSpringLengthDistance)
const double pathLengthThreshold = 0.01;
#endif

//#define HardcodedSegments
//#define HardcodedRotation

#define SINGLE_THREAD_PERTURB_MODE
#define OutputBigFloatPathWeights
//#define SerialMultithread
//#define BlockingMultithread
//#define BlockingOutputThread

#define ExportPathBatches

#if defined(ExportPathBatches)

const static int64_t ExportPathBatchCacheSize = 100000;

static std::mutex ExportPathBatchesMutex;
static std::ofstream curvesBinaryFile;
static std::ofstream curvesMetadataFile;

#else

const int64_t ExportPathBatchCacheSize = 1000000;

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
        Bootstrapper& bootstrapper)
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
            m_upInitialCurve = m_bootstrapper.CreateCurve(m_experimentParams.numSegmentsPerCurve, m_experimentParams.arclength, m_experimentParams.bootstrapSeed);
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

        const std::filesystem::path experimentDirPath = m_experimentParams.experimentDirPath;
        if (!std::filesystem::exists(experimentDirPath))
        {
            std::filesystem::create_directories(experimentDirPath);
        }

        return true;
    }

    void FullExperimentRunner::GeometryPerturb(
        int64_t threadIdx,
        int64_t numExperimentPaths,
        int64_t numPathsPerThread,
        int64_t numPathsToSkipPerThread,
        int64_t numSegmentsPerCurve,
        std::vector<std::mt19937_64>& rngGenerators,
        std::vector<Farlor::Vector3>& globalPos,
        std::vector<Farlor::Vector3>& globalTans,
        std::vector<float>& globalCurvatures,
        std::vector<Farlor::Vector3>& scratchPositionSpace,
        std::vector<Farlor::Vector3>& scratchTangentSpace,
        std::vector<float>& scratchCurvatureSpace,
        std::vector<double>& globalPathWeights,
        std::vector<double>& cachedSegmentWeights,
        float segmentLength,
        const twisty::PathWeighting::WeightLookupTableIntegral& weightingIntegral,
        const twisty::PerturbUtils::BoundrayConditions& boundaryConditions,
        const PathWeighting::NormalizerStuff::FN& fn
    )
    {
        uint32_t numPathsAccepted = 0;
        uint32_t numPathsUnaccepted = 0;
        uint32_t numPathsUnacceptedTangentPdf = 0;
        uint32_t numPathsUnacceptedCurvaturePdf = 0;

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

            for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
            {
                // Expect to go negative, thus int
                int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                // We can exit once this point is reached as we have generated all the paths necessary for this thread
                if (currentPathIdx >= numExperimentPaths)
                {
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

                // Each time, we first copy the "old path" to the "scratch space"
                for (uint32_t segIdx = 0; segIdx <= numSegmentsPerCurve; ++segIdx)
                {
                    scratchPositionSpace[CurrentThreadPosStartIdx + segIdx] = globalPos[CurrentThreadPosStartIdx + segIdx];

                }
                twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&scratchPositionSpace[CurrentThreadPosStartIdx],
                    &scratchTangentSpace[CurrentThreadTanStartIdx], &scratchCurvatureSpace[CurrentThreadCurvatureStartIdx],
                    numSegmentsPerCurve, boundaryConditions);

#ifdef HardcodedSegments
                int64_t leftPointIndex = 25;
                int64_t rightPointIndex = 75;
#else
                // Diff can range from 2 to 10
                std::uniform_int_distribution<int> diffDist(2, 10); // uniform, unbiased
                int64_t diff = diffDist(rngGenerators[threadIdx]);

                std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - 1 - diff); // uniform, unbiased
                int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);

                int64_t rightPointIndex = leftPointIndex + diff;
#endif
                assert(leftPointIndex < rightPointIndex);
                assert((rightPointIndex - leftPointIndex) >= diff);

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                {
                    printf("Left point idx: %d\n", leftPointIndex);
                    printf("Right point idx: %d\n", rightPointIndex);
                }
#endif
                // We need two frames for each segment to get the new curvature and torsion.
                // we need the frame left of the segment, as well as the frame right of the segment.
                // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                const Farlor::Vector3 leftPoint = scratchPositionSpace[CurrentThreadPosStartIdx + leftPointIndex];
                const Farlor::Vector3 rightPoint = scratchPositionSpace[CurrentThreadPosStartIdx + rightPointIndex];

                const Farlor::Vector3 axisOfRotation = (rightPoint - leftPoint).Normalized();

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                printf("Axis before (%.6f, %.6f, %.6f)\n",
                    axisOfRotation[0], axisOfRotation[1], axisOfRotation[2]
                );
#endif

#ifdef HardcodedRotation
                float randomAngle = 1.38f;
#else
                std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * AmountOfFullRotation, TwistyPi * AmountOfFullRotation);
                const float randomAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
#endif               

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                {
                    printf("randomAngle: %.6f\n", randomAngle);
                }
#endif

                float rotationMatrix[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                RotationMatrixAroundAxis(randomAngle, (float*)(&axisOfRotation), rotationMatrix);
#if defined(DetailedPurturb) && defined(SingleThreadMode)
                {
                    printf("Normalized axis(%.6f, %.6f, %.6f)\n",
                        axisOfRotation[0], axisOfRotation[1], axisOfRotation[2]
                    );

                    printf("Rotation Matrix\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n",
                        rotationMatrix[0], rotationMatrix[1], rotationMatrix[2],
                        rotationMatrix[3], rotationMatrix[4], rotationMatrix[5],
                        rotationMatrix[6], rotationMatrix[7], rotationMatrix[8]
                    );
                }
#endif

                for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                {
                    Farlor::Vector3 shiftedPoint = scratchPositionSpace[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                    // Rotate and stuff back in shifted point
                    RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));
                    // Update the point with the rotated version
                    scratchPositionSpace[CurrentThreadPosStartIdx + pointIdx] = shiftedPoint + leftPoint;
                }

                //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                //We can do a different approach later.
                // Here, we want to do a perturb update call
                twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&scratchPositionSpace[CurrentThreadPosStartIdx],
                    &scratchTangentSpace[CurrentThreadTanStartIdx], &scratchCurvatureSpace[CurrentThreadCurvatureStartIdx],
                    numSegmentsPerCurve, boundaryConditions);

                uint32_t numBetas = (rightPointIndex - 1) - leftPointIndex;
                std::vector<Farlor::Vector3> oldBetas(numBetas);
                std::vector<Farlor::Vector3> newBetas(numBetas);
                for (int64_t tanIdx = 0; tanIdx < numBetas; ++tanIdx)
                {
                    oldBetas[tanIdx] = globalTans[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];
                    newBetas[tanIdx] = scratchTangentSpace[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];
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

#ifdef SINGLE_THREAD_PERTURB_MODE
                    //std::cout << "Acceptance Prob: " << acceptanceProb << std::endl;
                    //std::cout << "Likelihood: " << likelihood << std::endl;
#endif

                    if (acceptanceProb <= likelihood)
                    {

                        double oldPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(globalCurvatures[CurrentThreadCurvatureStartIdx]),
                            numSegmentsPerCurve, weightingIntegral);

                        double newPathWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(&(scratchCurvatureSpace[CurrentThreadCurvatureStartIdx]),
                            numSegmentsPerCurve, weightingIntegral);

                        double lambdaLog10 = newPathWeightLog10 - oldPathWeightLog10;

                        double weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
                        while (weightAcceptance == 0)
                        {
                            weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
                        }

                        double weightAcceptanceLog10 = std::log10(weightAcceptance);

                        if (lambdaLog10 > weightAcceptanceLog10)
                        {
                            accepted = true;
                            for (uint32_t i = 0; i <= numSegmentsPerCurve; i++)
                            {
                                globalPos[CurrentThreadPosStartIdx + i] = scratchPositionSpace[CurrentThreadPosStartIdx + i];
                            }

                            twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&globalPos[CurrentThreadPosStartIdx],
                                &globalTans[CurrentThreadTanStartIdx], &globalCurvatures[CurrentThreadCurvatureStartIdx],
                                numSegmentsPerCurve, boundaryConditions);
                        }
                        else
                        {
                            numPathsUnacceptedCurvaturePdf++;
                            accepted = false;
                        }
                    }
                    else
                    {
                        numPathsUnacceptedTangentPdf++;
                        // We reject
                        accepted = false;
                    }
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
                        int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                        assert(currentPathIdx >= numPathsPerThread * threadIdx);
                        globalPathWeights[currentPathIdx] = pathWeightLog10;

#if defined(ExportPathBatches)

                        //std::cout << "Accepted curve: " << std::endl;
                        //for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                        //{
                        //    std::cout << "\t" << globalPos[CurrentThreadPosStartIdx + pointIdx] << std::endl;
                        //}

                        // Add the path to the path batch
                        for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                        {
                            Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
                            pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
                        }
                        numCurvesInBatch++;

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
                else
                {
                    // Go back one path as we are redoing
                    pathCount--;
                    numPathsUnaccepted++;
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

        std::cout << "Num path accepted: " << numPathsAccepted << std::endl;
        std::cout << "Num path unaccepted: " << numPathsUnaccepted << std::endl;
        std::cout << "Num path unaccepted tangents: " << numPathsUnacceptedTangentPdf << std::endl;
        std::cout << "Num path unaccepted curvature: " << numPathsUnacceptedCurvaturePdf << std::endl;
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

//    void FullExperimentRunner::SpringBasedPerturb(
//        int64_t threadIdx,
//        int64_t numExperimentPaths,
//        int64_t numPathsPerThread,
//        int64_t numPathsToSkipPerThread,
//        int64_t numSegmentsPerCurve,
//        std::vector<std::mt19937_64>& rngGenerators,
//        std::vector<Farlor::Vector3>& globalPos,
//        std::vector<Farlor::Vector3>& globalTans,
//        std::vector<float>& globalCurvatures,
//        std::vector<double>& globalPathWeights,
//        std::vector<double>& cachedSegmentWeights,
//        float segmentLength,
//        float scattering,
//        float absorbtion,
//        const std::vector<double>& lookupTable,
//        float minCurvature,
//        float maxCurvature,
//        float curvatureStepSize
//    )
//    {
//        if (threadIdx == 11)
//        {
//            std::cout << "Thread 12 hit" << std::endl;
//        }
//
//#ifdef SerialMultithread
//        while (activeThreadIdx.load() != threadIdx)
//        {
//        };
//#endif
//
//#ifdef BlockingOutputThread
//        {
//            std::scoped_lock<std::mutex> lock(outputThreadMutex);
//            std::cout << "On thread: " << threadIdx << std::endl;
//        }
//#endif
//
//#if defined(ExportPathBatches)
//        // This should be per thread
//        int64_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
//        std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
//        {
//            for (int64_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
//            {
//                for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//                {
//                    int64_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
//                    pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                }
//            }
//        }
//#endif
//
//        // First, we discard random numbers to match the previous 
//        rngGenerators[threadIdx].discard(numPathsPerThread * threadIdx);
//
//        const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
//        const int64_t NumTanPerCurve = (numSegmentsPerCurve + 1);
//        const int64_t NumCurvaturesPerCurve = numSegmentsPerCurve;
//
//        const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
//        const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
//        const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;
//
//        // We store forces for all points.
//        // Movable points have forces updated.
//        // This excluedes the first two points, as the first segment is locked
//        // This also excludes the final point. They still have springs attached, however they remain at zero
//        std::vector<Farlor::Vector3> netForcePerPoint(numSegmentsPerCurve + 1);
//        for (auto& force : netForcePerPoint)
//        {
//            // Zero out the force
//            force = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//        }
//        
//        std::uniform_real_distribution<float> zeroToOneUniformDist(0.0f, 1.0f);
//
//        std::vector<Farlor::Vector3> oldPoints(numSegmentsPerCurve + 1);
//        std::vector<Farlor::Vector3> prevPoints(numSegmentsPerCurve + 1);
//        std::vector<Farlor::Vector3> newPoints(numSegmentsPerCurve + 1);
//
//        for (int64_t ptIdx = 0; ptIdx < netForcePerPoint.size(); ptIdx++)
//        {
//            // The left point is the first point of the first movable segment
//            Farlor::Vector3 point = globalPos[CurrentThreadPosStartIdx + ptIdx];
//
//            oldPoints[ptIdx] = point;
//            prevPoints[ptIdx] = point;
//            newPoints[ptIdx] = point;
//        }
//
//        double initialPathArclength = 0.0;
//        for (int64_t ptIdx = 0; ptIdx < numSegmentsPerCurve; ptIdx++)
//        {
//            auto& leftPt = newPoints[ptIdx];
//            auto& rightPt = newPoints[ptIdx + 1];
//            initialPathArclength += (rightPt - leftPt).Magnitude();
//        }
//
//        // Assume we have a mass of 1
//        float pointMass = 0.1f;
//        float segmentStiffness = 10000.0f;
//        float jointStiffness = 1000.0f;
//        float desiredSegmentLengthSpring = segmentLength * 1.0f;
//        float desiredJointLengthSpring = segmentLength * 2.0f;
//        int64_t gravityRate = 10000;
//
//        Farlor::Vector3 gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]),
//            zeroToOneUniformDist(rngGenerators[threadIdx]));
//
//        // Update "15 times a second"
//        float timestep = 1.0f / 1000;
//
//        // Caclulate the first global path index this thread will start on
//        int64_t threadStartingPathIdx = numPathsPerThread * threadIdx;
//        float c = scattering + absorbtion;
//        float absorbtionConst = std::exp(-c * segmentLength) / (2.0 * TwistyPi * TwistyPi);
//        float lnAbsorbtionConst = log(absorbtionConst);
//
//        {
//#ifdef BlockingMultithread
//            std::scoped_lock<std::mutex> lock(blockingMultithreadMutex);
//#endif
//            // Now, we can begin the actual algorithm
//            {
//
//                // This is the perturbation piece.
//                // Can we do this in place, most likely
//                // This will modify pCurrentThreadCurve
//                // Remember, the structure of this is:
//                // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M
//
//                // Start at the thread's first path idx
//
//                int64_t numCurvesInBatch = 0;
//                int64_t outputIdx = 0;
//
//                int64_t cacheStartPathIdx = numPathsPerThread * threadIdx;
//                for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
//                {
//                    // Expect to go negative, thus int
//                    int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
//
//                    // We can exit once this point is reached as we have generated all the paths necessary for this thread
//                    if (currentPathIdx >= numExperimentPaths)
//                    {
//#ifdef BlockingOutputThread
//                        {
//                            std::scoped_lock<std::mutex> lock(outputThreadMutex);
//                            std::cout << "Returning, all paths complete" << std::endl;
//                        }
//#endif
//
//#if defined(ExportPathBatches)
//                        if (numCurvesInBatch > 0)
//                        {
//                            ExportPathBatchesMutex.lock();
//
//                            if (threadIdx == 11)
//                            {
//                                std::cout << "Should be exporting thread 12" << std::endl;
//                            }
//
//
//                            curvesMetadataFile << threadIdx << " ";
//                            curvesMetadataFile << outputIdx << " ";
//                            curvesMetadataFile << numCurvesInBatch << std::endl;
//
//                            curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//                            numCurvesInBatch = 0;
//                            outputIdx++;
//
//                            ExportPathBatchesMutex.unlock();
//                        }
//#endif
//
//                        // We dont want to continue if we have already generated the correct number of paths.
//                        break;
//                    }
//
//                    // Do the perturb now
//                    double pathWeight = 0.0;
//                    {
//                        {
//                            // Reset the force vector
//                            // All points
//                            for (auto& force : netForcePerPoint)
//                            {
//                                // Zero out the force
//                                force = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                            }
//
//                            // Segment Springs
//                            {
//                                for (int64_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 1); ptIdx++)
//                                {
//                                    int64_t leftIdx = ptIdx;
//                                    int64_t rightIdx = ptIdx + 1;
//                                    // The left point is the first point of the first movable segment
//                                    Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
//                                    // The right point is one to the right of that
//                                    Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];
//
//                                    Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, segmentStiffness, desiredSegmentLengthSpring);
//                                    // Only apply to left in this case
//                                    netForcePerPoint[leftIdx] -= forceAonB;
//                                    netForcePerPoint[rightIdx] += forceAonB;
//                                }
//                            }
//
//                            // Joint Springs
//                            {
//                                for (int64_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 2); ptIdx++)
//                                {
//                                    int64_t leftIdx = ptIdx;
//                                    int64_t rightIdx = ptIdx + 2;
//                                    // The left point is the first point of the first movable segment
//                                    Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
//                                    // The right point is one to the right of that
//                                    Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];
//
//                                    Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, jointStiffness, desiredJointLengthSpring);
//                                    // Only apply to left in this case
//                                    netForcePerPoint[leftIdx] -= forceAonB;
//                                    netForcePerPoint[rightIdx] += forceAonB;
//                                }
//                            }
//
//
//                            // Add in gravity cause why not
//                            for (int64_t ptIdx = 0; ptIdx < netForcePerPoint.size(); ptIdx++)
//                            {
//                                netForcePerPoint[ptIdx] += gravityForce * pointMass;
//                            }
//
//                            // Force the three set points to have no force
//                            {
//                                netForcePerPoint[0] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                                netForcePerPoint[1] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                                netForcePerPoint[numSegmentsPerCurve] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                            }
//
//                            // Dont update point 0, 1 and M
//                            for (int64_t pointIdx = 2; pointIdx < numSegmentsPerCurve; ++pointIdx)
//                            {
//                                Farlor::Vector3 acc = netForcePerPoint[pointIdx] * (1.0f / pointMass);
//                                newPoints[pointIdx] = 2.0f * prevPoints[pointIdx] - oldPoints[pointIdx] + acc * timestep * timestep;
//                            }
//
//                            // Assert points are at the start and end correctly
//                            if (newPoints[0] != m_upInitialCurve->m_basePos)
//                            {
//                                std::cout << "Path perturb failed as start pos moved" << std::endl;
//                                std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
//                            }
//
//                            if (newPoints[numSegmentsPerCurve] != m_upInitialCurve->m_targetPos)
//                            {
//                                std::cout << "Path perturb failed as end pos moved" << std::endl;
//                                std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
//                            }
//
//                            assert(newPoints[0] == m_upInitialCurve->m_basePos);
//                            assert(newPoints[numSegmentsPerCurve] == m_upInitialCurve->m_targetPos);
//                        }
//
//                        // Update points from current buffer
//                        for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//                        {
//                            globalPos[CurrentThreadPosStartIdx + pointIdx] = newPoints[pointIdx];
//                        }
//
//                        // Store the older versions of the points
//                        oldPoints = prevPoints;
//                        prevPoints = newPoints;
//
//                        // Update all tangents
//
//                        for (int64_t tanIdx = 0; tanIdx < numSegmentsPerCurve; ++tanIdx)
//                        {
//                            Farlor::Vector3 leftPos = globalPos[CurrentThreadPosStartIdx + tanIdx];
//                            Farlor::Vector3 rightPos = globalPos[CurrentThreadPosStartIdx + (tanIdx + 1)];
//
//                            globalTans[CurrentThreadTanStartIdx + tanIdx] = (rightPos - leftPos).Normalized();
//                        }
//
//                        // Update curvature values
//                        for (int64_t curvatureIdx = 0; curvatureIdx < numSegmentsPerCurve; ++curvatureIdx)
//                        {
//                            Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + curvatureIdx];
//                            Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + (curvatureIdx + 1)];
//
//                            Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);
//
//                            const float curvature = temp.Magnitude();
//                            globalCurvatures[CurrentThreadCurvatureStartIdx + curvatureIdx] = curvature;
//
//                            // Also, cache the weight of that changed segment
//                            float distance = curvature - minCurvature;
//                            float realIdx = distance / curvatureStepSize;
//                            int64_t leftIdx = floor(realIdx);
//                            int64_t rightIdx = leftIdx + 1;
//
//                            double leftLookup = lookupTable[leftIdx];
//                            double rightLookup = lookupTable[rightIdx];
//
//                            float leftDist = distance - (leftIdx * curvatureStepSize);
//
//                            double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
//                            //std::cout << "Interpolated result: " << interpolatedResult << std::endl;
//                            double interpolatedResultLog = log(interpolatedResult);
//
//                            double segmentWeight = interpolatedResultLog;
//
//#ifdef DelayedAbsorbtionContribution
//                            // Do nothing, we dont add it in here
//#else
//                            // Take natural log of this constant
//                            segmentWeight += lnAbsorbtionConst;
//#endif
//                            // Add segment weighting into running path weight
//                            pathWeight += segmentWeight;
//                        }
//                    }
//
//                    if (pathCount < numPathsToSkipPerThread)
//                    {
//                        // Skip
//                    }
//                    else
//                    {
//                        // Else, contribute to the paths
//                        int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
//                        assert(currentPathIdx >= numPathsPerThread * threadIdx);
//
//                        // Select new gravity
//                        if (currentPathIdx % gravityRate == 0)
//                        {
//                            gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]), zeroToOneUniformDist(rngGenerators[threadIdx]));
//                        }
//
//                        //std::cout << "----- Path: " << currentPathIdx << std::endl;
//                        globalPathWeights[currentPathIdx] = pathWeight;
//                        std::cout << "Done:" << std::endl;
//
//
//#if defined(ExportPathBatches)
//                        // Add the path to the path batch
//                        for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//                        {
//                            Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
//                            pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
//                        }
//
//                        numCurvesInBatch++;
//#endif
//
//#if defined(ExportPathBatches)
//                        if (numCurvesInBatch == ExportPathBatchCacheSize)
//                        {
//                            ExportPathBatchesMutex.lock();
//
//                            curvesMetadataFile << threadIdx << " ";
//                            curvesMetadataFile << outputIdx << " ";
//                            curvesMetadataFile << numCurvesInBatch << std::endl;
//
//                            curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//                            numCurvesInBatch = 0;
//                            outputIdx++;
//
//                            ExportPathBatchesMutex.unlock();
//                        }
//#endif
//
//
//                    }
//                }
//
//#if defined(ExportPathBatches)
//                if (numCurvesInBatch > 0)
//                {
//                    ExportPathBatchesMutex.lock();
//
//                    curvesMetadataFile << threadIdx << " ";
//                    curvesMetadataFile << outputIdx << " ";
//                    curvesMetadataFile << numCurvesInBatch << std::endl;
//
//                    curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//                    numCurvesInBatch = 0;
//                    outputIdx++;
//
//                    ExportPathBatchesMutex.unlock();
//                }
//#endif
//
//            }
//        }
//#ifdef SerialMultithread
//        activeThreadIdx--;
//#endif
//    }




//    void FullExperimentRunner::HybridMethod(
//        int64_t threadIdx,
//        int64_t numExperimentPaths,
//        int64_t numPathsPerThread,
//        int64_t numPathsToSkipPerThread,
//        int64_t numSegmentsPerCurve,
//        std::vector<std::mt19937>& rngGenerators,
//        std::vector<Farlor::Vector3>& globalPos,
//        std::vector<Farlor::Vector3>& globalTans,
//        std::vector<float>& globalCurvatures,
//        std::vector<double>& globalPathWeights,
//        std::vector<double>& cachedSegmentWeights,
//        float segmentLength,
//        float scattering,
//        float absorbtion,
//        const std::vector<double>& lookupTable,
//        float minCurvature,
//        float maxCurvature,
//        float curvatureStepSize
//    )
//    {
//#ifdef SerialMultithread
//        while (activeThreadIdx.load() != threadIdx)
//        {
//        };
//#endif
//
//#ifdef BlockingOutputThread
//        {
//            std::scoped_lock<std::mutex> lock(outputThreadMutex);
//            std::cout << "On thread: " << threadIdx << std::endl;
//        }
//#endif
//
//#if defined(ExportPathBatches)
//        // This should be per thread
//        int64_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
//        std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
//        {
//            for (int64_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
//            {
//                for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//                {
//                    int64_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
//                    pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                }
//            }
//        }
//#endif
//
//
//        // Both method variables
//        MethodType currentMethod = StartingHybridMethod;
//        int64_t currentMethodCount = 0;
//
//        const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
//        const int64_t NumTanPerCurve = (numSegmentsPerCurve + 1);
//        const int64_t NumCurvaturesPerCurve = numSegmentsPerCurve;
//
//        const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
//        const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
//        const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;
//
//        // First, we discard random numbers to match the previous 
//        rngGenerators[threadIdx].discard(numPathsPerThread * threadIdx);
//        
//        std::uniform_real_distribution<float> zeroToOneUniformDist(0.0f, 1.0f);
//        std::vector<Farlor::Vector3> newPoints(numSegmentsPerCurve + 1);
//        for (int64_t ptIdx = 0; ptIdx < (numSegmentsPerCurve + 1); ptIdx++)
//        {
//            // The left point is the first point of the first movable segment
//            Farlor::Vector3 point = globalPos[CurrentThreadPosStartIdx + ptIdx];
//            newPoints[ptIdx] = point;
//        }
//
//
//        // Geometry method specific stuff
//
//
//        // Spring force specific stuff
//        std::vector<Farlor::Vector3> oldPoints(numSegmentsPerCurve + 1);
//        std::vector<Farlor::Vector3> prevPoints(numSegmentsPerCurve + 1);
//        std::vector<Farlor::Vector3> netForcePerPoint(numSegmentsPerCurve + 1);
//
//        // Assume we have a mass of 1
//        const float pointMass = 0.1f;
//        const float segmentStiffness = 10000.0f;
//        const float jointStiffness = 1000.0f;
//        const float desiredSegmentLengthSpring = segmentLength * 1.0f;
//        const float desiredJointLengthSpring = segmentLength * 2.0f;
//        const int64_t gravityRate = 10000;
//
//        Farlor::Vector3 gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]),
//            zeroToOneUniformDist(rngGenerators[threadIdx]));
//
//        // Update "15 times a second"
//        float timestep = 1.0f / 1000;
//
//        // End Spring Specific Initialization
//
//        //// Initialization code
//        //bool done = false;
//        //while (!done)
//        //{
//        //    // If using the geometry method
//        //    if (currentMethod == 0)
//        //    {
//
//        //    }
//        //    // Spring method
//        //    else if (currentMethod == 1)
//        //    {
//
//        //    }
//        //    else
//        //    {
//        //        assert(false);
//        //        // Method not supported
//        //    }
//        //}
//
//
//
//        // Caclulate the first global path index this thread will start on
//        int64_t threadStartingPathIdx = numPathsPerThread * threadIdx;
//        float c = scattering + absorbtion;
//        float absorbtionConst = std::exp(-c * segmentLength) / (2.0 * TwistyPi * TwistyPi);
//        float lnAbsorbtionConst = log(absorbtionConst);
//
//        {
//#ifdef BlockingMultithread
//            std::scoped_lock<std::mutex> lock(blockingMultithreadMutex);
//#endif
//            // Now, we can begin the actual algorithm
//            {
//
//                // This is the perturbation piece.
//                // Can we do this in place, most likely
//                // This will modify pCurrentThreadCurve
//                // Remember, the structure of this is:
//                // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M
//
//                // Start at the thread's first path idx
//
//                int64_t numCurvesInBatch = 0;
//                int64_t outputIdx = 0;
//
//                bool justSwitchedPerturbMethod = true;
//
//                int64_t cacheStartPathIdx = numPathsPerThread * threadIdx;
//                for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
//                {
//                    // Expect to go negative, thus int
//                    int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
//
//                    // We can exit once this point is reached as we have generated all the paths necessary for this thread
//                    if (currentPathIdx >= numExperimentPaths)
//                    {
//#ifdef BlockingOutputThread
//                        {
//                            std::scoped_lock<std::mutex> lock(outputThreadMutex);
//                            std::cout << "Returning, all paths complete" << std::endl;
//                        }
//#endif
//
//#if defined(ExportPathBatches)
//                        if (numCurvesInBatch > 0)
//                        {
//                            ExportPathBatchesMutex.lock();
//
//                            if (threadIdx == 11)
//                            {
//                                std::cout << "Should be exporting thread 12" << std::endl;
//                            }
//
//
//                            curvesMetadataFile << threadIdx << " ";
//                            curvesMetadataFile << outputIdx << " ";
//                            curvesMetadataFile << numCurvesInBatch << std::endl;
//
//                            curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//                            numCurvesInBatch = 0;
//                            outputIdx++;
//
//                            ExportPathBatchesMutex.unlock();
//                        }
//#endif
//
//                        // We dont want to continue if we have already generated the correct number of paths.
//                        break;
//                    }
//
//                    // Do the perturb now
//                    double pathWeight = 0.0;
//                    {
//                        if (currentMethod == MethodType::Geometry)
//                        {
//
//
//
//                            // Do the perturb now
//                            {
//#ifdef HardcodedSegments
//                                int64_t leftPointIndex = 17;
//                                int64_t rightPointIndex = 39;
//#else
//                                std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - 3); // uniform, unbiased
//                                int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);
//                                std::uniform_int_distribution<int> rightPointIndexUniformDist(leftPointIndex + 2, numSegmentsPerCurve - 1); // uniform, unbiased
//                                int64_t rightPointIndex = rightPointIndexUniformDist(rngGenerators[threadIdx]);
//#endif
//
//                                assert(leftPointIndex < rightPointIndex);
//                                assert((rightPointIndex - leftPointIndex) >= 2);
//
//#if defined(DetailedPurturb) && defined(SingleThreadMode)
//                                {
//                                    printf("Left point idx: %d\n", leftPointIndex);
//                                    printf("Right point idx: %d\n", rightPointIndex);
//                                }
//#endif
//
//                                assert(leftPointIndex < rightPointIndex);
//                                assert((rightPointIndex - leftPointIndex) >= 2);
//
//                                // We need two frames for each segment to get the new curvature and torsion.
//                                // we need the frame left of the segment, as well as the frame right of the segment.
//
//                                // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
//                                Farlor::Vector3 leftPoint = newPoints[leftPointIndex];
//                                Farlor::Vector3 rightPoint = newPoints[rightPointIndex];
//
//                                Farlor::Vector3 axisOfRotation = (rightPoint - leftPoint).Normalized();
//
//#ifdef HardcodedRotation
//                                float randomAngle = 1.38f;
//#else
//                                std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * AmountOfFullRotation, TwistyPi * AmountOfFullRotation);
//                                float randomAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
//#endif               
//
//#if defined(DetailedPurturb) && defined(SingleThreadMode)
//                                {
//                                    printf("randomAngle: %.6f\n", randomAngle);
//                                }
//#endif
//
//                                float rotationMatrix[9];
//                                RotationMatrixAroundAxis(randomAngle, (float*)(&axisOfRotation), rotationMatrix);
//
//#if defined(DetailedPurturb) && defined(SingleThreadMode)
//                                {
//                                    printf("Rotation Matrix\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n\t(%.6f, %.6f, %.6f)\n",
//                                        rotationMatrix[0], rotationMatrix[1], rotationMatrix[2],
//                                        rotationMatrix[3], rotationMatrix[4], rotationMatrix[5],
//                                        rotationMatrix[6], rotationMatrix[7], rotationMatrix[8]
//                                    );
//                                }
//#endif
//
//                                int64_t numChanged = 0;
//                                for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
//                                {
//                                    numChanged++;
//
//                                    Farlor::Vector3 shiftedPoint = newPoints[pointIdx] - leftPoint;
//                                    // Rotate and stuff back in shifted point
//                                    RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));
//
//                                    // Update the point with the rotated version
//                                    newPoints[pointIdx] = shiftedPoint + leftPoint;
//                                }
//                            }
//                        }
//                        else if (currentMethod == MethodType::Spring)
//                        {
//                            if (justSwitchedPerturbMethod)
//                            {
//                                justSwitchedPerturbMethod = false;
//
//                                for (int64_t ptIdx = 0; ptIdx < (numSegmentsPerCurve + 1); ptIdx++)
//                                {
//                                    // The left point is the first point of the first movable segment
//                                    oldPoints[ptIdx] = newPoints[ptIdx];
//                                    prevPoints[ptIdx] = newPoints[ptIdx];
//                                }
//
//                                gravityForce = UniformDirection(zeroToOneUniformDist(rngGenerators[threadIdx]), zeroToOneUniformDist(rngGenerators[threadIdx]));
//                            }
//
//                            // Spring
//                            {
//                                // Reset the force vector
//                                // All points
//                                for (auto& force : netForcePerPoint)
//                                {
//                                    // Zero out the force
//                                    force = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                                }
//
//                                // Segment Springs
//                                {
//                                    for (int64_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 1); ptIdx++)
//                                    {
//                                        int64_t leftIdx = ptIdx;
//                                        int64_t rightIdx = ptIdx + 1;
//                                        // The left point is the first point of the first movable segment
//                                        Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
//                                        // The right point is one to the right of that
//                                        Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];
//
//                                        Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, segmentStiffness, desiredSegmentLengthSpring);
//                                        // Only apply to left in this case
//                                        netForcePerPoint[leftIdx] -= forceAonB;
//                                        netForcePerPoint[rightIdx] += forceAonB;
//                                    }
//                                }
//
//                                // Joint Springs
//                                {
//                                    for (int64_t ptIdx = 0; ptIdx < (netForcePerPoint.size() - 2); ptIdx++)
//                                    {
//                                        int64_t leftIdx = ptIdx;
//                                        int64_t rightIdx = ptIdx + 2;
//                                        // The left point is the first point of the first movable segment
//                                        Farlor::Vector3 leftPoint = globalPos[CurrentThreadPosStartIdx + leftIdx];
//                                        // The right point is one to the right of that
//                                        Farlor::Vector3 rightPoint = globalPos[CurrentThreadPosStartIdx + rightIdx];
//
//                                        Farlor::Vector3 forceAonB = SpringForceAonB(leftPoint, rightPoint, jointStiffness, desiredJointLengthSpring);
//                                        // Only apply to left in this case
//                                        netForcePerPoint[leftIdx] -= forceAonB;
//                                        netForcePerPoint[rightIdx] += forceAonB;
//                                    }
//                                }
//
//
//                                // Add in gravity cause why not
//                                for (int64_t ptIdx = 0; ptIdx < netForcePerPoint.size(); ptIdx++)
//                                {
//                                    netForcePerPoint[ptIdx] += gravityForce * pointMass;
//                                }
//
//                                // Force the three set points to have no force
//                                {
//                                    netForcePerPoint[0] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                                    netForcePerPoint[1] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                                    netForcePerPoint[numSegmentsPerCurve] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
//                                }
//
//                                // Dont update point 0, 1 and M
//                                for (int64_t pointIdx = 2; pointIdx < numSegmentsPerCurve; ++pointIdx)
//                                {
//                                    Farlor::Vector3 acc = netForcePerPoint[pointIdx] * (1.0f / pointMass);
//                                    newPoints[pointIdx] = 2.0f * prevPoints[pointIdx] - oldPoints[pointIdx] + acc * timestep * timestep;
//                                }
//
//                                // Assert points are at the start and end correctly
//                                if (newPoints[0] != m_upInitialCurve->m_basePos)
//                                {
//                                    std::cout << "Path perturb failed as start pos moved" << std::endl;
//                                    std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
//                                }
//
//                                if (newPoints[numSegmentsPerCurve] != m_upInitialCurve->m_targetPos)
//                                {
//                                    std::cout << "Path perturb failed as end pos moved" << std::endl;
//                                    std::cout << "Error on thread and curve idx: " << threadIdx << ", " << currentPathIdx << std::endl;
//                                }
//
//                                assert(newPoints[0] == m_upInitialCurve->m_basePos);
//                                assert(newPoints[numSegmentsPerCurve] == m_upInitialCurve->m_targetPos);
//                            }
//
//                            // Store the older versions of the points
//                            oldPoints = prevPoints;
//                            prevPoints = newPoints;
//                        }
//                        else
//                        {
//                            // Method not implemented
//                            assert(false);
//                        }
//
//                        // Update points from current buffer
//                        for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//                        {
//                            globalPos[CurrentThreadPosStartIdx + pointIdx] = newPoints[pointIdx];
//                        }
//
//                        // Update all tangents
//                        for (int64_t tanIdx = 0; tanIdx < numSegmentsPerCurve; ++tanIdx)
//                        {
//                            Farlor::Vector3 leftPos = globalPos[CurrentThreadPosStartIdx + tanIdx];
//                            Farlor::Vector3 rightPos = globalPos[CurrentThreadPosStartIdx + (tanIdx + 1)];
//
//                            globalTans[CurrentThreadTanStartIdx + tanIdx] = (rightPos - leftPos).Normalized();
//                        }
//
//                        // Update curvature values
//                        for (int64_t curvatureIdx = 0; curvatureIdx < numSegmentsPerCurve; ++curvatureIdx)
//                        {
//                            Farlor::Vector3 leftTan = globalTans[CurrentThreadTanStartIdx + curvatureIdx];
//                            Farlor::Vector3 rightTan = globalTans[CurrentThreadTanStartIdx + (curvatureIdx + 1)];
//
//                            Farlor::Vector3 temp = (rightTan - leftTan) * (1.0f / segmentLength);
//
//                            const float curvature = temp.Magnitude();
//                            globalCurvatures[CurrentThreadCurvatureStartIdx + curvatureIdx] = curvature;
//
//                            // Also, cache the weight of that changed segment
//                            float distance = curvature - minCurvature;
//                            float realIdx = distance / curvatureStepSize;
//                            int64_t leftIdx = floor(realIdx);
//                            int64_t rightIdx = leftIdx + 1;
//
//                            double leftLookup = lookupTable[leftIdx];
//                            double rightLookup = lookupTable[rightIdx];
//
//                            float leftDist = distance - (leftIdx * curvatureStepSize);
//
//                            double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
//                            //std::cout << "Interpolated result: " << interpolatedResult << std::endl;
//                            double interpolatedResultLog = log(interpolatedResult);
//
//                            double segmentWeight = interpolatedResultLog;
//
//#ifdef DelayedAbsorbtionContribution
//                            // Do nothing, we dont add it in here
//#else
//                            // Take natural log of this constant
//                            segmentWeight += lnAbsorbtionConst;
//#endif
//
//                            // Add segment weighting into running path weight
//                            pathWeight += segmentWeight;
//                        }
//                    }
//
//                    if (pathCount < numPathsToSkipPerThread)
//                    {
//                        // Skip
//                    }
//                    else
//                    {
//                        currentMethodCount++;
//                        if (currentMethodCount >= HybridRunCounts[static_cast<int64_t>(currentMethod)])
//                        {
//                            // Reset the current method count
//                            currentMethodCount = 0;
//
//                            // Switch method
//                            currentMethod = static_cast<MethodType>(static_cast<int64_t>(currentMethod) + 1);
//                            if (static_cast<int64_t>(currentMethod) >= static_cast<int64_t>(MethodType::Count))
//                            {
//                                currentMethod = static_cast<MethodType>(0);
//                            }
//
//                            justSwitchedPerturbMethod = true;
//                        }
//
//                        // Else, contribute to the paths
//                        int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
//                        assert(currentPathIdx >= numPathsPerThread * threadIdx);
//
//                        //std::cout << "----- Path: " << currentPathIdx << std::endl;
//                        globalPathWeights[currentPathIdx] = pathWeight;
//
//
//#if defined(ExportPathBatches)
//                        // Add the path to the path batch
//                        for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
//                        {
//                            Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
//                            pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
//                        }
//
//                        numCurvesInBatch++;
//#endif
//
//#if defined(ExportPathBatches)
//                        if (numCurvesInBatch == ExportPathBatchCacheSize)
//                        {
//                            ExportPathBatchesMutex.lock();
//
//                            curvesMetadataFile << threadIdx << " ";
//                            curvesMetadataFile << outputIdx << " ";
//                            curvesMetadataFile << numCurvesInBatch << std::endl;
//
//                            curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//                            numCurvesInBatch = 0;
//                            outputIdx++;
//
//                            ExportPathBatchesMutex.unlock();
//                        }
//#endif
//
//
//                    }
//                }
//
//#if defined(ExportPathBatches)
//                if (numCurvesInBatch > 0)
//                {
//                    ExportPathBatchesMutex.lock();
//
//                    curvesMetadataFile << threadIdx << " ";
//                    curvesMetadataFile << outputIdx << " ";
//                    curvesMetadataFile << numCurvesInBatch << std::endl;
//
//                    curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
//                    numCurvesInBatch = 0;
//                    outputIdx++;
//
//                    ExportPathBatchesMutex.unlock();
//                }
//#endif
//
//            }
//        }
//#ifdef SerialMultithread
//        activeThreadIdx--;
//#endif
//    }

    ExperimentRunner::ExperimentResults FullExperimentRunner::RunExperiment()
    {
        int64_t numFailures = 0;
        int64_t totalFailures = 0;
        int64_t totalSuccess = 0;

        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();

        /* --------------------- */
        auto setupTimeStart = std::chrono::high_resolution_clock::now();

        std::cout << "Got here1" << std::endl;

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
        
        twisty::PerturbUtils::BoundrayConditions boundaryConditions;
        boundaryConditions.arclength = m_upInitialCurve->m_arclength;
        boundaryConditions.m_startPos = m_upInitialCurve->m_basePos;
        boundaryConditions.m_startDir = m_upInitialCurve->m_baseTangent;
        boundaryConditions.m_endPos = m_upInitialCurve->m_targetPos;
        boundaryConditions.m_endDir = m_upInitialCurve->m_targetTangent;
        
        // Constants
        double minCurvature = 0.0;
        double maxCurvature = 0.0;
        twisty::PathWeighting::CalcMinMaxCurvature(minCurvature, maxCurvature, ds);
        const float curvatureStepSize = (maxCurvature - minCurvature) / m_experimentParams.weightingParameters.numCurvatureSteps;

        // This is a horible hack to make sure we always purturb a new curve
        Curve curveToBend = *m_upInitialCurve;

        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        // Create threads and dispatch them
#ifdef SINGLE_THREAD_PERTURB_MODE
        int64_t numPurturbThreads = 1;
#else
#ifdef HardcodedNumPurturbThreads
        int64_t numPurturbThreads = 3;
#else
        int64_t numPurturbThreads = std::thread::hardware_concurrency();
#endif
#endif
        std::cout << "We have " << numPurturbThreads << " avalible for purturbation." << std::endl;

        // Setup rng stuff
        std::vector<std::mt19937_64> perThreadRngGenerators(numPurturbThreads);
        int64_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }
        for (int64_t i = 0; i < numPurturbThreads; ++i)
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
        twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(initialCurvePositions.data(), initialCurveTangents.data(),
            initialCurveCurvatures.data(), m_upInitialCurve->m_numSegments, boundaryConditions);

#if defined(DetailedPurturb)
        {
            std::cout << "Tangents" << std::endl;
            for (int64_t tanIdx = 0; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
            {
                std::cout << "\t" << initialCurveTangents[tanIdx] << std::endl;
            }

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

        std::vector<Farlor::Vector3> perThreadCurvePositions(NumPosPerCurve * numPurturbThreads);
        std::vector<Farlor::Vector3> perThreadCurveTangents(NumTanPerCurve * numPurturbThreads);
        std::vector<float> perThreadCurveCurvatures(NumCurvaturePerCurve * numPurturbThreads);

        std::vector<Farlor::Vector3> perThreadPositionScratch(NumPosPerCurve * numPurturbThreads);
        std::vector<Farlor::Vector3> perThreadTangentScratch(NumTanPerCurve * numPurturbThreads);
        std::vector<float> perThreadCurvatureScratch(NumCurvaturePerCurve * numPurturbThreads);

        for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
        {
            // Copy Pos
            for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
            {
                perThreadCurvePositions[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
                perThreadPositionScratch[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
            }

            // Copy Tan
            for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                perThreadCurveTangents[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
                perThreadTangentScratch[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
            }

            // Copy Curvatures
            for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
            {
                perThreadCurveCurvatures[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
                perThreadCurvatureScratch[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
            }
        }

        std::vector<double> cachedSegmentWeights(m_experimentParams.numSegmentsPerCurve * numPurturbThreads);
        std::vector<double> compressedWeightBuffer(ExportPathBatchCacheSize);

        const std::string fnFilename = "SavedFN.fnd";
        const std::filesystem::path fnFilePath = std::filesystem::current_path() / fnFilename;
        PathWeighting::NormalizerStuff::FN* pFN = nullptr;

        // If we can load the fn data, load it
        if (std::filesystem::exists(fnFilePath))
        {
            std::cout << "Using cached fd file at: " << fnFilePath << std::endl;
            std::ifstream inFile(fnFilePath);
            pFN = new PathWeighting::NormalizerStuff::FN(inFile);
            inFile.close();
        }
        // We need to generate it this time and save it off to use next time
        else
        {
            // This is the max M value
            const int maxorder = m_upInitialCurve->m_numSegments;

            // Generate the fn data table
            const int numZSamples = 5000;
            const int numIntegrationSamples = 5000;

            // Arbitrarily set min and max |r_vec| values.
            // Why this specific max bound, I do not know
            const double rMin = 0.0;
            const double rMax = 200.0;
            pFN = new PathWeighting::NormalizerStuff::FN(numZSamples, numIntegrationSamples, maxorder, rMin, rMax);

            std::ofstream outFile(fnFilePath);
            pFN->WriteToFile(outFile);
            outFile.close();
        }
        PathWeighting::NormalizerStuff::FN& fn = *pFN;

        Farlor::Vector3 Z = (boundaryConditions.m_endPos - boundaryConditions.m_startPos) * (m_upInitialCurve->m_numSegments + 2) / boundaryConditions.arclength
            - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
        std::cout << "Z: " << Z << std::endl;
        std::cout << "|Z|: " << Z.Magnitude() << std::endl;

        PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = PathWeighting::NormalizerStuff::Norm(fn, m_upInitialCurve->m_numSegments,
            Z.Magnitude(), boundaryConditions.arclength);


        //const boost::multiprecision::cpp_dec_float_100 singleSegmentNormalizer = 2.0 * TwistyPi * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(0.625));
        //boost::multiprecision::cpp_dec_float_100 segmentNormalizer = 1.0;
        //for (int64_t segIdx = 0; segIdx < (m_experimentParams.numSegmentsPerCurve - 1); ++segIdx)
        //{
        //    segmentNormalizer = segmentNormalizer * singleSegmentNormalizer;
        //}

        //boost::multiprecision::cpp_dec_float_100 pathNormalizer = 1.0;
        //pathNormalizer = pathNormalizer * boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(static_cast<float>(m_experimentParams.numSegmentsPerCurve)
        //    / m_upInitialCurve->m_arclength), 3.0);
        //pathNormalizer = pathNormalizer * segmentNormalizer;
        //pathNormalizer = pathNormalizer * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(-0.325));

        const boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10 = boost::multiprecision::log10(pathNormalizer);

        std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
        std::cout << "PathNormalizerLog10: " << pathNormalizerLog10 << std::endl;
        //exit(1);

        auto setupTimeEnd = std::chrono::high_resolution_clock::now();
        /* --------------------- */


        /* --------------------- */
        const int64_t numDispatches = (m_experimentParams.numPathsInExperiment + ExportPathBatchCacheSize - 1) / ExportPathBatchCacheSize;
        std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
        std::cout << "numPathsPerBatch: " << ExportPathBatchCacheSize << std::endl;
        std::cout << "Num dispatches required: " << numDispatches << std::endl;
        int64_t numPathsLeft = m_experimentParams.numPathsInExperiment;
        int64_t numPathsGenerated = 0;

        // We need to calculate the absorbtion/scattering piece
        boost::multiprecision::cpp_dec_float_100 bigTotalExperimentWeight = 0.0;

        long long perturbTimeCount = 0;
        long long weightCalcTimeCount = 0;

        // We need a number of dispatches
        boost::multiprecision::cpp_dec_float_100 minimumPathWeight = 0.0;
        boost::multiprecision::cpp_dec_float_100 maximumPathWeight = 0.0;

        for (int64_t dispatchIdx = 0; dispatchIdx < numDispatches; ++dispatchIdx)
        {
            auto dispatchTimeStart = std::chrono::high_resolution_clock::now();

            auto perturbTimeStart = std::chrono::high_resolution_clock::now();

            int64_t pathsInDispatch = std::min(ExportPathBatchCacheSize, numPathsLeft);
            std::cout << "Paths in dispatch " << dispatchIdx << ": " << pathsInDispatch << std::endl;
            {
                int64_t numPathsPerThread = (pathsInDispatch + numPurturbThreads - 1) / numPurturbThreads;
                std::vector<std::thread> threads(numPurturbThreads);
                for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
                {
                    std::thread newThread(&FullExperimentRunner::GeometryPerturb, this,
                        threadIdx,
                        pathsInDispatch,
                        numPathsPerThread,
                        m_experimentParams.numPathsToSkip,
                        m_experimentParams.numSegmentsPerCurve,
                        std::ref(perThreadRngGenerators),
                        std::ref(perThreadCurvePositions),
                        std::ref(perThreadCurveTangents),
                        std::ref(perThreadCurveCurvatures),
                        std::ref(perThreadPositionScratch),
                        std::ref(perThreadTangentScratch),
                        std::ref(perThreadCurvatureScratch),
                        std::ref(compressedWeightBuffer),
                        std::ref(cachedSegmentWeights),
                        m_upInitialCurve->m_segmentLength,
                        lookupEvaluator,
                        boundaryConditions,
                        fn
                    );

                    threads[threadIdx] = std::move(newThread);
                }

                for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
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

            std::vector<boost::multiprecision::cpp_dec_float_100> bigFloatWeightsLog10(pathsInDispatch);
            int64_t numWeightingThreads = std::thread::hardware_concurrency();

            boost::multiprecision::cpp_dec_float_100 totalDispatchWeight = 0.0;
            int64_t numWeightsPerThread = (pathsInDispatch + numWeightingThreads - 1) / numWeightingThreads;
            {
                std::vector<std::thread> threads(numWeightingThreads);
                std::vector<boost::multiprecision::cpp_dec_float_100> threadWeights(numWeightingThreads);
                for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    threadWeights[threadIdx] = 0.0;
                    std::thread newThread(&FullExperimentRunner::WeightCombineThreadKernel, this, threadIdx, pathsInDispatch, numWeightsPerThread, m_upInitialCurve->m_arclength,
                        m_upInitialCurve->m_numSegments, std::ref(compressedWeightBuffer), std::ref(bigFloatWeightsLog10), std::ref(threadWeights[threadIdx]),
                        pathNormalizer);
                    threads[threadIdx] = std::move(newThread);
                }

                for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    if (threads[threadIdx].joinable())
                    {
                        threads[threadIdx].join();
                    }
                }

#if defined(DetailedPurturb)
                {
                    std::cout << "Thread weights: " << std::endl;
                    for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                    {
                        std::cout << threadWeights[threadIdx] << std::endl;
                    }
                }
#endif

                for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
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
            std::filesystem::path experimentDirectoryPath = m_experimentParams.experimentDirPath;
            std::string bigfloatOutputFile = "BigFloatWeights.txt";
            experimentDirectoryPath.append(bigfloatOutputFile);

            std::cout << "Output: " << dispatchIdx << " : " << bigFloatWeightsLog10.size() << std::endl;

            std::ofstream bigfloatOFS;
            if (dispatchIdx == 0)
            {
                bigfloatOFS.open(experimentDirectoryPath.c_str());
                bigfloatOFS << m_experimentParams.numPathsInExperiment << std::endl;
            }
            else
            {
                bigfloatOFS.open(experimentDirectoryPath.c_str(), std::ios::app);
            }
            
            for (int64_t i = 0; i < bigFloatWeightsLog10.size(); ++i)
            {
                // We have to add in the path normalizer here as it wasnt acounted for anywhere else before this for these specific saved off weights
                bigfloatOFS << (bigFloatWeightsLog10[i] + pathNormalizerLog10) << std::endl;
            }
#endif
            numPathsLeft -= pathsInDispatch;
            numPathsGenerated += pathsInDispatch;

            auto dispatchTimeEnd = std::chrono::high_resolution_clock::now();
            auto dispatchRunTime = std::chrono::duration_cast<std::chrono::milliseconds>(dispatchTimeEnd - dispatchTimeStart);
            std::cout << "\tDispatch " << dispatchIdx  << " Time: " << dispatchRunTime.count() << "ms" << std::endl;
        }

        std::cout << "Minimum Weight: " << minimumPathWeight << std::endl;
        std::cout << "Maximum Weight: " << maximumPathWeight << std::endl;
        std::cout << "Big total weight before: " << bigTotalExperimentWeight << std::endl;

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
        std::vector< boost::multiprecision::cpp_dec_float_100> finalValuesVec;
        finalValuesVec.push_back(bigTotalExperimentWeight);
        results.experimentWeights = finalValuesVec;
        results.totalPathsGenerated = numPathsGenerated;
        results.numFailedPaths = 0;
        return results;
    }

    void FullExperimentRunner::WeightCombineThreadKernel(const int64_t threadIdx, int64_t numWeights, int64_t numWeightsPerThread, float arclength, int64_t numSegmentsPerCurve,
        const std::vector<double>& compressedWeights, std::vector<boost::multiprecision::cpp_dec_float_100>& bigFloatWeightsLog10,
        boost::multiprecision::cpp_dec_float_100& threadWeight, boost::multiprecision::cpp_dec_float_100 pathNormalizer)
    {

        for (int64_t i = 0; i < numWeightsPerThread; i++)
        {
            int64_t idx = threadIdx * numWeightsPerThread + i;
            if (idx >= numWeights)
            {
                break;
            }

            const boost::multiprecision::cpp_dec_float_100 bigfloatCompressed = compressedWeights[idx];
            const boost::multiprecision::cpp_dec_float_100 decompressed = boost::multiprecision::pow(10.0, bigfloatCompressed);
            const boost::multiprecision::cpp_dec_float_100 pathWeight = decompressed * pathNormalizer;
            
            
            // Pulled from Jerry analysis
            bigFloatWeightsLog10[idx] = bigfloatCompressed;
            threadWeight += pathWeight;
        }
    }

    void FullExperimentRunner::Shutdown()
    {
    }
}