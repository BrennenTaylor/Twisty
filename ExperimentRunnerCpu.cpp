#include "ExperimentRunnerCpu.h"

#include "CurveUtils.h"
#include "MathConsts.h"

#include <omp.h>

#include <std::assert.h>
#include <ctime>
#include <fstream>
#include <limits>

#include <chrono>

#define TWISTY_CPU_PARALLEL 0
#define TWISTY_CPU_SERIAL_PURTURB 0

constexpr bool DetailedPurturb = false;

namespace twisty
{
    ExperimentRunnerCpu::ExperimentRunnerCpu(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper)
        : ExperimentRunner(experimentParams, bootstrapper)
        , m_rng()
    {
        uint32_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }
        std::cout << "Purturb seed used: " << seed << std::endl;
        m_rng = std::mt19937(seed);
    }

    ExperimentRunnerCpu::~ExperimentRunnerCpu()
    {
    }

    bool ExperimentRunnerCpu::Setup()
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

            // Once we have a curve, we know arclength.
            // Thus, we can setup the min and max curvatures
            float ds = m_upInitialCurve->m_arclength / m_upInitialCurve->m_numSegments;

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

    ExperimentRunner::ExperimentResults ExperimentRunnerCpu::RunExperiment()
    {
        uint32_t numFailures = 0;
        uint32_t totalFailures = 0;
        uint32_t totalSuccess = 0;

        // We do a single path batch on the CPU
        uint32_t numberOfDataSegments = m_experimentParams.numPathsInExperiment* m_experimentParams.numSegmentsPerCurve;

        //KTSegments initialSegments(m_experimentParams.numSegmentsPerCurve);
        std::vector<float> m_curvatures(m_experimentParams.numSegmentsPerCurve);
        std::vector<Farlor::Vector3> m_positions(m_experimentParams.numSegmentsPerCurve);
        std::vector<Farlor::Vector3> m_tangents(m_experimentParams.numSegmentsPerCurve);

        // Intitialize curvature and position values for this path batch
        m_curvatures = m_upInitialCurve->m_curvatures;
        m_positions = m_upInitialCurve->m_positions;
        m_tangents = m_upInitialCurve->m_tangents;


        // Say that we will start outputing the path batch output
        BeginPathBatchOutput();

        // Parallel default initialize all values to be the default curve
        uint32_t numberOfPathBatches = (m_experimentParams.numPathsInExperiment + m_experimentParams.exportPathBatchSize - 1) / m_experimentParams.exportPathBatchSize;

        // This is a horible hack to make sure we always purturb a new curve
        Curve curveToBend = *m_upInitialCurve;

        std::vector<long long> pathBatchTimes(numberOfPathBatches);

        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        for (uint32_t pathBatchIdx = 0; pathBatchIdx < numberOfPathBatches; pathBatchIdx++)
        {
            auto pathBatchTimeStart = std::chrono::high_resolution_clock::now();

            // Initialize the path batch data structure
            PathBatch pathBatch;
            pathBatch.index = pathBatchIdx;
            pathBatch.numberOfPathsInBatch = std::min(m_experimentParams.numPathsInExperiment - (pathBatchIdx * m_experimentParams.exportPathBatchSize), m_experimentParams.exportPathBatchSize);
            pathBatch.m_curvatures = std::vector<float>(pathBatch.numberOfPathsInBatch * m_experimentParams.numSegmentsPerCurve);
            pathBatch.m_positions = std::vector<Farlor::Vector3>(pathBatch.numberOfPathsInBatch * m_experimentParams.numSegmentsPerCurve);
            pathBatch.m_tangents = std::vector<Farlor::Vector3>(pathBatch.numberOfPathsInBatch * m_experimentParams.numSegmentsPerCurve);
            pathBatch.perPathVailidity = std::vector<bool>(pathBatch.numberOfPathsInBatch);

            // Copy over the initial torsion and curvatures for this path batch
//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp parallel for shared(pathBatch)
//#endif
            for (int32_t pathIdx = 0; pathIdx < pathBatch.numberOfPathsInBatch; pathIdx++)
            {
                pathBatch.perPathVailidity[pathIdx] = false;
                memcpy(&pathBatch.m_curvatures[pathIdx * m_experimentParams.numSegmentsPerCurve], &m_curvatures, sizeof(float) * m_experimentParams.numSegmentsPerCurve);
                memcpy(&pathBatch.m_positions[pathIdx * m_experimentParams.numSegmentsPerCurve], &m_positions, sizeof(Farlor::Vector3) * m_experimentParams.numSegmentsPerCurve);
                memcpy(&pathBatch.m_tangents[pathIdx * m_experimentParams.numSegmentsPerCurve], &m_tangents, sizeof(Farlor::Vector3) * m_experimentParams.numSegmentsPerCurve);
            }

            //Curve initialCurve = *m_upInitialCurve;
            uint32_t numPathsAttempted = 0;
            uint32_t numSuccessfulPaths = 0;

            uint32_t numRootSolveFailures = 0;
            uint32_t numCurveErrorFailures = 0;

            //Curve previousCurve = initialCurve;
//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp parallel for shared(initialCurve, previousCurve, pathBatch, numPathsAttempted, numSuccessfulPaths, numRootSolveFailures, numCurveErrorFailures)
//#endif
            // Each path is generated on its own thread...?
            for (int32_t pathIdx = 0; pathIdx < pathBatch.numberOfPathsInBatch; pathIdx++)
            {
                Curve seedCurve(curveToBend);

//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp critical
//#endif
                {
                    seedCurve = curveToBend;
                }

                const uint32_t numAllowedFailures = 300;
                std::unique_ptr<Curve> upNewCurve = nullptr;
                uint32_t numCurrentFocusFailures = 0;
                while (!upNewCurve && numCurrentFocusFailures < numAllowedFailures)
                {
                    uint32_t flag = 0;
//#ifdef TWISTY_CPU_PARALLEL
//#ifdef TWISTY_CPU_SERIAL_PURTURB
//#pragma omp critical
//#endif
//#endif
                    {
                        upNewCurve = PurturbCurve(curveToBend, flag);
                    }

//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp critical
//#endif
                    {
                        numPathsAttempted++;
                    }

                    if (!upNewCurve)
                    {
                        if (flag == 1)
                        {
//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp critical
//#endif
                            {
                                numCurrentFocusFailures++;
                                numRootSolveFailures++;
                            }
                        }

                        if (flag == 2)
                        {
//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp critical
//#endif
                            {
                                numCurrentFocusFailures++;
                                numCurveErrorFailures++;
                            }
                        }
                    }
                }

                if (upNewCurve)
                {
                    pathBatch.perPathVailidity[pathIdx] = true;
                    for (uint32_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
                    {
                        int32_t index = pathIdx * m_experimentParams.numSegmentsPerCurve + segmentIdx;
                        pathBatch.m_curvatures[index] = upNewCurve->m_curvatures[segmentIdx];
                        pathBatch.m_positions[index] = upNewCurve->m_positions[segmentIdx];
                        pathBatch.m_tangents[index] = upNewCurve->m_tangents[segmentIdx];
                    }

                    // Quick and dirty, just bootstrap the next paths with previous paths
//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp critical
//#endif
                    {
                        curveToBend = *upNewCurve;
                    }
                }
                else
                {
                    std::cout << "Failed to generate this path" << std::endl;
                }
//#ifdef TWISTY_CPU_PARALLEL
//#pragma omp critical
//#endif
                {
                    numSuccessfulPaths++;
                }
            }

            auto pathBatchTimeEnd = std::chrono::high_resolution_clock::now();

            auto pathBatchTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(pathBatchTimeEnd - pathBatchTimeStart);
            pathBatchTimes[pathBatchIdx] = pathBatchTimeMs.count();

            std::cout << "Num paths attempted: " << numPathsAttempted << std::endl;
            std::cout << "Num successful paths: " << numSuccessfulPaths << std::endl;
            std::cout << "Num root solve failures: " << numRootSolveFailures << std::endl;
            std::cout << "Num curve error failures: " << numCurveErrorFailures << std::endl;

            std::cout << "% Successful: " << static_cast<float>(numSuccessfulPaths) / static_cast<float>(numPathsAttempted) << std::endl;
            std::cout << "% Root Solve Fail: " << static_cast<float>(numRootSolveFailures) / static_cast<float>(numPathsAttempted) << std::endl;
            std::cout << "% Curve Error Failures: " << static_cast<float>(numCurveErrorFailures) / static_cast<float>(numPathsAttempted) << std::endl;



            OutputPathBatch(pathBatch);
        }

        EndPathBatchOutput();


        std::cout << "Experiment Time Reporting: " << std::endl;

        auto experimentTimeEnd = std::chrono::high_resolution_clock::now();

        auto experimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(experimentTimeEnd - experimentTimeStart);
        std::cout << "\tTotal Experiment Time: " << experimentTimeMs.count() << std::endl;

        unsigned long long totalCount = 0;
        for (uint32_t i = 0; i < pathBatchTimes.size(); i++)
        {
            std::cout << "\tPath Batch " << i << " Time: " << pathBatchTimes[i] << std::endl;
            totalCount += pathBatchTimes[i];
        }
        std::cout << "\tTotal Path Batch Time: " << totalCount << std::endl;

        std::cout << "Paths Generated!!" << std::endl;

        return {};
    }

    void ExperimentRunnerCpu::Shutdown()
    {
    }

    std::unique_ptr<Curve> ExperimentRunnerCpu::PurturbCurve(const Curve& curve, uint32_t& flag)
    {
        if (DetailedPurturb)
        {
            std::cout << "Performing purturb" << std::endl;
        }

        // Actually do the purturbation
        std::unique_ptr<Curve> upNewCurve = SimpleGeometryCurvePerturb(curve, flag);
        return upNewCurve;
    }

    Farlor::Matrix3x3  RotationMatrixAroundAxis(float angle, Farlor::Vector3 axis)
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

    std::pair<float, float> CurvatureAndTorsionBetweenTwoFrames(const Farlor::Matrix3x3& startFrame, const Farlor::Matrix3x3& endFrame, float segmentLength)
    {
        std::pair<float, float> curvatureAndTorsion = { 0.0f, 0.0f };
        {
            float curvature = ((endFrame.m_rows[0] - startFrame.m_rows[0]) * (1.0f / segmentLength)).Magnitude();
            curvatureAndTorsion.first = curvature;
        }

        {
            auto torsionLeft = -1.0f * startFrame.m_rows[1];
            auto torsionRight = (endFrame.m_rows[2] - startFrame.m_rows[2]) * (1.0f / segmentLength);
            float torsion = torsionLeft.Dot(torsionRight);
            curvatureAndTorsion.second = torsion;
        }
        return curvatureAndTorsion;
    }

    std::unique_ptr<Curve> ExperimentRunnerCpu::SimpleGeometryCurvePerturb(const Curve& curve, uint32_t& flag)
    {
        if (DetailedPurturb)
        {
            std::cout << "Begin Purturb --------" << std::endl;
        }

        std::unique_ptr<Curve> upNewCurve = std::make_unique<Curve>(curve);

        // We bound on left by one as we dont want to rotate the first segment at all
        // Left bound by m-2 as we at least want there to be one point between the left and right points selected so an actual perturbation occurs
        std::uniform_int_distribution<int> leftPointIndexUniformDist(1, upNewCurve->m_numSegments - 3); // uniform, unbiased
        int32_t leftPointIndex = leftPointIndexUniformDist(m_rng);
        std::uniform_int_distribution<int> rightPointIndexUniformDist(leftPointIndex + 2, upNewCurve->m_numSegments - 1); // uniform, unbiased
        int32_t rightPointIndex = rightPointIndexUniformDist(m_rng);
/*
        leftPointIndex = 1;
        rightPointIndex = 9;*/

        std::assert(leftPointIndex < rightPointIndex);
        std::assert((rightPointIndex - leftPointIndex) >= 2);

        if (DetailedPurturb)
        {
            std::cout << "\tLeft Index: " << leftPointIndex << std::endl;
            std::cout << "\tRight Index: " << rightPointIndex << std::endl;
        }

        // 0 - 2 PI uniform distribution
        //std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * 0.01f, TwistyPi * 0.01f);
        //std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * 0.1f, TwistyPi * 0.1f);
        std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi, TwistyPi);

        // End targets of purturbation
        Farlor::Vector3 targetN = m_bootstrapper.GetTargetNormal();
        Farlor::Vector3 targetP = m_bootstrapper.GetTargetPosition();


        /** This is where the fun begins **/
        std::vector<Farlor::Vector3> points;
        // All but the last point
        upNewCurve->ReconstructCurvePositionsFirstOrder(points);

        // We need two frames for each segment to get the new curvature and torsion.
        // we need the frame left of the segment, as well as the frame right of the segment.

        // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
        Farlor::Vector3 leftPoint = points[leftPointIndex];
        Farlor::Vector3 rightPoint = points[rightPointIndex];
        Farlor::Vector3 axisOfRotation = (rightPoint - leftPoint).Normalized();

        float randomAngle = zeroToTwoPiUniformDist(m_rng);
        //randomAngle = 0.0f;
        Farlor::Matrix3x3 rotationMatrix = RotationMatrixAroundAxis(randomAngle, axisOfRotation);

        if (DetailedPurturb)
        {
            std::cout << "\tRotation Info: " << std::endl;
            std::cout << "\t\tRandom angle: " << randomAngle << std::endl;
            std::cout << "\t\tRotationMatrix: " << rotationMatrix << std::endl;
        }

        // Lets build up the new poly line.
        std::vector<Farlor::Vector3> updatedPolyline;

        // First, lets add in the points before the rotation. These experience no rotation.
        for (uint32_t pointIdx = 0; pointIdx <= leftPointIndex; pointIdx++)
        {
            updatedPolyline.push_back(points[pointIdx]);
        }

        // Now, we do the rotated points
        for (uint32_t pointIdx = leftPointIndex + 1; pointIdx < rightPointIndex; pointIdx++)
        {
            Farlor::Vector3 pointToRotate = points[pointIdx];
            Farlor::Vector3 shiftedPoint = pointToRotate - leftPoint;
            Farlor::Vector3 rotatedPoint = rotationMatrix * shiftedPoint;
            Farlor::Vector3 finalPoint = rotatedPoint + leftPoint;
            updatedPolyline.push_back(finalPoint);
        }

        // Finally, we get those at the end after the rotation occures
        for (uint32_t pointIdx = rightPointIndex; pointIdx < points.size(); pointIdx++)
        {
            updatedPolyline.push_back(points[pointIdx]);
        }
        updatedPolyline.push_back(m_upInitialCurve->m_targetPos);

        std::assert(points.size() + 1 == updatedPolyline.size());

        // Now that we have the polyline, we want to construct the reference frames.
        // Now, we build up reference frames.
        std::vector<Farlor::Vector3> tangents;

        // For now, simply compute the difference in positions.
        // We can do a different approach later.
        for (uint32_t posIdx = 0; posIdx < updatedPolyline.size() - 1; ++posIdx)
        {
            Farlor::Vector3 tangent = updatedPolyline[posIdx + 1] - updatedPolyline[posIdx];
            tangent = tangent.Normalized();
            tangents.push_back(tangent);
        }

        // End Frame
        // We only really need the tangent for it
        {
            tangents.push_back(m_bootstrapper.GetTargetNormal());
        }

        std::assert(tangents.size() == upNewCurve->m_numSegments + 1);

        for (uint32_t i = 0; i < upNewCurve->m_numSegments; ++i)
        {
            float curvature = ((tangents[i+1] - tangents[i]) * (1.0f / upNewCurve->m_segmentLength)).Magnitude();
            upNewCurve->m_curvatures[i] = curvature;
            upNewCurve->m_positions[i] = updatedPolyline[i];
            upNewCurve->m_tangents[i] = tangents[i];
        }

        return upNewCurve;
    }

    std::unique_ptr<Curve> ExperimentRunnerCpu::ComplexGeometryCurvePerturb(const Curve& curve, uint32_t& flag)
    {
        std::assert(false);
        return nullptr;
    }
}