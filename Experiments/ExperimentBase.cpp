#include "ExperimentBase.h"

#include <omp.h>
#include <random>

namespace twisty {
namespace ExperimentBase {
    FiveSegmentAngleIntegrationResult FiveSegmentAngleIntegration(const uint32_t numPhi1Vals,
          const uint32_t numTheta1Vals, const uint32_t numTheta2Vals,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
    {
        const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

        const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
        const Farlor::Vector3 point1
              = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

        const Farlor::Vector3 point5 = experimentGeometry.m_endPos;
        const Farlor::Vector3 point4
              = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;

        // Polar angle
        const float phi1Min = 0.0f;
        const float phi1Max = 1.0f;
        const float dPhi1 = (phi1Max - phi1Min) / numPhi1Vals;

        // Azimuthal
        const float theta1Min = -twisty::TwistyPi;
        const float theta1Max = twisty::TwistyPi;
        const float dTheta1 = (theta1Max - theta1Min) / numTheta1Vals;

        // Azimuthal
        const float theta2Min = -twisty::TwistyPi;
        const float theta2Max = twisty::TwistyPi;
        const float dTheta2 = (theta2Max - theta2Min) / numTheta2Vals;

        const uint64_t numTotalPaths = static_cast<uint64_t>(numPhi1Vals)
              * static_cast<uint64_t>(numTheta1Vals) * static_cast<uint64_t>(numTheta2Vals);

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

#pragma omp parallel for num_threads(maxThreads) default(none)                                     \
      shared(combinedWeightValues, minPathWeightPerThread, maxPathWeightPerThread)
        for (int phi1Idx = 0; phi1Idx < numPhi1Vals; phi1Idx++) {
            const int threadId = omp_get_thread_num();

            const float phi1Mapped = phi1Min + phi1Idx * dPhi1;
            const float phi1 = std::acos(1.0 - 2.0 * phi1Mapped);

            for (int theta1Idx = 0; theta1Idx < numTheta1Vals; theta1Idx++) {
                const float theta1 = theta1Min + theta1Idx * dTheta1;

                /*
                  x = ρsinφcosθ
                  y = ρsinφsinθ
                  z = ρcosφ 
            */

                const float sinPhi1 = std::sin(phi1);
                const float cosPhi1 = std::cos(phi1);
                const float sinTheta1 = std::sin(theta1);
                const float cosTheta1 = std::cos(theta1);

                // Calculate the first segment position
                const Farlor::Vector3 segment1Dir
                      = Farlor::Vector3(sinPhi1 * cosTheta1, sinPhi1 * sinTheta1, cosPhi1);
                const Farlor::Vector3 point2 = point1 + segment1Dir * ds;

                const float remainingDistance2 = (point4 - point2).SqrMagnitude();

                if ((4 * ds * ds) < remainingDistance2) {
                    continue;
                }

                // If not, we keep going through the possible combinations
                for (int theta2Idx = 0; theta2Idx < numTheta2Vals; theta2Idx++) {
                    const float theta2 = theta2Min + theta2Idx * dTheta2;

                    const Farlor::Vector3 x_p = (point2 + point4) * 0.5;
                    const Farlor::Vector3 lineUnitDir = (point4 - point2).Normalized();

                    Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
                    if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
                        otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
                    }

                    const Farlor::Vector3 normalToLine
                          = lineUnitDir.Cross(otherCrossVec).Normalized();

                    // We should have an even number of segments remaining
                    const float hypot = ds;
                    const float D_2 = (point4 - point2).Magnitude() * 0.5f;
                    assert(D_2 < hypot && "This should never be reached due to earlier check.");

                    const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
                    Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

                    // Now rotate randomly theta amount around the axis.
                    {
                        const float sinRotAngle = std::sinf(theta2 / 2.0f);
                        float quaternionRotation[4]
                              = { std::cosf(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
                                    lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


                        Farlor::Vector3 shiftedPoint = x_t - point2;
                        // Rotate and stuff back in shifted point
                        twisty::RotateVectorByQuaternion(
                              quaternionRotation, shiftedPoint.m_data.data());
                        // Update the point with the rotated version
                        x_t = shiftedPoint + point2;
                    }
                    const Farlor::Vector3 point3 = x_t;

                    std::array<Farlor::Vector3, 6> points = { experimentGeometry.m_startPos, point1,
                        point2, point3, point4, experimentGeometry.m_endPos };
                    std::array<Farlor::Vector3, 5> tangents;
                    std::array<float, 4> curvatures;

                    twisty::PerturbUtils::UpdateTangentsFromPos(
                          points.data(), tangents.data(), 5, experimentGeometry);
                    twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                          tangents.data(), curvatures.data(), 5, experimentGeometry);

                    const double scatteringWeightLog10
                          = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                                  curvatures.data(), 4, weightLookupTable)
                          + pathNormalizerLog10;

                    // Update the min and max values
                    if (scatteringWeightLog10 < minPathWeightPerThread[threadId]) {
                        minPathWeightPerThread[threadId] = scatteringWeightLog10;
                    }
                    if (scatteringWeightLog10 > maxPathWeightPerThread[threadId]) {
                        maxPathWeightPerThread[threadId] = scatteringWeightLog10;
                    }

                    twisty::CombinedWeightValues_C &activeWeightValue
                          = combinedWeightValuesPerThread[threadId];

                    if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                        twisty::CombinedWeightValues_C_AddValue(
                              activeWeightValue, scatteringWeightLog10);
                    } else {
#pragma omp critical
                        {
                            combinedWeightValues.push_back(activeWeightValue);
                        }
                        twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                        twisty::CombinedWeightValues_C_AddValue(
                              activeWeightValue, scatteringWeightLog10);
                    }
                }
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
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }


    FiveSegmentAngleIntegrationResult FiveSegmentAngleSpaceMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
    {
        const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

        const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
        const Farlor::Vector3 point1
              = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

        const Farlor::Vector3 point5 = experimentGeometry.m_endPos;
        const Farlor::Vector3 point4
              = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;

        // Polar angle
        const float phiMin = 0.0f;
        const float phiMax = 1.0f;

        // Azimuthal
        const float thetaMin = -twisty::TwistyPi;
        const float thetaMax = twisty::TwistyPi;


        const uint64_t numTotalPaths = numExperimentPaths;

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

        // Random Gen per thread
        std::vector<std::mt19937_64> randomGenPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            randomGenPerThread[i].seed(i);
        }

#pragma omp parallel for num_threads(maxThreads) default(none) shared(                             \
      randomGenPerThread, combinedWeightValues, minPathWeightPerThread, maxPathWeightPerThread)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::mt19937_64 &randomGen = randomGenPerThread[threadId];

            std::uniform_real_distribution<float> phi1Dist(phiMin, phiMax);
            std::uniform_real_distribution<float> thetaDist(thetaMin, thetaMax);

            const float phi1Mapped = phi1Dist(randomGen);
            const float phi1 = std::acos(1.0 - 2.0 * phi1Mapped);

            const float theta1 = thetaDist(randomGen);
            /*
                  x = ρsinφcosθ
                  y = ρsinφsinθ
                  z = ρcosφ 
            */

            const float sinPhi1 = std::sin(phi1);
            const float cosPhi1 = std::cos(phi1);
            const float sinTheta1 = std::sin(theta1);
            const float cosTheta1 = std::cos(theta1);

            // Calculate the first segment position
            const Farlor::Vector3 segment1Dir
                  = Farlor::Vector3(sinPhi1 * cosTheta1, sinPhi1 * sinTheta1, cosPhi1);
            const Farlor::Vector3 point2 = point1 + segment1Dir * ds;

            const float remainingDistance2 = (point4 - point2).SqrMagnitude();

            if ((4 * ds * ds) < remainingDistance2) {
                continue;
            }

            const float theta2 = thetaDist(randomGen);

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
                float quaternionRotation[4]
                      = { std::cosf(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
                            lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


                Farlor::Vector3 shiftedPoint = x_t - point2;
                // Rotate and stuff back in shifted point
                twisty::RotateVectorByQuaternion(quaternionRotation, shiftedPoint.m_data.data());
                // Update the point with the rotated version
                x_t = shiftedPoint + point2;
            }
            const Farlor::Vector3 point3 = x_t;

            std::array<Farlor::Vector3, 6> points = { experimentGeometry.m_startPos, point1, point2,
                point3, point4, experimentGeometry.m_endPos };
            std::array<Farlor::Vector3, 5> tangents;
            std::array<float, 4> curvatures;

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), 5, experimentGeometry);
            twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                  tangents.data(), curvatures.data(), 5, experimentGeometry);

            const double scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10(
                          curvatures.data(), 4, weightLookupTable)
                  + pathNormalizerLog10;

            // Update the min and max values
            if (scatteringWeightLog10 < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10;
            }
            if (scatteringWeightLog10 > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(activeWeightValue, scatteringWeightLog10);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(activeWeightValue, scatteringWeightLog10);
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
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }
}
}