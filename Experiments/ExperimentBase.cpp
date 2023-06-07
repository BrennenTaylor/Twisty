#include "ExperimentBase.h"
#include "PathWeighters.h"

#include <omp.h>
#include <random>


namespace twisty {
namespace ExperimentBase {
    Result FiveSegmentAngleIntegration(const uint32_t numPhi1Vals, const uint32_t numTheta1Vals,
          const uint32_t numTheta2Vals,
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
        //        std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

#pragma omp parallel for num_threads(maxThreads) default(none) shared(combinedWeightValues,        \
            minPathWeightPerThread, maxPathWeightPerThread, numPhi1Vals, phi1Min, phi1Max, dPhi1,  \
            numTheta1Vals, dTheta1, theta1Min, theta1Max, numTheta2Vals, dTheta2, theta2Min,       \
            theta2Max, ds, point0, point1, point4, point5, experimentGeometry, experimentParams,   \
            pathNormalizerLog10, weightLookupTable, combinedWeightValuesPerThread)
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
                        const float sinRotAngle = std::sin(theta2 / 2.0f);
                        float quaternionRotation[4]
                              = { std::cos(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
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

                    if (experimentParams.weightingParameters.weightingMethod
                          == twisty::WeightingMethod::RadiativeTransfer) {
                        twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                              tangents.data(), curvatures.data(), 5, experimentGeometry);
                    } else {
                        twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
                              tangents.data(), curvatures.data(), 5, experimentGeometry);
                    }

                    twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                          = twisty::PathWeighting::WeightCurveViaCurvatureLog10(curvatures.data(),
                                4, weightLookupTable,
                                experimentParams.weightingParameters.absorption);

                    if (!scatteringWeightLog10.isValid)
                        continue;
                    scatteringWeightLog10.weight += pathNormalizerLog10;

                    // Update the min and max values
                    if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                        minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
                    }
                    if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                        maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
                    }

                    twisty::CombinedWeightValues_C &activeWeightValue
                          = combinedWeightValuesPerThread[threadId];

                    if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                        twisty::CombinedWeightValues_C_AddValue(
                              activeWeightValue, scatteringWeightLog10.weight);
                    } else {
#pragma omp critical
                        {
                            combinedWeightValues.push_back(activeWeightValue);
                        }
                        twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                        twisty::CombinedWeightValues_C_AddValue(
                              activeWeightValue, scatteringWeightLog10.weight);
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


    Result FiveSegmentAngleSpaceMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
    {
        const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

        const Farlor::Vector3 point1
              = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

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
        //std::cout << "Max threads: " << maxThreads << '\n';
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

#pragma omp parallel for num_threads(maxThreads) default(none)                                     \
      shared(randomGenPerThread, combinedWeightValues, minPathWeightPerThread,                     \
                  maxPathWeightPerThread, numExperimentPaths, phiMin, phiMax, thetaMin, thetaMax,  \
                  point1, point4, ds, experimentParams, experimentGeometry, pathNormalizerLog10,   \
                  combinedWeightValuesPerThread, weightLookupTable)
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
                const float sinRotAngle = std::sin(theta2 / 2.0f);
                float quaternionRotation[4]
                      = { std::cos(theta2 / 2.0f), lineUnitDir.x * sinRotAngle,
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

            if (experimentParams.weightingParameters.weightingMethod
                  == twisty::WeightingMethod::RadiativeTransfer) {
                twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                      tangents.data(), curvatures.data(), 5, experimentGeometry);
            } else {
                twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
                      tangents.data(), curvatures.data(), 5, experimentGeometry);
            }

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10(curvatures.data(), 4,
                        weightLookupTable, experimentParams.weightingParameters.absorption);

            if (!scatteringWeightLog10.isValid)
                continue;
            scatteringWeightLog10.weight += pathNormalizerLog10;

            // Update the min and max values
            if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }
            if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
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


    Result SixSegmentAngleSpaceMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
    {
        const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

        const Farlor::Vector3 point1
              = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

        const Farlor::Vector3 point5
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
        //std::cout << "Max threads: " << maxThreads << '\n';
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

#pragma omp parallel for num_threads(maxThreads) default(none)                                     \
      shared(randomGenPerThread, combinedWeightValues, minPathWeightPerThread,                     \
                  maxPathWeightPerThread, numExperimentPaths, thetaMin, thetaMax, phiMin, phiMax,  \
                  point1, point5, ds, experimentGeometry, weightLookupTable, pathNormalizerLog10,  \
                  experimentParams, combinedWeightValuesPerThread)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::mt19937_64 &randomGen = randomGenPerThread[threadId];

            std::uniform_real_distribution<float> phiDist(phiMin, phiMax);
            std::uniform_real_distribution<float> thetaDist(thetaMin, thetaMax);

            const float phi1Mapped = phiDist(randomGen);
            const float phi1 = std::acos(1.0 - 2.0 * phi1Mapped);

            const float phi2Mapped = phiDist(randomGen);
            const float phi2 = std::cos(1.0f - 2.0f * phi2Mapped);

            const float theta1 = thetaDist(randomGen);
            const float theta2 = thetaDist(randomGen);
            const float theta3 = thetaDist(randomGen);
            /*
                  x = ρsinφcosθ
                  y = ρsinφsinθ
                  z = ρcosφ 
            */

            const float sinPhi1 = std::sin(phi1);
            const float cosPhi1 = std::cos(phi1);
            const float sinTheta1 = std::sin(theta1);
            const float cosTheta1 = std::cos(theta1);

            const float sinPhi2 = std::sin(phi2);
            const float cosPhi2 = std::cos(phi2);
            const float sinTheta2 = std::sin(theta2);
            const float cosTheta2 = std::cos(theta2);

            // Calculate the first segment position
            const Farlor::Vector3 segment1Dir
                  = Farlor::Vector3(sinPhi1 * cosTheta1, sinPhi1 * sinTheta1, cosPhi1);
            const Farlor::Vector3 point2 = point1 + segment1Dir * ds;

            const float remainingDistance_2 = (point5 - point2).SqrMagnitude();

            if ((9 * ds * ds) < remainingDistance_2) {
                continue;
            }

            const Farlor::Vector3 segment2Dir
                  = Farlor::Vector3(sinPhi2 * cosTheta2, sinPhi2 * sinTheta2, cosPhi2);
            const Farlor::Vector3 point3 = point2 + segment2Dir * ds;

            const float remainingDistance2_2 = (point5 - point3).SqrMagnitude();
            if ((4 * ds * ds) < remainingDistance2_2) {
                continue;
            }

            const Farlor::Vector3 x_p = (point3 + point5) * 0.5;
            const Farlor::Vector3 lineUnitDir = (point5 - point3).Normalized();

            Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
            if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
                otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
            }

            const Farlor::Vector3 normalToLine = lineUnitDir.Cross(otherCrossVec).Normalized();

            // We should have an even number of segments remaining
            const float hypot = ds;
            const float D_2 = (point5 - point3).Magnitude() * 0.5f;
            assert(D_2 < hypot && "This should never be reached due to earlier check.");

            const float distanceOffLine = std::sqrt((hypot * hypot) - (D_2 * D_2));
            Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

            // Now rotate randomly theta amount around the axis.
            {
                const float sinRotAngle = std::sin(theta3 / 2.0f);
                float quaternionRotation[4]
                      = { std::cos(theta3 / 2.0f), lineUnitDir.x * sinRotAngle,
                            lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };


                Farlor::Vector3 shiftedPoint = x_t - point3;
                // Rotate and stuff back in shifted point
                twisty::RotateVectorByQuaternion(quaternionRotation, shiftedPoint.m_data.data());
                // Update the point with the rotated version
                x_t = shiftedPoint + point3;
            }
            const Farlor::Vector3 point4 = x_t;

            std::array<Farlor::Vector3, 7> points = { experimentGeometry.m_startPos, point1, point2,
                point3, point4, point5, experimentGeometry.m_endPos };
            std::array<Farlor::Vector3, 6> tangents;
            std::array<float, 5> curvatures;

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), 6, experimentGeometry);
            twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                  tangents.data(), curvatures.data(), 6, experimentGeometry);

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10(curvatures.data(), 5,
                        weightLookupTable, experimentParams.weightingParameters.absorption);
            scatteringWeightLog10.weight += pathNormalizerLog10;

            if (!scatteringWeightLog10.isValid) {
                continue;
            }
            // Update the min and max values
            if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }
            if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
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


    // Path Generation Helper Functions
    // Returns the single point
    void ResolveTwoSegments(std::vector<Farlor::Vector3> &pointList,
          const size_t leftSegmentStartIdx, const size_t rightSegmentEndIdx, const double ds,
          std::mt19937_64 &rng)
    {
        if (leftSegmentStartIdx >= rightSegmentEndIdx) {
            throw std::runtime_error(
                  "Left segment start idx must be less than right segment end idx");
        }
        if ((rightSegmentEndIdx - leftSegmentStartIdx) != 2) {
            throw std::runtime_error("Indices must be 2 apart");
        }

        const Farlor::Vector3 &leftSegmentStart = pointList[leftSegmentStartIdx];
        const Farlor::Vector3 &rightSegmentEnd = pointList[rightSegmentEndIdx];

        const size_t finalPointIdx = leftSegmentStartIdx + 1;
        Farlor::Vector3 &finalPoint = pointList[finalPointIdx];

        // Place segment exactly in the center
        const float d = (rightSegmentEnd - leftSegmentStart).Magnitude();
        // If the segments are exactly d segments apart, then we can just place the point in the center
        if (abs((2.0f * ds) - d) < 0.001f) {
            finalPoint = (leftSegmentStart + rightSegmentEnd) * 0.5f;
            return;
        }

        // Otherwise, we have a few other cases
        // First, we make sure we can have an intersection at all
        if ((2.0f * ds) < d) {
            throw std::runtime_error("Spheres do not intersect. No intersection");
        }

        // Theta dist
        std::uniform_real_distribution<float> thetaDist(0.0f, 2.0f * twisty::TwistyPi);
        const float theta = thetaDist(rng);

        // Handle case where the two points are stacked. In this case, we can randomly place the segments around the sphere centered at both points x0, x2
        if (d < 0.001f) {
            std::uniform_real_distribution<float> phiDist(0, 1);
            const float phi = std::acos(1.0 - 2.0 * phiDist(rng));

            // Lets place relative to the z-axis cause why not
            Farlor::Vector3 centerOffset = Farlor::Vector3(std::sin(phi) * std::cos(theta),
                                                 std::sin(phi) * std::sin(theta), std::cos(phi))
                  * ds;
            finalPoint = leftSegmentStart + centerOffset;
            return;
        }

        // Ok, last case, phi is defined by the boundary of the problem. We also randomly rotate by theta
        const Farlor::Vector3 x_p = (leftSegmentStart + rightSegmentEnd) * 0.5;
        const Farlor::Vector3 lineUnitDir = (rightSegmentEnd - leftSegmentStart).Normalized();

        Farlor::Vector3 otherCrossVec(1.0, 0.0, 0.0);
        if (abs(lineUnitDir.Dot(otherCrossVec)) >= 0.99) {
            otherCrossVec = Farlor::Vector3(0.0, 1.0, 0.0);
        }

        const Farlor::Vector3 normalToLine = lineUnitDir.Cross(otherCrossVec).Normalized();

        const float d_2 = d * 0.5f;

        float distanceOffLine = 0.0f;
        if (ds > d_2) {
            distanceOffLine = std::sqrt((ds * ds) - (d_2 * d_2));
        }
        Farlor::Vector3 x_t = x_p + normalToLine * distanceOffLine;

        // Now rotate randomly theta amount around the axis.

        const float sinRotAngle = std::sin(theta / 2.0f);
        float quaternionRotation[4] = { std::cos(theta / 2.0f), lineUnitDir.x * sinRotAngle,
            lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };

        Farlor::Vector3 shiftedPoint = x_t - leftSegmentStart;
        // Rotate and stuff back in shifted point
        twisty::RotateVectorByQuaternion(quaternionRotation, shiftedPoint.m_data.data());
        // Update the point with the rotated version
        x_t = shiftedPoint + leftSegmentStart;

        finalPoint = x_t;
    };

    // Path Generation Helper Functions
    // Places two points
    void ResolveThreeSegments(std::vector<Farlor::Vector3> &pointList,
          const size_t leftSegmentStartIdx, const size_t rightSegmentEndIdx, const double ds,
          std::mt19937_64 &rng)
    {
        if (leftSegmentStartIdx >= rightSegmentEndIdx) {
            throw std::runtime_error(
                  "Left segment start idx must be less than right segment end idx");
        }
        if ((rightSegmentEndIdx - leftSegmentStartIdx) != 3) {
            throw std::runtime_error("Indices must be 3 apart");
        }

        const Farlor::Vector3 &leftSegmentStart = pointList[leftSegmentStartIdx];
        const Farlor::Vector3 &rightSegmentEnd = pointList[rightSegmentEndIdx];

        const size_t firstPlacedPointIdx = leftSegmentStartIdx + 1;
        Farlor::Vector3 &firstPlacedPoint = pointList[firstPlacedPointIdx];

        // Place segment exactly in the center
        const float d = (rightSegmentEnd - leftSegmentStart).Magnitude();

        // If the segments are exactly d segments apart, then we can just place the point in the center
        if (abs((3.0f * ds) - d) < 0.001f) {
            firstPlacedPoint
                  = leftSegmentStart + (rightSegmentEnd - leftSegmentStart) * (1.0f / 3.0f);
            ResolveTwoSegments(pointList, firstPlacedPointIdx, rightSegmentEndIdx, ds, rng);
            return;
        }

        // Otherwise, we have a few other cases
        // First, we make sure we can have an intersection at all
        if ((3.0f * ds) < d) {
            throw std::runtime_error("Spheres do not intersect. No intersection");
        }

        // Theta dist
        std::uniform_real_distribution<float> thetaDist(0.0f, 2.0f * twisty::TwistyPi);
        const float theta = thetaDist(rng);

        // Handle case where the two points are stacked. In this case, we can randomly place the segments around the sphere centered at both points x0, x2
        if (d < 0.001f) {
            std::uniform_real_distribution<float> phiDist(0, 1);
            const float phi = std::acos(1.0 - 2.0 * phiDist(rng));

            // Lets place relative to the z-axis cause why not
            Farlor::Vector3 centerOffset = Farlor::Vector3(std::sin(phi) * std::cos(theta),
                                                 std::sin(phi) * std::sin(theta), std::cos(phi))
                  * ds;
            firstPlacedPoint = leftSegmentStart + centerOffset;
            ResolveTwoSegments(pointList, firstPlacedPointIdx, rightSegmentEndIdx, ds, rng);
            return;
        }

        // Uniform dist
        std::uniform_real_distribution<float> uniformRandom(0.0f, 1.0f);

        // Z axis of new corrdinate frame
        const Farlor::Vector3 zAxis = (rightSegmentEnd - leftSegmentStart).Normalized();
        // Generate orthogonal basis vectors x axis and y axis
        Farlor::Vector3 randomVector = Farlor::Vector3(1.0f, 0.0f, 0.0f);
        if (std::abs(zAxis.Dot(randomVector)) > 0.999f) {
            randomVector = Farlor::Vector3(0.0f, 1.0f, 0.0f);
        }
        const Farlor::Vector3 xAxis = zAxis.Cross(randomVector).Normalized();
        const Farlor::Vector3 yAxis = zAxis.Cross(xAxis).Normalized();

        // Generation of curve stuff
        const double d2 = d * d;
        const double leftRadius = ds;
        const double leftRadius2 = leftRadius * leftRadius;
        const double rightRadius = 2.0f * ds;
        const double rightRadius2 = rightRadius * rightRadius;

        double phiExtent = 0.0f;

        if ((d + leftRadius) < rightRadius) {
            phiExtent = twisty::TwistyPi;
        } else {
            const double h = 0.5 + (leftRadius2 - rightRadius2) / (2.0 * d2);
            double a = 0.0f;
            if (abs(leftRadius2 - (h * h * d2)) < 0.001) {
                a = 0.0f;
            } else {
                a = std::sqrt(leftRadius2 - (h * h * d2));
            }

            phiExtent = (h * d < 0.0f) ? twisty::TwistyPi - std::asin(a / leftRadius)
                                       : std::asin(a / leftRadius);
            if (phiExtent != phiExtent) {
                throw std::runtime_error("Phi extent is nan");
            }
        }

        const float uniformPhiSamplingMax = 0.5f - std::cos(phiExtent) * 0.5f;
        std::uniform_real_distribution<double> phiDist(0, uniformPhiSamplingMax);
        const double phi = std::acos(1.0 - 2.0 * phiDist(rng));

        const Farlor::Vector3 firstPlacedSegmentDir = xAxis * std::sin(phi) * std::cos(theta)
              + yAxis * std::sin(phi) * std::sin(theta) + zAxis * std::cos(phi);

        firstPlacedPoint = leftSegmentStart + firstPlacedSegmentDir.Normalized() * ds;
        ResolveTwoSegments(pointList, firstPlacedPointIdx, rightSegmentEndIdx, ds, rng);
    };

    void ResolveEvenNumberOfSegments(const int numSegments, std::vector<Farlor::Vector3> &pointList,
          const size_t leftSegmentStartIdx, const size_t rightSegmentEndIdx, const double ds,
          std::mt19937_64 &rng)
    {
        if ((numSegments % 2) != 0) {
            throw std::runtime_error("Even number of segments required");
        }
        const int numSegmentsPerSide = numSegments / 2;
        if (numSegmentsPerSide != 3 && numSegmentsPerSide != 2 && ((numSegmentsPerSide % 2) != 0)) {
            throw std::runtime_error("Invalid number of segments per side. We can only resolve "
                                     "segments counts of 2, 3 or even.");
        }

        const Farlor::Vector3 &leftPoint = pointList[leftSegmentStartIdx];
        const Farlor::Vector3 &rightPoint = pointList[rightSegmentEndIdx];

        const size_t centerPointIdx = leftSegmentStartIdx + numSegmentsPerSide;
        Farlor::Vector3 &centerPoint = pointList[centerPointIdx];

        const double d = (rightPoint - leftPoint).Magnitude();

        // If the segments are exactly d segments apart, then just place the point in the center
        if (abs((numSegments * ds) - d) < 0.001f) {
            centerPoint = 0.5f * (rightPoint + leftPoint);

            if (numSegmentsPerSide == 2) {
                ResolveTwoSegments(pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
                ResolveTwoSegments(pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
            } else if (numSegmentsPerSide == 3) {
                ResolveThreeSegments(pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
                ResolveThreeSegments(pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
            } else {
                // Left half recurse
                ResolveEvenNumberOfSegments(
                      numSegmentsPerSide, pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
                // Right half recurse
                ResolveEvenNumberOfSegments(
                      numSegmentsPerSide, pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
            }
            // We are done after this and need to early out
            return;
        }

        // Generation of curve stuff
        const double radiusPerSide = ds * numSegmentsPerSide;

        // We want to early out in this case. Somehow we have an invalid environment or path construction
        if ((radiusPerSide + radiusPerSide) < d) {
            throw std::runtime_error("Spheres dont intersect");
        }

        // If stacked, we need to be careful
        if (d < 0.001f) {
            std::uniform_real_distribution<double> phiDist(0, 1);
            std::uniform_real_distribution<double> thetaDist(0.0f, 2.0f * twisty::TwistyPi);
            std::uniform_real_distribution<float> uniformRandom(0.0f, 1.0f);

            const double phi = std::acos(1.0 - 2.0 * phiDist(rng));
            const double theta = thetaDist(rng);

            const double sampledRadius = radiusPerSide * std::pow(uniformRandom(rng), 1.0 / 3.0);

            const Farlor::Vector3 zAxis = (rightPoint - leftPoint).Normalized();
            // Generate orthogonal basis vectors x axis and y axis
            Farlor::Vector3 randomVector = Farlor::Vector3(1.0f, 0.0f, 0.0f);
            if (std::abs(zAxis.Dot(randomVector)) > 0.999f) {
                randomVector = Farlor::Vector3(0.0f, 1.0f, 0.0f);
            }
            const Farlor::Vector3 xAxis = zAxis.Cross(randomVector).Normalized();
            const Farlor::Vector3 yAxis = zAxis.Cross(xAxis).Normalized();

            Farlor::Vector3 centerOffset = xAxis * std::sin(phi) * std::cos(theta)
                  + yAxis * std::sin(phi) * std::sin(theta) + zAxis * std::cos(phi);
            centerOffset = centerOffset * sampledRadius;
            centerPoint = leftPoint + centerOffset;

            if (numSegmentsPerSide == 2) {
                ResolveTwoSegments(pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
                ResolveTwoSegments(pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
            } else if (numSegmentsPerSide == 3) {
                ResolveThreeSegments(pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
                ResolveThreeSegments(pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
            } else {
                // Left half recurse
                ResolveEvenNumberOfSegments(
                      numSegmentsPerSide, pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
                // Right half recurse
                ResolveEvenNumberOfSegments(
                      numSegmentsPerSide, pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
            }
            // We are done after this and need to early out
            return;
        }

        const double d2 = d * d;
        const double radiusPerSide2 = radiusPerSide * radiusPerSide;

        const Farlor::Vector3 midPoint = 0.5f * (rightPoint + leftPoint);

        const double distToMidpoint = (midPoint - leftPoint).Magnitude();

        double phiExtent = 0.0f;


        const double h = 0.5;
        const double a = std::sqrt(radiusPerSide2 - (h * h * d2));

        phiExtent = std::asin(a / radiusPerSide);

        std::uniform_int_distribution<int> coinFlip(0, 1);

        const float uniformPhiSamplingMax = 0.5f - std::cos(phiExtent) * 0.5f;
        std::uniform_real_distribution<double> phiDist(0, uniformPhiSamplingMax);

        std::uniform_real_distribution<double> thetaDist(0.0f, 2.0f * twisty::TwistyPi);
        std::uniform_real_distribution<float> uniformRandom(0.0f, 1.0f);

        const double phi = std::acos(1.0 - 2.0 * phiDist(rng));
        const double theta = thetaDist(rng);

        const bool coinFlipResult = coinFlip(rng);

        const double hypot = distToMidpoint / std::cos(phi);

        const double maxRadius = numSegmentsPerSide * ds;
        const double minRadiusPercent = hypot / maxRadius;

        const double sampledRadius = maxRadius
              * std::pow(
                    minRadiusPercent + (1.0f - minRadiusPercent) * uniformRandom(rng), 1.0 / 3.0);

        if (coinFlipResult == false) {
            const Farlor::Vector3 zAxis = (rightPoint - leftPoint).Normalized();
            // Generate orthogonal basis vectors x axis and y axis
            Farlor::Vector3 randomVector = Farlor::Vector3(1.0f, 0.0f, 0.0f);
            if (std::abs(zAxis.Dot(randomVector)) > 0.999f) {
                randomVector = Farlor::Vector3(0.0f, 1.0f, 0.0f);
            }
            const Farlor::Vector3 xAxis = zAxis.Cross(randomVector).Normalized();
            const Farlor::Vector3 yAxis = zAxis.Cross(xAxis).Normalized();

            Farlor::Vector3 centerOffset = xAxis * std::sin(phi) * std::cos(theta)
                  + yAxis * std::sin(phi) * std::sin(theta) + zAxis * std::cos(phi);
            centerOffset = centerOffset * sampledRadius;
            pointList[centerPointIdx] = leftPoint + centerOffset;
            // Right half
        } else {
            const Farlor::Vector3 zAxis = (rightPoint - leftPoint).Normalized();
            // Generate orthogonal basis vectors x axis and y axis
            Farlor::Vector3 randomVector = Farlor::Vector3(1.0f, 0.0f, 0.0f);
            if (std::abs(zAxis.Dot(randomVector)) > 0.999f) {
                randomVector = Farlor::Vector3(0.0f, 1.0f, 0.0f);
            }
            const Farlor::Vector3 xAxis = zAxis.Cross(randomVector).Normalized();
            const Farlor::Vector3 yAxis = zAxis.Cross(xAxis).Normalized();

            Farlor::Vector3 centerOffset = xAxis * std::sin(phi) * std::cos(theta)
                  + yAxis * std::sin(phi) * std::sin(theta) + zAxis * std::cos(phi) * -1.0f;
            centerOffset = centerOffset * sampledRadius;
            pointList[centerPointIdx] = rightPoint + centerOffset;
        }

        // Ok, now that we have set the center point, we need to set the other points
        // Left half
        if (numSegmentsPerSide == 2) {
            ResolveTwoSegments(pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
            ResolveTwoSegments(pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
        } else if (numSegmentsPerSide == 3) {
            ResolveThreeSegments(pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
            ResolveThreeSegments(pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
        } else {
            // Left half recurse
            ResolveEvenNumberOfSegments(
                  numSegmentsPerSide, pointList, leftSegmentStartIdx, centerPointIdx, ds, rng);
            // Right half recurse
            ResolveEvenNumberOfSegments(
                  numSegmentsPerSide, pointList, centerPointIdx, rightSegmentEndIdx, ds, rng);
        }
    }

    Result FiveSegmentPathGenerationMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
    {
        const uint64_t numTotalPaths = numExperimentPaths;
        const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

        const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
        const Farlor::Vector3 point1
              = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

        const Farlor::Vector3 point5 = experimentGeometry.m_endPos;
        const Farlor::Vector3 point4
              = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        //std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

        // Random Gen per thread
        std::vector<std::mt19937_64> rngPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            rngPerThread[i].seed(i);
        }

#pragma omp parallel for num_threads(maxThreads) default(none)                                     \
      shared(combinedWeightValuesPerThread, minPathWeightPerThread, maxPathWeightPerThread,        \
                  numExperimentPaths, point0, point1, point4, point5, rngPerThread, ds,            \
                  experimentGeometry, experimentParams, weightLookupTable, pathNormalizerLog10,    \
                  combinedWeightValues)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::vector<Farlor::Vector3> points
                  = { point0, point1, Farlor::Vector3(0.0f, 0.0f, 0.0f),
                        Farlor::Vector3(0.0f, 0.0f, 0.0f), point4, point5 };

            ResolveThreeSegments(points, 1, 4, ds, rngPerThread[threadId]);

            std::array<Farlor::Vector3, 5> tangents;
            std::array<float, 4> curvatures;

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), 5, experimentGeometry);

            if (experimentParams.weightingParameters.weightingMethod
                  == WeightingMethod::RadiativeTransfer) {
                twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                      tangents.data(), curvatures.data(), 5, experimentGeometry);
            } else {
                twisty::PerturbUtils::UpdateCurvaturesFromTangents_SimplifiedModel(
                      tangents.data(), curvatures.data(), 5, experimentGeometry);
            }

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10(curvatures.data(), 4,
                        weightLookupTable, experimentParams.weightingParameters.absorption);
            scatteringWeightLog10.weight += pathNormalizerLog10;

            // Update the min and max values
            if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }
            if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
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

        // Go through each current thread combined weight value and add it in
        for (const auto &combinedWeightValue : combinedWeightValuesPerThread) {
            if (combinedWeightValue.m_numValues > 0) {
                pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
                numValidPaths += combinedWeightValue.m_numValues;
            }
        }

        boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;

        if (numValidPaths > 0) {
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }

    Result SixSegmentPathGenerationMC(const int64_t numExperimentPaths,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
    {
        const uint64_t numTotalPaths = numExperimentPaths;
        const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

        const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
        const Farlor::Vector3 point1
              = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

        const Farlor::Vector3 point6 = experimentGeometry.m_endPos;
        const Farlor::Vector3 point5
              = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        //std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

        // Random Gen per thread
        std::vector<std::mt19937_64> rngPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            rngPerThread[i].seed(i);
        }

        std::cout << "Starting path generation" << std::endl;

#pragma omp parallel for num_threads(maxThreads) default(none) shared(                             \
            combinedWeightValuesPerThread, minPathWeightPerThread, maxPathWeightPerThread, point1, \
            point5, experimentGeometry, ds, pathNormalizerLog10, numExperimentPaths, point0,       \
            point6, rngPerThread, weightLookupTable, experimentParams, combinedWeightValues)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::vector<Farlor::Vector3> points = { point0, point1,
                Farlor::Vector3(0.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 0.0f),
                Farlor::Vector3(0.0f, 0.0f, 0.0f), point5, point6 };

            const int numFreeSegments = 4;
            ResolveEvenNumberOfSegments(numFreeSegments, points, 1, 5, ds, rngPerThread[threadId]);

            std::array<Farlor::Vector3, 6> tangents;
            std::array<float, 5> curvatures;

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), 6, experimentGeometry);
            twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                  tangents.data(), curvatures.data(), 6, experimentGeometry);

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10(curvatures.data(), 5,
                        weightLookupTable, experimentParams.weightingParameters.absorption);

            if (!scatteringWeightLog10.isValid) {
                std::cout << "We have an invalid weight path, as it counts nothing, discard it"
                          << std::endl;
                continue;
            }
            scatteringWeightLog10.weight += pathNormalizerLog10;

            // Update the min and max values
            if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }
            if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
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

        // Go through each current thread combined weight value and add it in
        for (const auto &combinedWeightValue : combinedWeightValuesPerThread) {
            if (combinedWeightValue.m_numValues > 0) {
                pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
                numValidPaths += combinedWeightValue.m_numValues;
            }
        }

        boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;

        if (numValidPaths > 0) {
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }

    Result MSegmentPathGenerationMC(const int64_t numExperimentPaths, const int numSegmentsPerCurve,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable)
    {
        const uint64_t numTotalPaths = numExperimentPaths;
        const float ds = experimentGeometry.arclength / experimentParams.numSegmentsPerCurve;

        const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
        const Farlor::Vector3 point1
              = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

        const Farlor::Vector3 pointM = experimentGeometry.m_endPos;
        const Farlor::Vector3 pointMm1
              = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        //std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

        const uint32_t baseSeed = time(0);

        // Random Gen per thread
        std::vector<std::mt19937_64> rngPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            rngPerThread[i].seed(baseSeed + i);
        }

        std::cout << "Starting path generation" << std::endl;

#pragma omp parallel for num_threads(maxThreads) default(none)                                     \
      shared(combinedWeightValuesPerThread, minPathWeightPerThread, maxPathWeightPerThread,        \
                  numExperimentPaths, point0, point1, pointM, pointMm1, numSegmentsPerCurve,       \
                  rngPerThread, ds, experimentParams, experimentGeometry, weightLookupTable,       \
                  pathNormalizerLog10, combinedWeightValues)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::vector<Farlor::Vector3> points;
            points.resize(numSegmentsPerCurve + 1);
            points[0] = point0;
            points[1] = point1;
            points[numSegmentsPerCurve - 1] = pointMm1;
            points[numSegmentsPerCurve] = pointM;

            const int numFreeSegments = numSegmentsPerCurve - 2;
            ResolveEvenNumberOfSegments(
                  numFreeSegments, points, 1, numSegmentsPerCurve - 1, ds, rngPerThread[threadId]);

            std::vector<Farlor::Vector3> tangents;
            tangents.resize(numSegmentsPerCurve);
            std::vector<float> curvatures;
            curvatures.resize(numSegmentsPerCurve - 1);

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), numSegmentsPerCurve, experimentGeometry);
            twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                  tangents.data(), curvatures.data(), numSegmentsPerCurve, experimentGeometry);

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10(curvatures.data(),
                        numSegmentsPerCurve - 1, weightLookupTable,
                        experimentParams.weightingParameters.absorption);

            if (!scatteringWeightLog10.isValid) {
                continue;
            }

            if (isnan(scatteringWeightLog10.weight)) {
                throw std::runtime_error("We somehow got an invalid path weight");
            }
            scatteringWeightLog10.weight += pathNormalizerLog10;

            // Update the min and max values
            if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }
            if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
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

        // Go through each current thread combined weight value and add it in
        for (const auto &combinedWeightValue : combinedWeightValuesPerThread) {
            if (combinedWeightValue.m_numValues > 0) {
                pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
                numValidPaths += combinedWeightValue.m_numValues;
            }
        }

        boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;

        if (numValidPaths > 0) {
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }

    Result MSegmentPathGenerationMC(const uint64_t seed, const int64_t numExperimentPaths,
          const int numSegmentsPerCurve,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &cachedWeightLookupTable,
          const float maxDs)
    {
        const uint64_t numTotalPaths = numExperimentPaths;

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        //std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

        // Random Gen per thread
        std::vector<std::mt19937_64> rngPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            rngPerThread[i].seed(seed + i);
        }

        // // Assume that the directions are aligned
        // if (experimentGeometry.m_startDir.Dot(experimentGeometry.m_endDir) < 0.999f) {
        //     std::cout << "Start and end directions are not aligned" << std::endl;
        //     return { 0, 0, 0.0f, 0.0, 0.0 };
        // }

        // Min arclength

        float minArclength = 0.0f;

        // Old way
        // {
        //     const float a = Farlor::Vector3(
        //           0.0f, experimentGeometry.m_endPos.y, experimentGeometry.m_endPos.z)
        //                           .Magnitude();
        //     const float b = experimentGeometry.m_endPos.x;
        //     const uint32_t M = experimentParams.numSegmentsPerCurve;

        //     minArclength
        //           = (std::sqrt(a * a * (M - 2) * M + b * b * (M - 2) * (M - 2)) - 2 * b) / (M - 4);
        //     std::cout << "Min Arclength: " << minArclength << std::endl;
        // }

        // New Way
        {
            const uint32_t M = experimentParams.numSegmentsPerCurve;
            const Farlor::Vector3 Xs = experimentGeometry.m_startPos;
            const Farlor::Vector3 Xe = experimentGeometry.m_endPos;
            const Farlor::Vector3 Ns = experimentGeometry.m_startDir;
            const Farlor::Vector3 Ne = experimentGeometry.m_endDir;

            const float a = Farlor::Vector3(Ns + Ne).Dot((Ns + Ne)) - ((M - 4.0f) * M + 4.0f);
            const float b = -2.0f * M * (Ns + Ne).Dot((Xe - Xs));
            const float c = M * M * Farlor::Vector3(Xe - Xs).Dot((Xe - Xs));

            const float minArclengthCandidateOne
                  = (-b - std::sqrt(b * b - 4.0f * a * c)) / (2.0f * a);
            const float minArclengthCandidateTwo
                  = (-b + std::sqrt(b * b - 4.0f * a * c)) / (2.0f * a);
            std::cout << "Min Arclength candidate one: " << minArclengthCandidateOne << std::endl;
            std::cout << "Min Arclength candidate two: " << minArclengthCandidateTwo << std::endl;

            minArclength = 1000000.0f;
            if (!std::isnan(minArclengthCandidateOne) && minArclengthCandidateOne > 0.0f)
                minArclength = std::min(minArclength, minArclengthCandidateOne);
            if (!std::isnan(minArclengthCandidateTwo) && minArclengthCandidateTwo > 0.0f)
                minArclength = std::min(minArclength, minArclengthCandidateTwo);
            std::cout << "Selected arclength = " << minArclength << std::endl;
        }

        const float minDs = minArclength / experimentParams.numSegmentsPerCurve;

        const float actualMaxDs = maxDs;
        //maxArclength / experimentParams.numSegmentsPerCurve;

        if (actualMaxDs < minDs) {
            std::cout << "No solution for current environment" << std::endl;
            return { 0, 0, 0.0f, 0.0, 0.0 };
        }

        std::cout << "Starting path generation" << std::endl;

#pragma omp parallel for num_threads(maxThreads) default(none)                                     \
      shared(combinedWeightValuesPerThread, minPathWeightPerThread, maxPathWeightPerThread,        \
                  numExperimentPaths, rngPerThread, experimentGeometry, experimentParams,          \
                  actualMaxDs, minDs, cachedWeightLookupTable, pathNormalizerLog10,                \
                  combinedWeightValues, numSegmentsPerCurve)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::uniform_real_distribution<float> uniformRand(0.0f, 1.0f);
            const float ds = uniformRand(rngPerThread[threadId]) * (actualMaxDs - minDs) + minDs;

            twisty::PathWeighting::BaseWeightLookupTable &weightLookupTable
                  = *cachedWeightLookupTable.GetWeightLookupTable(ds);

            const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
            const Farlor::Vector3 point1
                  = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

            const Farlor::Vector3 pointM = experimentGeometry.m_endPos;
            const Farlor::Vector3 pointMm1
                  = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;


            std::vector<Farlor::Vector3> points;
            points.resize(numSegmentsPerCurve + 1);
            points[0] = point0;
            points[1] = point1;
            points[numSegmentsPerCurve - 1] = pointMm1;
            points[numSegmentsPerCurve] = pointM;

            const int numFreeSegments = numSegmentsPerCurve - 2;
            ResolveEvenNumberOfSegments(
                  numFreeSegments, points, 1, numSegmentsPerCurve - 1, ds, rngPerThread[threadId]);

            std::vector<Farlor::Vector3> tangents;
            tangents.resize(numSegmentsPerCurve);
            std::vector<float> curvatures;
            curvatures.resize(numSegmentsPerCurve - 1);

            twisty::PerturbUtils::BoundaryConditions bcWithArclength = experimentGeometry;
            bcWithArclength.arclength = ds * experimentParams.numSegmentsPerCurve;

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), numSegmentsPerCurve, bcWithArclength);
            twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                  tangents.data(), curvatures.data(), numSegmentsPerCurve, bcWithArclength);

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaCurvatureLog10(curvatures.data(),
                        numSegmentsPerCurve - 1, weightLookupTable,
                        experimentParams.weightingParameters.absorption);

            if (!scatteringWeightLog10.isValid) {
                continue;
            }

            if (isnan(scatteringWeightLog10.weight)) {
                throw std::runtime_error("We somehow got an invalid path weight");
            }
            scatteringWeightLog10.weight += pathNormalizerLog10;

            // Update the min and max values
            if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }
            if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
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

        // Go through each current thread combined weight value and add it in
        for (const auto &combinedWeightValue : combinedWeightValuesPerThread) {
            if (combinedWeightValue.m_numValues > 0) {
                pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
                numValidPaths += combinedWeightValue.m_numValues;
            }
        }

        boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;

        if (numValidPaths > 0) {
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }

    Result MSegmentPathGenerationMC(const uint64_t seed, const int64_t numExperimentPaths,
          const int numSegmentsPerCurve,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &environmentCachedWeightLookupTable,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &objectCachedWeightLookupTable,
          const float maxDs)
    {
        const uint64_t numTotalPaths = numExperimentPaths;

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        //std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

        // Random Gen per thread
        std::vector<std::mt19937_64> rngPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            rngPerThread[i].seed(seed + i);
        }

        // Min arclength
        float minArclength = twisty::PathGeneration::CalculateMinimumArclength(
              experimentGeometry, experimentParams.numSegmentsPerCurve);
        const float minDs = minArclength / experimentParams.numSegmentsPerCurve;

        const float actualMaxDs = maxDs;
        //maxArclength / experimentParams.numSegmentsPerCurve;

        if (actualMaxDs < minDs) {
            std::cout << "No solution for current environment" << std::endl;
            return { 0, 0, 0.0f, 0.0, 0.0 };
        }

        std::vector<int> firstSamplePerThread(maxThreads, 1);
        std::vector<std::vector<Farlor::Vector3>> samplePointsPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            samplePointsPerThread[i].reserve(numSegmentsPerCurve + 1);
        }

        std::vector<twisty::PathWeighting::PathWeightValue> previousSampleWeightsValuesPerThread(
              maxThreads);

#pragma omp parallel for num_threads(maxThreads) default(none) shared(                             \
            combinedWeightValuesPerThread, minPathWeightPerThread, maxPathWeightPerThread,         \
            environmentCachedWeightLookupTable, objectCachedWeightLookupTable, numExperimentPaths, \
            rngPerThread, actualMaxDs, minDs, experimentGeometry, numSegmentsPerCurve,             \
            experimentParams, pathNormalizerLog10, std::cout, combinedWeightValues)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::uniform_real_distribution<float> uniformRand(0.0f, 1.0f);
            const float ds = uniformRand(rngPerThread[threadId]) * (actualMaxDs - minDs) + minDs;

            twisty::PathWeighting::BaseWeightLookupTable &environmentWeightLookupTable
                  = *environmentCachedWeightLookupTable.GetWeightLookupTable(ds);

            twisty::PathWeighting::BaseWeightLookupTable &objectWeightLookupTable
                  = *objectCachedWeightLookupTable.GetWeightLookupTable(ds);

            const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
            const Farlor::Vector3 point1
                  = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

            const Farlor::Vector3 pointM = experimentGeometry.m_endPos;
            const Farlor::Vector3 pointMm1
                  = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;

            std::vector<Farlor::Vector3> points;
            points.resize(numSegmentsPerCurve + 1);
            points[0] = point0;
            points[1] = point1;
            points[numSegmentsPerCurve - 1] = pointMm1;
            points[numSegmentsPerCurve] = pointM;

            const int numFreeSegments = numSegmentsPerCurve - 2;
            ResolveEvenNumberOfSegments(
                  numFreeSegments, points, 1, numSegmentsPerCurve - 1, ds, rngPerThread[threadId]);

            std::vector<Farlor::Vector3> tangents;
            tangents.resize(numSegmentsPerCurve);
            std::vector<float> curvatures;
            curvatures.resize(numSegmentsPerCurve - 1);

            twisty::PerturbUtils::BoundaryConditions bcWithArclength = experimentGeometry;
            bcWithArclength.arclength = ds * experimentParams.numSegmentsPerCurve;

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), numSegmentsPerCurve, bcWithArclength);
            twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                  tangents.data(), curvatures.data(), numSegmentsPerCurve, bcWithArclength);

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaPositionLog10_PositionDependent(points,
                        curvatures, environmentWeightLookupTable, objectWeightLookupTable,
                        experimentParams.weightingParameters.absorption);

            if (!scatteringWeightLog10.isValid) {
                std::cout << "We have an invalid weight path, as it counts nothing, discard it"
                          << std::endl;
                continue;
            }

            if (isnan(scatteringWeightLog10.weight)) {
                throw std::runtime_error("We somehow got an invalid path weight");
            }

            scatteringWeightLog10.weight += pathNormalizerLog10;

            if (firstSamplePerThread[threadId] == 1) {
                firstSamplePerThread[threadId] = 0;
                for (int i = 0; i < numSegmentsPerCurve + 1; i++) {
                    samplePointsPerThread[threadId][i] = points[i];
                }
                previousSampleWeightsValuesPerThread[threadId] = scatteringWeightLog10;
            } else {
                // Here, we want to perform metropolis hastings
                const double currentSampleWeight = std::pow(10.0f, scatteringWeightLog10.weight);
                const double previousSampleWeight
                      = std::pow(10.0f, previousSampleWeightsValuesPerThread[threadId].weight);

                const double acceptanceProb = currentSampleWeight / previousSampleWeight;
                const double randomAcceptanceProb = uniformRand(rngPerThread[threadId]);

                if (randomAcceptanceProb <= acceptanceProb) {
                    // We accept the sample
                    // Update the previous sample weight
                    for (int i = 0; i < numSegmentsPerCurve + 1; i++) {
                        samplePointsPerThread[threadId][i] = points[i];
                    }
                    previousSampleWeightsValuesPerThread[threadId] = scatteringWeightLog10;
                } else {
                    // We utilize the sample previously found
                }
            }

            const twisty::PathWeighting::PathWeightValue &acceptedWeightValue
                  = previousSampleWeightsValuesPerThread[threadId];

            // Update the min and max values
            if (acceptedWeightValue.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = acceptedWeightValue.weight;
            }
            if (acceptedWeightValue.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = acceptedWeightValue.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, acceptedWeightValue.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, acceptedWeightValue.weight);
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

        // Go through each current thread combined weight value and add it in
        for (const auto &combinedWeightValue : combinedWeightValuesPerThread) {
            if (combinedWeightValue.m_numValues > 0) {
                pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
                numValidPaths += combinedWeightValue.m_numValues;
            }
        }

        boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;

        if (numValidPaths > 0) {
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }

#ifdef __linux__
    Result MSegmentPathGenerationMC_VDB(const uint64_t seed, const int64_t numExperimentPaths,
          const int numSegmentsPerCurve,
          const twisty::PerturbUtils::BoundaryConditions &experimentGeometry,
          const twisty::ExperimentRunner::ExperimentParameters &experimentParams,
          const double pathNormalizerLog10,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &environmentCachedWeightLookupTable,
          const twisty::PathWeighting::CachedMultiArclengthWeightLookupTable
                &objectCachedWeightLookupTable,
          const float maxDs, openvdb::FloatGrid::Ptr grid)
    {
        const uint64_t numTotalPaths = numExperimentPaths;

        std::vector<twisty::CombinedWeightValues_C> combinedWeightValues;
        combinedWeightValues.reserve(
              (numTotalPaths + MaxNumPathsPerCombinedWeight - 1) / MaxNumPathsPerCombinedWeight);

        const int maxThreads = omp_get_max_threads();
        //std::cout << "Max threads: " << maxThreads << '\n';
        std::vector<twisty::CombinedWeightValues_C> combinedWeightValuesPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            twisty::CombinedWeightValues_C_Reset(combinedWeightValuesPerThread[i]);
        }

        // Per thread min values
        std::vector<double> minPathWeightPerThread(maxThreads, std::numeric_limits<double>::max());
        // Per thread max values
        std::vector<double> maxPathWeightPerThread(maxThreads, -std::numeric_limits<double>::max());

        // Random Gen per thread
        std::vector<std::mt19937_64> rngPerThread(maxThreads);
        for (int i = 0; i < maxThreads; i++) {
            rngPerThread[i].seed(seed + i);
        }

        // Min arclength

        float minArclength = 0.0f;
        // New Way
        {
            const uint32_t M = experimentParams.numSegmentsPerCurve;
            const Farlor::Vector3 Xs = experimentGeometry.m_startPos;
            const Farlor::Vector3 Xe = experimentGeometry.m_endPos;
            const Farlor::Vector3 Ns = experimentGeometry.m_startDir;
            const Farlor::Vector3 Ne = experimentGeometry.m_endDir;

            const float a = Farlor::Vector3(Ns + Ne).Dot((Ns + Ne)) - ((M - 4.0f) * M + 4.0f);
            const float b = -2.0f * M * (Ns + Ne).Dot((Xe - Xs));
            const float c = M * M * Farlor::Vector3(Xe - Xs).Dot((Xe - Xs));

            const float minArclengthCandidateOne
                  = (-b - std::sqrt(b * b - 4.0f * a * c)) / (2.0f * a);
            const float minArclengthCandidateTwo
                  = (-b + std::sqrt(b * b - 4.0f * a * c)) / (2.0f * a);
            //std::cout << "Min Arclength candidate one: " << minArclengthCandidateOne << std::endl;
            //std::cout << "Min Arclength candidate two: " << minArclengthCandidateTwo << std::endl;

            minArclength = 1000000.0f;
            if (!std::isnan(minArclengthCandidateOne) && minArclengthCandidateOne > 0.0f)
                minArclength = std::min(minArclength, minArclengthCandidateOne);
            if (!std::isnan(minArclengthCandidateTwo) && minArclengthCandidateTwo > 0.0f)
                minArclength = std::min(minArclength, minArclengthCandidateTwo);
            //std::cout << "Selected arclength = " << minArclength << std::endl;
        }

        const float minDs = minArclength / experimentParams.numSegmentsPerCurve;

        const float actualMaxDs = maxDs;
        //maxArclength / experimentParams.numSegmentsPerCurve;

        if (actualMaxDs < minDs) {
            std::cout << "No solution for current environment" << std::endl;
            return { 0, 0, 0.0f, 0.0, 0.0 };
        }

        //std::cout << "Starting path generation" << std::endl;

#pragma omp parallel for num_threads(maxThreads) default(none) shared(                             \
            combinedWeightValuesPerThread, minPathWeightPerThread, maxPathWeightPerThread,         \
            environmentCachedWeightLookupTable, objectCachedWeightLookupTable, numExperimentPaths, \
            rngPerThread, actualMaxDs, minDs, experimentGeometry, numSegmentsPerCurve,             \
            experimentParams, pathNormalizerLog10, std::cout, combinedWeightValues, grid)
        for (int64_t pathIdx = 0; pathIdx < numExperimentPaths; pathIdx++) {
            const int threadId = omp_get_thread_num();

            std::uniform_real_distribution<float> uniformRand(0.0f, 1.0f);
            const float ds = uniformRand(rngPerThread[threadId]) * (actualMaxDs - minDs) + minDs;

            twisty::PathWeighting::BaseWeightLookupTable &environmentWeightLookupTable
                  = *environmentCachedWeightLookupTable.GetWeightLookupTable(ds);

            twisty::PathWeighting::BaseWeightLookupTable &objectWeightLookupTable
                  = *objectCachedWeightLookupTable.GetWeightLookupTable(ds);

            const Farlor::Vector3 point0 = experimentGeometry.m_startPos;
            const Farlor::Vector3 point1
                  = experimentGeometry.m_startPos + ds * experimentGeometry.m_startDir;

            const Farlor::Vector3 pointM = experimentGeometry.m_endPos;
            const Farlor::Vector3 pointMm1
                  = experimentGeometry.m_endPos - ds * experimentGeometry.m_endDir;


            std::vector<Farlor::Vector3> points;
            points.resize(numSegmentsPerCurve + 1);
            points[0] = point0;
            points[1] = point1;
            points[numSegmentsPerCurve - 1] = pointMm1;
            points[numSegmentsPerCurve] = pointM;

            const int numFreeSegments = numSegmentsPerCurve - 2;
            ResolveEvenNumberOfSegments(
                  numFreeSegments, points, 1, numSegmentsPerCurve - 1, ds, rngPerThread[threadId]);

            std::vector<Farlor::Vector3> tangents;
            tangents.resize(numSegmentsPerCurve);
            std::vector<float> curvatures;
            curvatures.resize(numSegmentsPerCurve - 1);

            twisty::PerturbUtils::BoundaryConditions bcWithArclength = experimentGeometry;
            bcWithArclength.arclength = ds * experimentParams.numSegmentsPerCurve;

            twisty::PerturbUtils::UpdateTangentsFromPos(
                  points.data(), tangents.data(), numSegmentsPerCurve, bcWithArclength);
            twisty::PerturbUtils::UpdateCurvaturesFromTangents_RadiativeTransfer(
                  tangents.data(), curvatures.data(), numSegmentsPerCurve, bcWithArclength);

            twisty::PathWeighting::PathWeightValue scatteringWeightLog10
                  = twisty::PathWeighting::WeightCurveViaPositionLog10_PositionDependent(points,
                        curvatures, environmentWeightLookupTable, objectWeightLookupTable,
                        experimentParams.weightingParameters.absorption, grid);

            if (!scatteringWeightLog10.isValid) {
                std::cout << "We have an invalid weight path, as it counts nothing, discard it"
                          << std::endl;
                continue;
            }

            if (isnan(scatteringWeightLog10.weight)) {
                throw std::runtime_error("We somehow got an invalid path weight");
            }

            scatteringWeightLog10.weight += pathNormalizerLog10;

            // Detect the case we get a wildly insane path weight
            // if (scatteringWeightLog10.weight > -10) {
            //     std::cout << scatteringWeightLog10.weight << std::endl;
            //     std::cout << "Not sure what is going on, but this path weight is suspect as "
            //                  "hell.\nDont include it."
            //               << std::endl;
            //     continue;
            // }

            // Update the min and max values
            if (scatteringWeightLog10.weight < minPathWeightPerThread[threadId]) {
                minPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }
            if (scatteringWeightLog10.weight > maxPathWeightPerThread[threadId]) {
                maxPathWeightPerThread[threadId] = scatteringWeightLog10.weight;
            }

            twisty::CombinedWeightValues_C &activeWeightValue
                  = combinedWeightValuesPerThread[threadId];

            if (activeWeightValue.m_numValues < MaxNumPathsPerCombinedWeight) {
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
            } else {
#pragma omp critical
                {
                    combinedWeightValues.push_back(activeWeightValue);
                }
                twisty::CombinedWeightValues_C_Reset(activeWeightValue);
                twisty::CombinedWeightValues_C_AddValue(
                      activeWeightValue, scatteringWeightLog10.weight);
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

        // Go through each current thread combined weight value and add it in
        for (const auto &combinedWeightValue : combinedWeightValuesPerThread) {
            if (combinedWeightValue.m_numValues > 0) {
                pathIntegralResult += twisty::ExtractFinalValue(combinedWeightValue);
                numValidPaths += combinedWeightValue.m_numValues;
            }
        }

        boost::multiprecision::cpp_dec_float_100 finalResult = 0.0;

        if (numValidPaths > 0) {
            finalResult
                  = boost::multiprecision::cpp_dec_float_100(pathIntegralResult / numValidPaths);
        }

        return { numValidPaths, numTotalPaths, finalResult, overallMinPathWeightLog10,
            overallMaxPathWeightLog10 };
    }
#endif
}
}
