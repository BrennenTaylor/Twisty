#include "PathGeneration.h"

#include <MathConsts.h>
#include <CurvePerturbUtils.h>

#include <omp.h>
#include <random>

namespace twisty {
namespace PathGeneration {

    float CalculateMinimumArclength(
          twisty::PerturbUtils::BoundaryConditions boundaryConditions, uint32_t numSegmentsPerCurve)
    {
        float minArclength = 0.0f;
        // New Way
        {
            const uint32_t M = numSegmentsPerCurve;
            const Farlor::Vector3 Xs = boundaryConditions.m_startPos;
            const Farlor::Vector3 Xe = boundaryConditions.m_endPos;
            const Farlor::Vector3 Ns = boundaryConditions.m_startDir;
            const Farlor::Vector3 Ne = boundaryConditions.m_endDir;

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
        return minArclength;
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
        const Farlor::Vector3 &rightSegmentEnd = pointList[-rightSegmentEndIdx];

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
}
}
