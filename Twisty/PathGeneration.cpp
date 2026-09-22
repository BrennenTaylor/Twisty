#include "PathGeneration.h"

#include <MathConsts.h>
#include <CurvePerturbUtils.h>

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <random>
#include <stdexcept>

namespace twisty {
namespace PathGeneration {

    float CalculateMinimumArclength(
          twisty::PerturbUtils::BoundaryConditions boundaryConditions, uint32_t numSegmentsPerCurve)
    {
        float minArclength = 0.0f;
        // New Way
        {
            const uint32_t M = numSegmentsPerCurve;
            const glm::vec3 Xs = boundaryConditions.m_startPos;
            const glm::vec3 Xe = boundaryConditions.m_endPos;
            const glm::vec3 Ns = boundaryConditions.m_startDir;
            const glm::vec3 Ne = boundaryConditions.m_endDir;

            const float a = glm::dot(Ns + Ne, Ns + Ne) - ((M - 4.0f) * M + 4.0f);
            const float b = -2.0f * M * glm::dot(Ns + Ne, Xe - Xs);
            const float c = M * M * glm::dot(Xe - Xs, Xe - Xs);

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
    void ResolveTwoSegments(std::vector<glm::vec3> &pointList,
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

        const glm::vec3 &leftSegmentStart = pointList[leftSegmentStartIdx];
        const glm::vec3 &rightSegmentEnd = pointList[-rightSegmentEndIdx];

        const size_t finalPointIdx = leftSegmentStartIdx + 1;
        glm::vec3 &finalPoint = pointList[finalPointIdx];

        // Place segment exactly in the center
        const float d = glm::length(rightSegmentEnd - leftSegmentStart);
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
            glm::vec3 centerOffset = glm::vec3(std::sin(phi) * std::cos(theta),
                                                 std::sin(phi) * std::sin(theta), std::cos(phi))
                  * static_cast<float>(ds);
            finalPoint = leftSegmentStart + centerOffset;
            return;
        }

        // Ok, last case, phi is defined by the boundary of the problem. We also randomly rotate by theta
        const glm::vec3 x_p = (leftSegmentStart + rightSegmentEnd) * 0.5f;
        const glm::vec3 lineUnitDir = glm::normalize(rightSegmentEnd - leftSegmentStart);

        glm::vec3 otherCrossVec(1.0, 0.0, 0.0);
        if (abs(glm::dot(lineUnitDir, otherCrossVec)) >= 0.99) {
            otherCrossVec = glm::vec3(0.0, 1.0, 0.0);
        }

        const glm::vec3 normalToLine = glm::normalize(glm::cross(lineUnitDir, otherCrossVec));

        const float d_2 = d * 0.5f;

        float distanceOffLine = 0.0f;
        if (ds > d_2) {
            distanceOffLine = std::sqrt((ds * ds) - (d_2 * d_2));
        }
        glm::vec3 x_t = x_p + normalToLine * distanceOffLine;

        // Now rotate randomly theta amount around the axis.

        const float sinRotAngle = std::sin(theta / 2.0f);
        float quaternionRotation[4] = { std::cos(theta / 2.0f), lineUnitDir.x * sinRotAngle,
            lineUnitDir.y * sinRotAngle, lineUnitDir.z * sinRotAngle };

        glm::vec3 shiftedPoint = x_t - leftSegmentStart;
        // Rotate and stuff back in shifted point
        twisty::RotateVectorByQuaternion(quaternionRotation, &shiftedPoint[0]);
        // Update the point with the rotated version
        x_t = shiftedPoint + leftSegmentStart;

        finalPoint = x_t;
    };

    // Path Generation Helper Functions
    // Places two points
    void ResolveThreeSegments(std::vector<glm::vec3> &pointList,
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

        const glm::vec3 &leftSegmentStart = pointList[leftSegmentStartIdx];
        const glm::vec3 &rightSegmentEnd = pointList[rightSegmentEndIdx];

        const size_t firstPlacedPointIdx = leftSegmentStartIdx + 1;
        glm::vec3 &firstPlacedPoint = pointList[firstPlacedPointIdx];

        // Place segment exactly in the center
        const float d = glm::length(rightSegmentEnd - leftSegmentStart);

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
            glm::vec3 centerOffset = glm::vec3(std::sin(phi) * std::cos(theta),
                                                 std::sin(phi) * std::sin(theta), std::cos(phi))
                  * static_cast<float>(ds);
            firstPlacedPoint = leftSegmentStart + centerOffset;
            ResolveTwoSegments(pointList, firstPlacedPointIdx, rightSegmentEndIdx, ds, rng);
            return;
        }

        // Uniform dist
        std::uniform_real_distribution<float> uniformRandom(0.0f, 1.0f);

        // Z axis of new corrdinate frame
        const glm::vec3 zAxis = glm::normalize(rightSegmentEnd - leftSegmentStart);
        // Generate orthogonal basis vectors x axis and y axis
        glm::vec3 randomVector = glm::vec3(1.0f, 0.0f, 0.0f);
        if (std::abs(glm::dot(zAxis, randomVector)) > 0.999f) {
            randomVector = glm::vec3(0.0f, 1.0f, 0.0f);
        }
        const glm::vec3 xAxis = glm::normalize(glm::cross(zAxis, randomVector));
        const glm::vec3 yAxis = glm::normalize(glm::cross(zAxis, xAxis));

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

        const glm::vec3 firstPlacedSegmentDir
              = xAxis * static_cast<float>(std::sin(phi)) * static_cast<float>(std::cos(theta))
              + yAxis * static_cast<float>(std::sin(phi))
                    * static_cast<float>(std::sin(theta))
              + zAxis * static_cast<float>(std::cos(phi));

        firstPlacedPoint = leftSegmentStart + glm::normalize(firstPlacedSegmentDir) * static_cast<float>(ds);
        ResolveTwoSegments(pointList, firstPlacedPointIdx, rightSegmentEndIdx, ds, rng);
    };

    void ResolveEvenNumberOfSegments(const int numSegments, std::vector<glm::vec3> &pointList,
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

        const glm::vec3 &leftPoint = pointList[leftSegmentStartIdx];
        const glm::vec3 &rightPoint = pointList[rightSegmentEndIdx];

        const size_t centerPointIdx = leftSegmentStartIdx + numSegmentsPerSide;
        glm::vec3 &centerPoint = pointList[centerPointIdx];

        const double d = glm::length(rightPoint - leftPoint);

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

            const glm::vec3 zAxis = glm::normalize(rightPoint - leftPoint);
            // Generate orthogonal basis vectors x axis and y axis
            glm::vec3 randomVector = glm::vec3(1.0f, 0.0f, 0.0f);
            if (std::abs(glm::dot(zAxis, randomVector)) > 0.999f) {
                randomVector = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            const glm::vec3 xAxis = glm::normalize(glm::cross(zAxis, randomVector));
            const glm::vec3 yAxis = glm::normalize(glm::cross(zAxis, xAxis));

            glm::vec3 centerOffset = xAxis * static_cast<float>(std::sin(phi))
                        * static_cast<float>(std::cos(theta))
                  + yAxis * static_cast<float>(std::sin(phi))
                        * static_cast<float>(std::sin(theta))
                  + zAxis * static_cast<float>(std::cos(phi));
            centerOffset = centerOffset * static_cast<float>(sampledRadius);
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

        const glm::vec3 midPoint = 0.5f * (rightPoint + leftPoint);

        const double distToMidpoint = glm::length(midPoint - leftPoint);

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
            const glm::vec3 zAxis = glm::normalize(rightPoint - leftPoint);
            // Generate orthogonal basis vectors x axis and y axis
            glm::vec3 randomVector = glm::vec3(1.0f, 0.0f, 0.0f);
            if (std::abs(glm::dot(zAxis, randomVector)) > 0.999f) {
                randomVector = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            const glm::vec3 xAxis = glm::normalize(glm::cross(zAxis, randomVector));
            const glm::vec3 yAxis = glm::normalize(glm::cross(zAxis, xAxis));

            glm::vec3 centerOffset = xAxis * static_cast<float>(std::sin(phi))
                        * static_cast<float>(std::cos(theta))
                  + yAxis * static_cast<float>(std::sin(phi))
                        * static_cast<float>(std::sin(theta))
                  + zAxis * static_cast<float>(std::cos(phi));
            centerOffset = centerOffset * static_cast<float>(sampledRadius);
            pointList[centerPointIdx] = leftPoint + centerOffset;
            // Right half
        } else {
            const glm::vec3 zAxis = glm::normalize(rightPoint - leftPoint);
            // Generate orthogonal basis vectors x axis and y axis
            glm::vec3 randomVector = glm::vec3(1.0f, 0.0f, 0.0f);
            if (std::abs(glm::dot(zAxis, randomVector)) > 0.999f) {
                randomVector = glm::vec3(0.0f, 1.0f, 0.0f);
            }
            const glm::vec3 xAxis = glm::normalize(glm::cross(zAxis, randomVector));
            const glm::vec3 yAxis = glm::normalize(glm::cross(zAxis, xAxis));

            glm::vec3 centerOffset = xAxis * static_cast<float>(std::sin(phi))
                        * static_cast<float>(std::cos(theta))
                  + yAxis * static_cast<float>(std::sin(phi))
                        * static_cast<float>(std::sin(theta))
                  + zAxis * static_cast<float>(std::cos(phi)) * -1.0f;
            centerOffset = centerOffset * static_cast<float>(sampledRadius);
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
