
#include "ExperimentRunnerCpu.h"
#include "CurveUtils.h"
#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"

#include <FMath/Vector3.h>

#include <fmt/format.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <random>

using namespace twisty;

constexpr bool DetailedPurturb = true;

enum class CurvePerturbMethod
{
    SimpleGeometry = 0,
    ComplexGeometry = 1,
    RootSolve = 2
};

constexpr CurvePerturbMethod g_CurvePerturbMethod = CurvePerturbMethod::SimpleGeometry;

std::unique_ptr<Curve> SimpleCurvePerturb(const Curve& curve, const Bootstrapper& bootstrapper, const ExperimentRunner::ExperimentParameters& params);
std::unique_ptr<Curve> ComplexCurvePerturb(const Curve& curve, const Bootstrapper& bootstrapper, const ExperimentRunner::ExperimentParameters& params);

int main()
{
    // Bootstrap method
    const Range defaultBounds = { -1.0f, 1.0f };
    const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
    const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
    RayGeometry rayEmitter(emitterStart, emitterDir);

    const Farlor::Vector3 recieverPos{ 10.0f, 0.0f, 0.0f };
    const Farlor::Vector3 recieverDir{ 0.0, -1.0f, 0.0f };

    RayGeometry rayReciever(recieverPos, recieverDir);


    const Range arclengthRange = { 25.0f, 25.0f };

    GeometryBootstrapper bootstrapper(rayEmitter, rayReciever, arclengthRange, 0);

    ExperimentRunner::ExperimentParameters experimentParams;
    experimentParams.exportGeneratedCurves = false;
    experimentParams.numSegmentsPerCurve = 200;
    experimentParams.maxPathBatchSize = 100000;
    experimentParams.maximumBootstrapCurveError = 0.5f;
    experimentParams.curvePurturbSeed = 0;

    std::unique_ptr<twisty::Curve> upInitialCurve = nullptr;

    bool successfulGen = false;
    while (!successfulGen)
    {
        upInitialCurve = bootstrapper.CreateCurve(experimentParams.numSegmentsPerCurve);
        if (!upInitialCurve)
        {
            printf("Failed to create bootstrap curve.\n");
            return false;
        }

        // Once we have a curve, we know arclength.
        // Thus, we can setup the min and max curvatures
        float ds = upInitialCurve->m_arclength / upInitialCurve->m_numSegments;

        // Lets also get the error of the initial curve, just to know
        float curveError = CurveUtils::CalculateCurveError(*upInitialCurve);
        std::cout << "Seed curve error: " << curveError << std::endl;

        if (curveError < experimentParams.maximumBootstrapCurveError)
        {
            successfulGen = true;
        }
    }

    std::unique_ptr<Curve> newCurve(nullptr);
    // Actually do the purturbation
    if (g_CurvePerturbMethod == CurvePerturbMethod::SimpleGeometry)
    {
        newCurve = SimpleCurvePerturb(*upInitialCurve, bootstrapper, experimentParams);
    }
    else if (g_CurvePerturbMethod == CurvePerturbMethod::ComplexGeometry)
    {
        newCurve = ComplexCurvePerturb(*upInitialCurve, bootstrapper, experimentParams);
    }
    else
    {
        assert(false);
    }
}

// Assumes line vector is unit direction
Farlor::Vector3 ProjectVectorToLine(Farlor::Vector3 vecToProject, Farlor::Vector3 unitDir)
{
    unitDir = unitDir.Normalized();
    return vecToProject.Dot(unitDir) * unitDir;
}

float AngleClamp(float angle)
{
    float clampedAngle = angle;
    while (clampedAngle < 0.0)
    {
        clampedAngle += 2 * TwistyPi;
    }
    
    while (clampedAngle > (2.0 * TwistyPi))
    {
        clampedAngle -= (2.0 * TwistyPi);
    }
    return clampedAngle;
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

// Assumes radians
float ClampZeroToPi(float angle)
{
    while (angle < 0)
    {
        angle += TwistyPi;
    }

    while (angle > TwistyPi)
    {
        angle -= TwistyPi;
    }
    return angle;
}

// Assumes radians
float ClampZeroToTwoPi(float angle)
{
    while (angle < 0)
    {
        angle += (2.0f * TwistyPi);
    }

    while (angle > (2.0f * TwistyPi))
    {
        angle -= (2.0f * TwistyPi);
    }
    return angle;
}

float CalculateCurvatureGivenDistanceAndTorsion(float distance, float torsionAngle, float aTanComp, float aNormComp, float aBinormComp, float bLength)
{
    float x = (distance * distance) - (aTanComp * aTanComp) - (aNormComp * aNormComp) - (aBinormComp * aBinormComp);
    float y = (x - (bLength * bLength)) / (-2.0f * bLength);

    float a = aTanComp;
    float b = (aNormComp * cos(torsionAngle) + aBinormComp * sin(torsionAngle));

    const float asinPiece = asin(y / sqrt(a * a + b * b));
    const float atanPiece = atan(a / b);
    float curvatureAngle = asinPiece - atanPiece;

    return ClampZeroToPi(curvatureAngle);
}

float CalculateTorsionGivenDistanceAndCurvature(float distance, float curvatureAngle, float aTanComp, float aNormComp, float aBinormComp, float bLength)
{
    const float bTanComp = bLength * cos(curvatureAngle);

    const float x = (distance * distance) - (aTanComp * aTanComp) - (bTanComp * bTanComp) + (2.0f * aTanComp * bTanComp) - (aNormComp * aNormComp) - (aBinormComp * aBinormComp);
    const float y = (x - (bLength * bLength * sin(curvatureAngle) * sin(curvatureAngle))) / (-2.0f * bLength * sin(curvatureAngle));

    const float asinPiece = asin(y / sqrt(aNormComp * aNormComp + aBinormComp * aBinormComp));
    const float atanPiece = atan(aNormComp / aBinormComp);
    float theta = asinPiece - atanPiece;

    if (DetailedPurturb)
    {
        std::cout << "\tCenter theta calculation" << std::endl;
        std::cout << "\t\tX: " << x << std::endl;
        std::cout << "\t\tY: " << y << std::endl;
        std::cout << "\t\tasin Piece: " << asinPiece << std::endl;
        std::cout << "\t\tatan Piece: " << atanPiece << std::endl;
        std::cout << "\t\tCenter theta: " << theta << std::endl;
    }

    return ClampZeroToTwoPi(theta);
}

// Requires that the left rigid body, A, is non moving and thus already known
float CalculateDistanceGivenCurvatureAndTorsion(float curvatureAngle, float torsionAngle, float aTanComp, float aNormComp, float aBinormComp, float bLength)
{
    float bTanComp = bLength * cos(curvatureAngle);
    float bNormComp = bLength * sin(curvatureAngle) * cos(torsionAngle);
    float bBinormComp = bLength * sin(curvatureAngle) * sin(torsionAngle);

    float distance = (aTanComp - bTanComp) * (aTanComp - bTanComp) + (aNormComp - bNormComp) * (aNormComp - bNormComp) + (aBinormComp - bBinormComp) * (aBinormComp - bBinormComp);
    distance = sqrt(distance);
    return distance;
}

Farlor::Matrix3x3 RotationMatrixForAToB(Farlor::Vector3 a, Farlor::Vector3 b)
{
    // Lets get the normalized versions
    Farlor::Vector3 aNorm = a;
    Farlor::Vector3 bNorm = b;

    Farlor::Vector3 v = aNorm.Cross(bNorm);
    float s = v.Magnitude();
    float c = aNorm.Dot(bNorm);

    Farlor::Matrix3x3 v_x(
        Farlor::Vector3(0.0f, -v.z, v.y),
        Farlor::Vector3(v.z, 0.0f, -v.x),
        Farlor::Vector3(-v.y, v.x, 0.0f)
    );

    Farlor::Matrix3x3 v_x_2 = v_x * v_x;

    Farlor::Matrix3x3 rotation = Farlor::Matrix3x3::s_Identity + v_x + ((1.0f - c) / s) * v_x_2;
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

std::unique_ptr<Curve> SimpleCurvePerturb(const Curve& curve, const Bootstrapper& bootstrapper, const ExperimentRunner::ExperimentParameters& params)
{
    if (DetailedPurturb)
    {
        std::cout << "Begin Purturb --------" << std::endl;
    }

    uint32_t seed = params.curvePurturbSeed;
    if (seed == 0)
    {
        seed = time(0);
    }
    std::cout << "Purturb seed used: " << seed << std::endl;
    std::mt19937 m_rng(seed);

    std::unique_ptr<Curve> upNewCurve = std::make_unique<Curve>(curve);

    std::uniform_int_distribution<int> leftSegmentIndexUniformDist(0, upNewCurve->m_numSegments - 2); // uniform, unbiased
    int32_t leftIndex = leftSegmentIndexUniformDist(m_rng);
    std::uniform_int_distribution<int> rightSegmentIndexUniformDist(leftIndex + 1, upNewCurve->m_numSegments - 1); // uniform, unbiased
    int32_t rightIndex = rightSegmentIndexUniformDist(m_rng);

    assert(leftIndex < rightIndex);

    if (DetailedPurturb)
    {
        std::cout << "\tLeft Index: " << leftIndex << std::endl;
        std::cout << "\tRight Index: " << rightIndex << std::endl;
    }

    // 0 - 2 PI uniform distribution
    std::uniform_real_distribution<float> zeroToTwoPiUniformDist(0.0f, 2 * TwistyPi);

    // End targets of purturbation
    Farlor::Vector3 targetN = bootstrapper.GetTargetNormal();
    Farlor::Vector3 targetP = bootstrapper.GetTargetPosition();


    /** This is where the fun begins **/
    std::vector<Farlor::Vector3> positions;
    std::vector<Farlor::Matrix3x3> frames;
    upNewCurve->ReconstructCurvePositionsAndFramesFirstOrder(positions, frames);

    // We need two frames for each segment to get the new curvature and torsion.
    // we need the frame left of the segment, as well as the frame right of the segment.

    Farlor::Vector3 leftPos = positions[leftIndex + 1];
    Farlor::Vector3 rightPos = positions[rightIndex + 1];

    Farlor::Vector3 axisOfRotation = (rightPos - leftPos).Normalized();

    float randomAngle = zeroToTwoPiUniformDist(m_rng);
    Farlor::Matrix3x3 rotationMatrix = RotationMatrixAroundAxis(randomAngle, axisOfRotation);

    Farlor::Matrix3x3 leftSegmentLeftMatrix = frames[leftIndex];
    Farlor::Matrix3x3 leftSegmentRightMatrix = frames[leftIndex + 1];

    Farlor::Matrix3x3 rightSegmentLeftMatrix = frames[rightIndex];
    Farlor::Matrix3x3 rightSegmentRightMatrix = frames[rightIndex + 1];

    // In the case leftIndex + 1 = rightIndex, these will be the same.
    Farlor::Matrix3x3 rotatedLeftMatrix = rotationMatrix * leftSegmentRightMatrix;
    Farlor::Matrix3x3 rotatedRightMatrix = rotationMatrix * rightSegmentLeftMatrix;

    if (DetailedPurturb)
    {
        std::cout << "\tPrevious left values" << std::endl;
        std::cout << "\t\tPrevious left curvature: " << upNewCurve->m_segments[leftIndex].m_curvature << std::endl;
        std::cout << "\t\tPrevious left torsion: " << upNewCurve->m_segments[leftIndex].m_torsion << std::endl;

        std::cout << "\tPrevious right values" << std::endl;
        std::cout << "\t\tPrevious right curvature: " << upNewCurve->m_segments[rightIndex].m_curvature << std::endl;
        std::cout << "\t\tPrevious right torsion: " << upNewCurve->m_segments[rightIndex].m_torsion << std::endl;
    }

    auto newLeftCurvatureTorsionVals = CurvatureAndTorsionBetweenTwoFrames(leftSegmentLeftMatrix, rotatedLeftMatrix, upNewCurve->m_segments[leftIndex].m_length);
    upNewCurve->m_segments[leftIndex].m_curvature = newLeftCurvatureTorsionVals.first;
    upNewCurve->m_segments[leftIndex].m_torsion = newLeftCurvatureTorsionVals.second;

    auto newRightCurvatureTorsionVals = CurvatureAndTorsionBetweenTwoFrames(rotatedRightMatrix, rightSegmentRightMatrix, upNewCurve->m_segments[rightIndex].m_length);
    upNewCurve->m_segments[rightIndex].m_curvature = newRightCurvatureTorsionVals.first;
    upNewCurve->m_segments[rightIndex].m_torsion = newRightCurvatureTorsionVals.second;


    if (DetailedPurturb)
    {
        std::cout << "\tNew left values" << std::endl;
        std::cout << "\t\tNew left curvature: " << upNewCurve->m_segments[leftIndex].m_curvature << std::endl;
        std::cout << "\t\tNew left torsion: " << upNewCurve->m_segments[leftIndex].m_torsion << std::endl;

        std::cout << "\tNew right values" << std::endl;
        std::cout << "\t\tNew right curvature: " << upNewCurve->m_segments[rightIndex].m_curvature << std::endl;
        std::cout << "\t\tNew right torsion: " << upNewCurve->m_segments[rightIndex].m_torsion << std::endl;
    }


    // Calculate error of new curvature
    float errorThreshold = 1.0f;
    float newCurveError = CurveUtils::CalculateCurveError(*upNewCurve);
    Farlor::Vector3 finalPos(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 finalDir(0.0f, 0.0f, 0.0f);
    upNewCurve->CalculateFinalPosAndTangent(finalPos, finalDir);

    if (DetailedPurturb)
    {
        //std::cout << "\tFinal Pos: " << finalPos << std::endl;
        //std::cout << "\tFinal Dir: " << finalDir << std::endl;

        //// Basic difference in pos
        //std::cout << "\tdx_error: " << (finalPos.x - targetP.x) << std::endl;
        //std::cout << "\tdy_error: " << (finalPos.y - targetP.y) << std::endl;
        //std::cout << "\tdz_error: " << (finalPos.z - targetP.z) << std::endl;
        //// L1 norm of tangents
        //std::cout << "\tn_l1_error: " << (finalDir.x - targetN.x) + (finalDir.y - targetN.y) + (finalDir.z - targetN.z) << std::endl;
        //// L2 norm of tangents
        //std::cout << "\tn_l2_error: " << (finalDir - targetN).Magnitude() << std::endl;
        //std::cout << "\tCurve Error: " << newCurveError << std::endl;
    }

    // If we have too much error, we dont accept
    if (newCurveError > errorThreshold)
    {
        if (DetailedPurturb)
        {
            // We failed to achieve the acceptance criteria in the specified number of iterations
            std::cout << "Failed Puturb: " << newCurveError << std::endl;
        }

        return {};
    }

    if (DetailedPurturb)
    {
        std::cout << "End Purturb --------" << std::endl;
    }

    return upNewCurve;
}

std::unique_ptr<Curve> ComplexCurvePerturb(const Curve& curve, const Bootstrapper& bootstrapper, const ExperimentRunner::ExperimentParameters& params)
{
    // Initial Test
    {
        Farlor::Vector3 a(0.0f, 0.0f, 1.0f);
        Farlor::Vector3 b(1.0f, 0.0f, 0.0f);
        Farlor::Matrix3x3 rotation = RotationMatrixForAToB(a, b);
        Farlor::Vector3 c = rotation * a;
        std::cout << "A: " << a << std::endl;
        std::cout << "B: " << b << std::endl;
        std::cout << "C: " << c << std::endl;
    }

    if (DetailedPurturb)
    {
        std::cout << "Begin Purturb --------" << std::endl;
    }

    uint32_t seed = params.curvePurturbSeed;
    if (seed == 0)
    {
        seed = time(0);
    }
    std::cout << "Purturb seed used: " << seed << std::endl;
    std::mt19937 m_rng(seed);

    std::unique_ptr<Curve> upNewCurve = std::make_unique<Curve>(curve);

    std::uniform_int_distribution<int> midGen(1, upNewCurve->m_numSegments - 2); // uniform, unbiased
    int32_t middleIndex = midGen(m_rng);
    std::uniform_int_distribution<int> leftGen(0, middleIndex - 1); // uniform, unbiased
    std::uniform_int_distribution<int> rightGen(middleIndex + 1, upNewCurve->m_numSegments - 1); // uniform, unbiased
    int32_t leftIndex = leftGen(m_rng);
    int32_t rightIndex = rightGen(m_rng);

    assert(leftIndex < middleIndex);
    assert(middleIndex < rightIndex);

    if (DetailedPurturb)
    {
        std::cout << "\tLeft Index: " << leftIndex << std::endl;
        std::cout << "\tMid Index: " << middleIndex << std::endl;
        std::cout << "\tRight Index: " << rightIndex << std::endl;
    }

    // Modify the curvature of the middle selection
    std::uniform_real_distribution<float> curvatureAngleGen(0.0f, 2 * TwistyPi);

    // End targets of purturbation
    Farlor::Vector3 targetN = bootstrapper.GetTargetNormal();
    Farlor::Vector3 targetP = bootstrapper.GetTargetPosition();


    /** This is where the fun begins **/
    std::vector<Farlor::Vector3> positions;
    std::vector<Farlor::Matrix3x3> frames;
    upNewCurve->ReconstructCurvePositionsAndFramesFirstOrder(positions, frames);

    if (DetailedPurturb)
    {
        std::cout << "\tPositions size: " << positions.size() << std::endl;
        std::cout << "\tFrames size: " << frames.size() << std::endl;
    }

    Farlor::Vector3 rb0_start = positions[0];
    Farlor::Vector3 rb0_end = positions[leftIndex + 1];

    Farlor::Vector3 rb1_start = positions[leftIndex + 1];
    Farlor::Vector3 rb1_end = positions[middleIndex + 1];

    Farlor::Vector3 rb2_start = positions[middleIndex + 1];
    Farlor::Vector3 rb2_end = positions[rightIndex + 1];

    Farlor::Vector3 rb3_start = positions[rightIndex + 1];
    Farlor::Vector3 rb3_end = positions[upNewCurve->m_numSegments];

    if (DetailedPurturb)
    {
        std::cout << "\tRigid Body Information" << std::endl;

        // RB0
        std::cout << "\t\tRigid Body 0 start: " << rb0_start << std::endl;
        std::cout << "\t\tRigid Body 0 end: " << rb0_end << std::endl;
        std::cout << "\t\tRigid Body 0 length: " << (rb0_end - rb0_start).Magnitude() << std::endl;

        // RB1
        std::cout << "\t\tRigid Body 1 start: " << rb1_start << std::endl;
        std::cout << "\t\tRigid Body 1 end: " << rb1_end << std::endl;
        std::cout << "\t\tRigid Body 1 length: " << (rb1_end - rb1_start).Magnitude() << std::endl;

        // RB2
        std::cout << "\t\tRigid Body 2 start: " << rb2_start << std::endl;
        std::cout << "\t\tRigid Body 2 end: " << rb2_end << std::endl;
        std::cout << "\t\tRigid Body 2 length: " << (rb2_end - rb2_start).Magnitude() << std::endl;

        // RB3
        std::cout << "\t\tRigid Body 3 start: " << rb3_start << std::endl;
        std::cout << "\t\tRigid Body 3 end: " << rb3_end << std::endl;
        std::cout << "\t\tRigid Body 3 length: " << (rb3_end - rb3_start).Magnitude() << std::endl;
    }

    if (DetailedPurturb)
    {
        std::cout << "\tCenter Purturb Step" << std::endl;
    }

    // Distance which needs to fit between first and last rigid bodies.
    // World space
    float centerDistance = (rb3_start - rb0_end).Magnitude();
    if (DetailedPurturb)
    {
        std::cout << "\t\tCenter Distance: " << centerDistance << std::endl;
    }

    // We want to focus on the center piece first.
    // This means the calculation of curvature and torsion angles for RB1 and RB2's connection

    // Get the axis we are working with
    // All are in world space
    const Farlor::Matrix3x3 refSegFrame = frames[middleIndex];
    const Farlor::Vector3 refPoint = positions[middleIndex + 1];

    assert(refSegFrame.m_rows[0].Magnitude() == 1.0f);
    assert(refSegFrame.m_rows[1].Magnitude() == 1.0f);
    assert(refSegFrame.m_rows[2].Magnitude() == 1.0f);

    if (DetailedPurturb)
    {
        std::cout << "\t\tReference Frame: " << refSegFrame.m_rows[0] << std::endl;
        std::cout << "\t\t                 " << refSegFrame.m_rows[1] << std::endl;
        std::cout << "\t\t                 " << refSegFrame.m_rows[2] << std::endl;

        std::cout << "\t\tReference Frame Origin : " << refPoint << std::endl;
    }

    const Farlor::Vector3 refAxis = refSegFrame.m_rows[0];
    
    // Some information is fixed.
    // Vector from rb1_end
    Farlor::Vector3 A_vec_WS = rb1_start - refPoint;
    float A_vec_WS_length = A_vec_WS.Magnitude();


    // Project A_vec_WS onto each ref frame axis to get components relative to ref frame
    float A_vec_RFS_tan = A_vec_WS.Dot(refSegFrame.m_rows[0]);
    float A_vec_RFS_norm = A_vec_WS.Dot(refSegFrame.m_rows[1]);
    float A_vec_RFS_binorm = A_vec_WS.Dot(refSegFrame.m_rows[2]);

    Farlor::Vector3 A_vec_RFS(A_vec_RFS_tan, A_vec_RFS_norm, A_vec_RFS_binorm);
    float A_vec_RFS_length = A_vec_RFS.Magnitude();

    // A curvature angle
    float aCurvatureAngle = ClampZeroToTwoPi(atan2(A_vec_RFS_norm, A_vec_RFS_tan));

    // A torsion angle
    float aTorsionAngle = ClampZeroToTwoPi(atan2(A_vec_RFS_binorm, A_vec_RFS_norm));


    // A information
    if (DetailedPurturb)
    {
        std::cout << "\tA body information" << std::endl;
        std::cout << "\t\tA Vector WS: " << A_vec_WS << std::endl;
        std::cout << "\t\tA Vector WS length: " << A_vec_WS_length << std::endl;
        std::cout << "\t\tA Vector RFS: " << A_vec_RFS << std::endl;
        std::cout << "\t\tA Vector RFS length: " << A_vec_RFS_length << std::endl;
        std::cout << "\t\tA curvature angle: " << aCurvatureAngle << std::endl;
        std::cout << "\t\tA torsion angle: " << aTorsionAngle << std::endl;
    }

    // Lets test it.
    // To do this, we find a unit length vector along the axis in the neg direction with the legnth
    {
        Farlor::Vector3 startVector = refSegFrame.m_rows[0] * A_vec_RFS_length;

        Farlor::Matrix3x3 curvatureRotation = RotationMatrixAroundAxis(aCurvatureAngle, refSegFrame.m_rows[2]);
        Farlor::Matrix3x3 torsionRotation = RotationMatrixAroundAxis(aTorsionAngle, refSegFrame.m_rows[0]);

        Farlor::Matrix3x3 rotationMat = torsionRotation * curvatureRotation;
        Farlor::Vector3 rotatedVector = rotationMat * startVector;

        Farlor::Vector3 cartisianCoordRF_a(
            A_vec_WS_length * cos(aCurvatureAngle),
            A_vec_WS_length * sin(aCurvatureAngle) * cos(aTorsionAngle),
            A_vec_WS_length * sin(aCurvatureAngle) * sin(aTorsionAngle)
        );

        if (DetailedPurturb)
        {
            std::cout << "\tTest Information" << std::endl;
            std::cout << "\t\tA Vector WS: " << A_vec_WS << std::endl;
            std::cout << "\t\tA Vector WS length: " << A_vec_WS_length << std::endl;
            std::cout << "\t\tTest Vector: " << rotatedVector << std::endl;
            std::cout << "\t\tTest vector length: " << rotatedVector.Magnitude() << std::endl;
            std::cout << "\t\tCartisian Conv Test, A Ref Frame: " << cartisianCoordRF_a << std::endl;
        }
    }

    // B rigid body length
// World space starting vector
    Farlor::Vector3 B_vec_WS = rb2_end - refPoint;
    float B_vec_WS_length = B_vec_WS.Magnitude();

    // Test rotating to the tangent axis
    if (DetailedPurturb)
    {
        Farlor::Vector3 rotateTo = refSegFrame.m_rows[0].Normalized();
        Farlor::Matrix3x3 rotation = RotationMatrixForAToB(B_vec_WS.Normalized(), rotateTo.Normalized());
        Farlor::Vector3 rotated = rotation * B_vec_WS.Normalized();
        std::cout << "\tRotate Vector Test" << std::endl;
        std::cout << "\t\tRotate To: " << rotateTo << std::endl;
        std::cout << "\t\tRotatee: " << B_vec_WS.Normalized() << std::endl;
        std::cout << "\t\tRotated: " << rotated << std::endl;

    }

    std::cout << "\tProjected A tests" << std::endl;

    const float aTanComp = A_vec_RFS_length * cos(aCurvatureAngle);
    const float aNormComp = A_vec_RFS_length * sin(aCurvatureAngle) * cos(aTorsionAngle);
    const float aBinormComp = A_vec_RFS_length * sin(aCurvatureAngle) * sin(aTorsionAngle);
    
    // Project A_vec_WS onto each ref frame axis to get components relative to ref frame
    float B_vec_tanComp = B_vec_WS.Dot(refSegFrame.m_rows[0]);
    float B_vec_normComp = B_vec_WS.Dot(refSegFrame.m_rows[1]);
    float B_vec_binormComp = B_vec_WS.Dot(refSegFrame.m_rows[2]);

    // Reference frame starting vector
    Farlor::Vector3 B_vec_RFS(B_vec_tanComp, B_vec_normComp, B_vec_binormComp);
    float B_vec_RFS_length = B_vec_RFS.Magnitude();

    // Theta angles of minimum and maximum distance
    float minTheta = ClampZeroToTwoPi(aTorsionAngle);
    float maxTheta = ClampZeroToTwoPi(minTheta + TwistyPi);
    
    std::cout << "\t\tminAngle: " << minTheta << std::endl;
    std::cout << "\t\tmaxAngle: " << maxTheta << std::endl;


    // Now that we know these, lets solve for the curvature range.
    float minCurvatureAngle = CalculateCurvatureGivenDistanceAndTorsion(centerDistance, maxTheta, aTanComp, aNormComp, aBinormComp, B_vec_RFS_length);
    float maxCurvatureAngle = CalculateCurvatureGivenDistanceAndTorsion(centerDistance, minTheta, aTanComp, aNormComp, aBinormComp, B_vec_RFS_length);

    minCurvatureAngle = ClampZeroToPi(minCurvatureAngle);
    maxCurvatureAngle = ClampZeroToPi(maxCurvatureAngle);

    std::uniform_real_distribution<float> curvatureAngleDist(minCurvatureAngle, maxCurvatureAngle);
    float randomCurvatureAngle = curvatureAngleDist(m_rng);

    if (DetailedPurturb)
    {
        std::cout << "\tCurvature angle range calculation" << std::endl;
        std::cout << "\t\tMin Curvature Angle: " << minCurvatureAngle << std::endl;
        std::cout << "\t\tMax Curvature Angle: " << maxCurvatureAngle << std::endl;
        std::cout << "\t\tRandom Curvature Angle: " << randomCurvatureAngle << std::endl;
    }

    // For now, we dont change it
    float bCurvatureAngle = randomCurvatureAngle;// (75.0f / 180.0f)* TwistyPi;

    // Going into these calculations, we need a curvature value
    // Torsions are hardcoded by "min, max"
    // We also need 3 known values on either side which form the right triangles.

    // NOTE: Only For this given curvature angle! This is important

    if (DetailedPurturb)
    {
        std::cout << "\tDistance calculations" << std::endl;
    }

    float dMax = CalculateDistanceGivenCurvatureAndTorsion(bCurvatureAngle, maxTheta, aTanComp, aNormComp, aBinormComp, B_vec_RFS_length);
    float dMin = CalculateDistanceGivenCurvatureAndTorsion(bCurvatureAngle, minTheta, aTanComp, aNormComp, aBinormComp, B_vec_RFS_length);
    
    if (DetailedPurturb)
    {
        std::cout << "\t\tMin D: " << dMin << std::endl;
        std::cout << "\t\tMax D: " << dMax << std::endl;
        std::cout << "\t\tTarget D: " << centerDistance << std::endl;
    }

    if (centerDistance < dMin || centerDistance > dMax)
    {
        std::cout << "\tBad curvature, no possible solution" << std::endl;
        return {};
    }

    float bTorsionAngle = CalculateTorsionGivenDistanceAndCurvature(centerDistance, bCurvatureAngle, aTanComp, aNormComp, aBinormComp, B_vec_RFS_length);

    Farlor::Vector3 b_vec_REF(0.0f, 0.0f, 0.0f);
    // Use curvature and torsion to get a B vector
    {
        Farlor::Vector3 startVector = refSegFrame.m_rows[0] * B_vec_RFS_length;

        Farlor::Matrix3x3 curvatureRotation = RotationMatrixAroundAxis(bCurvatureAngle, refSegFrame.m_rows[2]);
        Farlor::Matrix3x3 torsionRotation = RotationMatrixAroundAxis(bTorsionAngle, refSegFrame.m_rows[0]);

        Farlor::Matrix3x3 rotationMat = torsionRotation * curvatureRotation;
        Farlor::Vector3 rotatedVector = rotationMat * startVector;

        Farlor::Vector3 cartisianCoordRF_b(
            B_vec_WS_length * cos(bCurvatureAngle),
            B_vec_WS_length * sin(bCurvatureAngle) * cos(bTorsionAngle),
            B_vec_WS_length * sin(bCurvatureAngle) * sin(bTorsionAngle)
        );

        b_vec_REF = cartisianCoordRF_b;

        if (DetailedPurturb)
        {
            std::cout << "\tTest Information" << std::endl;
            std::cout << "\t\tTest Vector: " << rotatedVector << std::endl;
            std::cout << "\t\tTest vector length: " << rotatedVector.Magnitude() << std::endl;
        }
    }

    float testDistance = (aTanComp - b_vec_REF.x) * (aTanComp - b_vec_REF.x) + (aNormComp - b_vec_REF.y) * (aNormComp - b_vec_REF.y) + (aBinormComp - b_vec_REF.z) * (aBinormComp - b_vec_REF.z);
    testDistance = sqrt(testDistance);

    std::cout << "Test Distance: " << testDistance << std::endl;
    std::cout << "Desired Distance: " << centerDistance << std::endl;


    // Lets test by rotating the vector to bend
    //Farlor::Vector3 vectorToBend = refAxis * B_vec_WS_length;

    //

    //Farlor::Matrix3x3 curvatureRotation = RotationMatrixAroundAxis(curvatureAngle, refSegFrame.m_rows[2].Normalized());
    //Farlor::Matrix3x3 torsionRotation = RotationMatrixAroundAxis(theta, refSegFrame.m_rows[0].Normalized());

    //Farlor::Vector3 rotatedVector = (curvatureRotation * torsionRotation) * vectorToBend;


    //float testD = (A_vec_WS - rotatedVector).Magnitude();

    //std::cout << "saVec: " << A_vec_WS << std::endl;
    //std::cout << "rotatedVector: " << rotatedVector << std::endl;
    //std::cout << "testD: " << testD << std::endl;


    // Calculate error of new curvature
    float errorThreshold = 1.0f;
    float newCurveError = CurveUtils::CalculateCurveError(*upNewCurve);
    Farlor::Vector3 finalPos(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 finalDir(0.0f, 0.0f, 0.0f);
    upNewCurve->CalculateFinalPosAndTangent(finalPos, finalDir);

    if (DetailedPurturb)
    {
        //std::cout << "\tFinal Pos: " << finalPos << std::endl;
        //std::cout << "\tFinal Dir: " << finalDir << std::endl;

        //// Basic difference in pos
        //std::cout << "\tdx_error: " << (finalPos.x - targetP.x) << std::endl;
        //std::cout << "\tdy_error: " << (finalPos.y - targetP.y) << std::endl;
        //std::cout << "\tdz_error: " << (finalPos.z - targetP.z) << std::endl;
        //// L1 norm of tangents
        //std::cout << "\tn_l1_error: " << (finalDir.x - targetN.x) + (finalDir.y - targetN.y) + (finalDir.z - targetN.z) << std::endl;
        //// L2 norm of tangents
        //std::cout << "\tn_l2_error: " << (finalDir - targetN).Magnitude() << std::endl;
        //std::cout << "\tCurve Error: " << newCurveError << std::endl;
    }

    // If we have too much error, we dont accept
    if (newCurveError > errorThreshold)
    {
        if (DetailedPurturb)
        {
            // We failed to achieve the acceptance criteria in the specified number of iterations
            std::cout << "Failed Puturb: " << newCurveError << std::endl;
        }

        return {};
    }

    if (DetailedPurturb)
    {
        std::cout << "End Purturb --------" << std::endl;
    }

    return upNewCurve;
}