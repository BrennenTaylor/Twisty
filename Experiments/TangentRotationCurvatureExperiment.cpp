#include <iostream>

#include "CurvePerturbUtils.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "PerturbUtils.h"

#include <FMath/FMath.h>

#include <cstdint>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Please call as: %s rotationOffLeftDegrees rotationOfAxisDegrees numStepsAroundAxis", argv[0]);
        return 1;
    }

    const double rotationOffLeftDegrees = std::stod(argv[1]);
    const double rotationOffLeftRadians = twisty::RadianFromDegree(rotationOffLeftDegrees);

    const double rotationOfAxisDegrees = std::stod(argv[2]);
    const double rotationOfAxisRadians = twisty::RadianFromDegree(rotationOfAxisDegrees);

    const uint32_t numStepsAroundAxis = std::stoi(argv[3]);

    const double ds = 12.0 / 200.0;

    const Farlor::Vector3 intoAxis(0.0, 0.0, 1.0);
    const Farlor::Vector3 upAxis(0.0, 1.0, 0.0);
    const Farlor::Vector3 left(1.0, 0.0, 0.0);

    Farlor::Vector3 rotationAxis = left;
    {
        float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        twisty::RotationMatrixAroundAxis(rotationOfAxisRadians, (float*)(&upAxis), rotationMatrix);
        twisty::RotateVectorByMatrix(rotationMatrix, (float*)(&rotationAxis));
    }

    Farlor::Vector3 baseRight = left;
    {
        float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        twisty::RotationMatrixAroundAxis(rotationOffLeftRadians, (float*)(&intoAxis), rotationMatrix);
        twisty::RotateVectorByMatrix(rotationMatrix, (float*)(&baseRight));
    }

    //std::cout << "Left: " << left << std::endl;
    //std::cout << "Right: " << baseRight << std::endl;
    //std::cout << "Rotation Axis: " << rotationAxis << std::endl;

    twisty::WeightingParameters weightingParameters;
    weightingParameters.mu = 0.1;
    weightingParameters.eps = 0.1;
    weightingParameters.numStepsInt = 2000;
    weightingParameters.minBound = 0.0;
    weightingParameters.maxBound = 10.0 / weightingParameters.eps;
    weightingParameters.numCurvatureSteps = 10000;
    // Lets give some absorbtion as well
    // Absorbtion 1/20 off the time
    weightingParameters.absorbtion = 0.05;
    // 1/5 scatter means one event every 5 units, thus 2 scattering events in the shortest
    // or 5 in the longest 100 unit path
    weightingParameters.scatter = 0.2;

    twisty::PathWeighting::WeightLookupTableIntegral lookupEvaluator(weightingParameters, ds);


    double stepSize = (2.0 * twisty::TwistyPi) / (numStepsAroundAxis - 1);
    //std::cout << "Angle, Curvature, Weight" << std::endl;
    for (uint32_t i = 0; i < numStepsAroundAxis; ++i)
    {
        double experimentRotationAngleRadians = i * stepSize;
        Farlor::Vector3 right = baseRight;

        float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        twisty::RotationMatrixAroundAxis(experimentRotationAngleRadians, (float*)(&rotationAxis), rotationMatrix);
        twisty::RotateVectorByMatrix(rotationMatrix, (float*)(&right));

        float curvature = twisty::PerturbUtils::CurvatureCalculation(left, right, ds);

        double weight = twisty::PathWeighting::TableLookupCurvature(curvature, lookupEvaluator);

        std::cout << experimentRotationAngleRadians << ", " << curvature << ", " << weight << std::endl;
    }



    return 0;
}