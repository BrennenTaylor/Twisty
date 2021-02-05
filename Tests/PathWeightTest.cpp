#include <PathWeightUtils.h>

#include <assert.h>
#include <string>
#include <sstream>
#include <fstream>

bool ThresholdFloatEquality(float first, float second, float epsilon)
{
    return abs(first - second) <= epsilon;
}

int main(int argc, char* argv[])
{
    if (argc < 6)
    {
        std::cout << "Call as: " << argv[0] << " mu epsilon bds maxKds pathLength" << std::endl;
        return 1;
    }


    //ExperimentRunner::ExperimentParameters experimentParams;
    //experimentParams.numPathsInExperiment = numPathsToGenerate;
    //experimentParams.numPathsToSkip = numPathsToSkip;
    //experimentParams.exportGeneratedCurves = true;
    //experimentParams.experimentName = experimentName;
    //experimentParams.numSegmentsPerCurve = 200;
    //experimentParams.maximumBootstrapCurveError = 0.5f;
    //experimentParams.curvePerturbMethod = ExperimentRunner::CurvePerturbMethod::SimpleGeometry;
    //experimentParams.curvePurturbSeed = perturbSeed;
    //experimentParams.rotateInitialSeedCurveRadians = 0.0f;

    //experimentParams.weightingParameters.mu = 0.1;
    //experimentParams.weightingParameters.eps = 0.1;
    //experimentParams.weightingParameters.numStepsInt = 20000;
    //experimentParams.weightingParameters.minBound = 0.0;
    //experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;
    //// Lets give some absorbtion as well
    //// Absorbtion 1/20 off the time
    //experimentParams.weightingParameters.absorbtion = 0.05;
    //// 1/5 scatter means one event every 5 units, thus 2 scattering events in the shortest
    //// or 5 in the longest 100 unit path
    //experimentParams.weightingParameters.scatter = 0.2;

    const uint32_t numSegments = 200;

    twisty::WeightingParameters weightParams;

    weightParams.mu = std::stod(argv[1]);
    weightParams.eps = std::stod(argv[2]);
    const double bds = std::stod(argv[3]);
    const double maxKds = std::stod(argv[4]);
    const double pathLength = std::stod(argv[5]);
    const double ds = pathLength / numSegments;
    weightParams.numStepsInt = 2000;
    weightParams.minBound = 0.0;
    weightParams.maxBound = 10.0 / weightParams.eps;
    const double minCurvature = 0.0;
    const double maxCurvature = (maxKds / ds);
    weightParams.numCurvatureSteps = 10000;
    weightParams.absorbtion = 0.05f;
    weightParams.scatter = bds / ds;

    std::cout << "ds: " << ds << std::endl;
    std::cout << "Scattering: " << weightParams.scatter << std::endl;
    std::cout << "Absorbtion: " << weightParams.absorbtion << std::endl;


    /*
        double ds, double mu, uint32_t numStepsInt, double minBound, double maxBound, double eps,
        double minCurvature, double maxCurvature, uint32_t numCurvatureSteps, double scattering
    */
    twisty::PathWeighting::WeightLookupTableIntegral weightTableIntegral(weightParams, ds);

    {
        std::stringstream outFilename;
        outFilename << "PathWeightTest_Out_" << weightParams.mu;
        outFilename << "_" << weightParams.eps;
        outFilename << "_" << bds;
        outFilename << "_" << maxKds;
        outFilename << "_" << weightParams.numStepsInt;
        outFilename << ".pwt";

        std::ofstream outFile(outFilename.str());

        const uint32_t numExperimentSteps = 1000;

        // TODO/NOTE: Curvature shouldnt be negative... right?
        const float minCurvature = 0.0f;
        const float maxCurvature = maxKds / ds;

        // For experiment, lets say 200 node path of arclength 10
        float dk = (maxCurvature - minCurvature) / numExperimentSteps;

        outFile << "Curvatures:" << std::endl;
        for (uint32_t i = 0; i < numExperimentSteps; ++i)
        {
            float currentCurvature = minCurvature + dk * i;
            outFile << currentCurvature * ds << std::endl;
        }

        outFile << "Weights:" << std::endl;
        for (uint32_t i = 0; i < numExperimentSteps; ++i)
        {
            float currentCurvature = minCurvature + dk * i;

            // Currently, this should probably be the start (or end) position of the segment.

            double segWeight = weightTableIntegral.Eval(weightParams.scatter, weightParams.absorbtion, currentCurvature);

            outFile << segWeight << std::endl;
        }
    }

    return 0;
}