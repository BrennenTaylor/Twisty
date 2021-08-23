#include "CurveUtils.h"
#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "ExperimentRunner.h"

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <filesystem>

std::filesystem::path GetExperimentDirectory(const std::string experimentDirectoryName)
{
    // Get currect directory
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "CWD: " << cwd << std::endl;

    std::string experimentDirectoryAppend(experimentDirectoryName);
    std::filesystem::path experimentDirectoryPath = cwd;
    experimentDirectoryPath.append(experimentDirectoryAppend);
    return experimentDirectoryPath;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cout << "Call as: " << argv[0] << " ExperimentDirectory CurveFilename" << std::endl;
        return 1;
    }

    const std::filesystem::path currentPath = std::filesystem::current_path();
    const std::string experimentDirectory = argv[1];
    const std::filesystem::path experimentDirectoryPath = currentPath / experimentDirectory;
    const std::string curveFilename = argv[2];
    const std::filesystem::path curvePath = experimentDirectoryPath / curveFilename;

    std::ifstream curveFilestream(curvePath.c_str(), std::ios::binary);
    std::unique_ptr<twisty::Curve> upCurve = twisty::Curve::LoadCurveFromStream(curveFilestream);
    if (!upCurve)
    {
        std::cout << "Failed to load curve" << std::endl;
        return 1;
    }

    //std::cout << "Num Segments: " << upCurve->m_numSegments << std::endl;
    //for (auto curvature : upCurve->m_curvatures)
    //{
    //    std::cout << "Curvature: " << curvature << std::endl;
    //}

    twisty::WeightingParameters weightingParams;
    weightingParams.mu = 0.1;
    weightingParams.eps = 0.1;
    weightingParams.numStepsInt = 2000;
    weightingParams.minBound = 0.0;
    weightingParams.maxBound = 10.0 / weightingParams.eps;
    weightingParams.numCurvatureSteps = 10000;
    // Lets give some absorbtion as well
    // Absorbtion 1/20 off the time
    weightingParams.absorbtion = 0.05;
    // 1/5 scatter means one event every 5 units, thus 2 scattering events in the shortest
    // or 5 in the longest 100 unit path
    weightingParams.scatter = 0.2;

    // Say that we will start outputing the path batch output
    const double ds = upCurve->m_arclength / upCurve->m_numSegments;

    // Constants
    double minCurvature = 0.0;
    double maxCurvature = 0.0;
    twisty::PathWeighting::CalcMinMaxCurvature(minCurvature, maxCurvature, ds);

    // Note: Has absorbtion const built in
    twisty::PathWeighting::WeightLookupTableIntegral lookupEvaluator(weightingParams, ds);
    double curveWeightLog10 = twisty::PathWeighting::WeightCurveViaCurvatureLog10(upCurve->m_curvatures.data(), upCurve->m_curvatures.size(), lookupEvaluator);
    std::cout << "Weight (Log10): " << curveWeightLog10 << std::endl;

    boost::multiprecision::cpp_dec_float_100 bigFloatCurveWeightLog10 = curveWeightLog10;
    boost::multiprecision::cpp_dec_float_100 bigFloatCurveWeight = boost::multiprecision::pow(10.0, bigFloatCurveWeightLog10);
    std::cout << "Weight: " << bigFloatCurveWeight << std::endl;

    // Need to add in path scattering const
    boost::multiprecision::cpp_dec_float_100 singleSegmentNormalizer = 2.0 * TwistyPi * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(0.625));
    boost::multiprecision::cpp_dec_float_100 segmentNormalizer = 1.0;
    for (int64_t segIdx = 0; segIdx < (upCurve->m_numSegments - 1); ++segIdx)
    {
        segmentNormalizer = segmentNormalizer * singleSegmentNormalizer;
    }

    boost::multiprecision::cpp_dec_float_100 pathNormalizer = 1.0;
    pathNormalizer = pathNormalizer * boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(static_cast<float>(upCurve->m_numSegments) / upCurve->m_arclength), 3.0);
    pathNormalizer = pathNormalizer * segmentNormalizer;
    pathNormalizer = pathNormalizer * boost::multiprecision::exp(boost::multiprecision::cpp_dec_float_100(-0.325));

    boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10 = boost::multiprecision::log10(pathNormalizer);

    std::cout << "Normalized Weight (Log10): " << bigFloatCurveWeightLog10 + pathNormalizerLog10 << std::endl;
    std::cout << "Normalized Weight (Multiplied): " << bigFloatCurveWeight * pathNormalizer << std::endl;
    std::cout << "Normalized Weight (Decompressed): " << boost::multiprecision::pow(10.0, (bigFloatCurveWeightLog10 + pathNormalizerLog10)) << std::endl;
}