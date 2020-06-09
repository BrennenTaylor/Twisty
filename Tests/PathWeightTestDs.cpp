#include <PathWeightUtils.h>

#include <assert.h>

using namespace twisty;

bool ThresholdFloatEquality(float first, float second, float epsilon)
{
    return abs(first - second) <= epsilon;
}

int main(int argc, char* argv[])
{
    if (argc < 7)
    {
        std::cout << "Not enough arguments" << std::endl;
        std::cout << "Call as: " << argv[0] << " numCurvatureSteps minDs maxDs curvature absorbtion scattering" << std::endl;
        return 1;
    }

    const uint32_t numSteps = atoi(argv[1]);
    const float minDs = atof(argv[2]);
    const float maxDs = atof(argv[3]);

    const float curvature = atof(argv[4]);
    const float absorbtion = atof(argv[5]);
    const float scattering = atof(argv[6]);

    const double mu = 0.1;
    const uint32_t numStepsInt = 2000;
    const double minBound = 0.0;
    const double maxBound = 100.0;
    const double eps = 0.1f;

    auto AbsorbtionFromPos = [absorbtion](Farlor::Vector3 x) -> float
    {
        return absorbtion;
    };

    auto ScatterFromPos = [scattering](Farlor::Vector3 x) -> float
    {
        return scattering;
    };

    PathSpaceUtils::RegularizedIntegral regIntEvaluator(mu, numStepsInt, minBound, maxBound, eps);

    {
        float dsStepSize = (maxDs - minDs) / numSteps;
        std::cout << "Absorbtion, " << absorbtion << std::endl;
        std::cout << "Scattering, " << scattering << std::endl;
        std::cout << "Curvature, " << curvature << std::endl;
        std::cout << "Segment Length, weight: " << std::endl;
        for (uint32_t i = 0; i <= numSteps; ++i)
        {
            Segment segToWeigh;
            segToWeigh.m_curvature = curvature;
            // Just pick a torsion
            segToWeigh.m_torsion = 0.0f;
            // Just pick a ds as well

            segToWeigh.m_length = minDs + dsStepSize * i;

            float minClip = 0.0f;
            float maxClip = FLT_MAX;

            auto segmentWeight = PathSpaceUtils::WeightSegment(segToWeigh, AbsorbtionFromPos, ScatterFromPos, regIntEvaluator, minClip, maxClip);
            std::cout << segToWeigh.m_length << ", " << segmentWeight << std::endl;
        }
    }

    return 0;
}