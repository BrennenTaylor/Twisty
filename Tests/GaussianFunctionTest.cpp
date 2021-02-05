#include <PathWeightUtils.h>

#include <assert.h>

using namespace twisty;

int main()
{
    {
        const uint32_t numSteps = 500;

        // TODO/NOTE: Curvature shouldnt be negative... right?
        const float minP = 0.0f;
        const float maxP = 10.0f;

        const float mu = 10.0f;

        float dp = (maxP- minP) / numSteps;
        std::cout << "P, gaussian weight: " << std::endl;
        for (uint32_t i = 0; i < numSteps; ++i)
        {
            float p = minP + i * dp;
            auto gaussianWeight = PathWeighting::GaussianPhase(p, mu);
            std::cout << p << ", " << gaussianWeight << std::endl;
        }
    }

    return 0;
}