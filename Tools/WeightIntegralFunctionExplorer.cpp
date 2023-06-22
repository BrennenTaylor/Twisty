#include <PathWeightUtils.h>

#include <fstream>

int main() {
    const uint32_t numStepsInt = 100000;
    const float eps = 0.001f;
    const float maxP = 10.0f / eps;

    const float pStep = maxP / (float)numStepsInt;
    const float pStepHalf = pStep * 0.5f;

    const float ds = 11.0f / 66;

    const float minCurvature = 0.0f;
    const float maxCurvature = (2.0f / ds) * 1.01f;;
    const float scatter = 0.9f;

    const float kds_min = minCurvature * ds;
    const float kds_max = maxCurvature * ds;

    const float kds_mid = 0.5* (kds_min + kds_max);

    const float bds = scatter * ds;

    const float mu = 0.1f;

    const double PI = 3.14159265358979323846;
    const double div = 1.0 / (2.0 * PI * PI);


    {
        std::ofstream file;
        file.open("IntegrandRT_min.csv");

        for (int idx = 0; idx < numStepsInt; idx++) {
            const float currentP = pStepHalf + (float)idx * pStep;
            double result = twisty::PathWeighting::IntegrandRT(currentP, kds_min, bds, eps, mu) * div;
            file << currentP << ',' << result << ',' << std::abs(result) << '\n';
        }
        file.close();
    }

    {
        std::ofstream file;
        file.open("IntegrandRT_max.csv");

        for (int idx = 0; idx < numStepsInt; idx++) {
            const float currentP = pStepHalf + (float)idx * pStep;
            double result = twisty::PathWeighting::IntegrandRT(currentP, kds_max, bds, eps, mu) * div;
            file << currentP << ',' << result << ',' << std::abs(result) << '\n';
        }
        file.close();
    }

    {
        std::ofstream file;
        file.open("IntegrandRT_mid.csv");

        for (int idx = 0; idx < numStepsInt; idx++) {
            const float currentP = pStepHalf + (float)idx * pStep;
            double result
                  = twisty::PathWeighting::IntegrandRT(currentP, kds_mid, bds, eps, mu) * div;
            file << currentP << ',' << result << ',' << std::abs(result) << '\n';
        }
        file.close();
    }


    return 0;
}