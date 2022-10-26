#include "FMath/Vector3.h"
#include <FMath/Vector2.h>

#include <limits>
#include <numbers>
#include <cmath>
#include <vector>
#include <random>
#include <assert.h>

constexpr uint32_t numBands = 4;
constexpr uint32_t NumCoeff = numBands + ((numBands - 1) * (numBands - 1)) + (numBands - 1);
constexpr uint32_t numSamplesSqrt = 2000;

struct SPHSample {
    Farlor::Vector2 sph = Farlor::Vector2(0.0f, 0.0f);
    std::array<float, NumCoeff> coeff = { 0.0f, 0.0f, 0.0f, 0.0f };
};

float LightFunction(const float theta, const float phi)
{
    return std::max(0.0f, 5.0f * std::cos(theta) - 4.0f)
          + std::max(0.0f,
                -4.0f * std::sin(theta - std::numbers::pi_v<float>) * std::cos(phi - 2.5f) - 3.0f);
}

// Will project the above light function
std::vector<float> ProjectPolarFunction(const std::vector<SPHSample> &samples)
{
    // Generate Samples
    std::vector<float> result(NumCoeff);

    const float weight = 4.0f * std::numbers::pi_v<float>;
    for (const auto &sample : samples) {
        const float theta = sample.sph.x;
        const float phi = sample.sph.y;
        for (int n = 0; n < NumCoeff; n++) {
            result[n] += LightFunction(theta, phi) * sample.coeff[n];
        }
    }

    const float factor = weight / samples.size();
    for (int n = 0; n < NumCoeff; n++) {
        result[n] = result[n] * factor;
    }

    return result;
}

// both l and m are positive integers
float P(int l, int m, float x)
{
    assert(l >= 0);
    assert(m >= 0);
    float ppm = 1.0f;
    if (m > 0) {
        float somx2 = std::sqrt((1.0f - x) * (1.0f + x));
        float fact = 1.0f;
        for (int i = 1; i <= m; i++) {
            ppm *= (-fact) * somx2;
            fact += 2.0f;
        }
    }

    if (l == m)
        return ppm;

    float ppmp1 = x * (2.0f * static_cast<float>(m) + 1.0f) * ppm;
    if (l == m + 1)
        return ppmp1;

    float pll = 0.0f;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = ((2.0f * static_cast<float>(ll) - 1.0f) * x * ppmp1
                    - (static_cast<float>(ll) + static_cast<float>(m) - 1.0f) * ppm)
              / (static_cast<float>(ll) - static_cast<float>(m));
        ppm = ppmp1;
        ppmp1 = pll;
    }
    return pll;
}

// Renomalization constant for SH
float K(int l, int m)
{
    float temp = ((2.0f * static_cast<float>(l) + 1.0f)
                       * std::tgamma(static_cast<float>(l - abs(m)) + 1.0f))
          / (4.0f * std::numbers::pi_v<float> * std::tgamma(static_cast<float>(l + abs(m)) + 1.0f));
    return std::sqrtf(temp);
}

// l is 0 - > inf
// m is [-l, l]
float SH(int l, int m, float theta, float phi)
{
    const float sqrt2 = std::sqrtf(2.0f);
    if (m == 0) {
        return K(l, 0) * P(l, 0, cos(theta));
    } else if (m > 0) {
        return sqrt2 * K(l, m) * cos(static_cast<float>(m) * phi) * P(l, m, std::cos(theta));
    } else {
        return sqrt2 * K(l, m) * std::sin(static_cast<float>(-m) * phi) * P(l, -m, std::cos(theta));
    }
}

int main()
{
    std::mt19937_64 generator(time(0));
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    std::vector<SPHSample> samples(numSamplesSqrt * numSamplesSqrt);
    for (uint32_t sampleIdx = 0; sampleIdx < numSamplesSqrt * numSamplesSqrt; sampleIdx++) {
        const uint32_t a = sampleIdx / numSamplesSqrt;
        const uint32_t b = sampleIdx % numSamplesSqrt;

        const float oneOverSqrtN = 1.0f / numSamplesSqrt;
        const float e0 = (a + uniform01(generator)) * oneOverSqrtN;
        const float e1 = (b + uniform01(generator)) * oneOverSqrtN;


        auto &sample = samples[sampleIdx];
        sample.sph.x = 2.0f * std::acos(std::sqrt(1.0f - e0));
        sample.sph.y = 2.0f * std::numbers::pi_v<float> * e1;

        // Precompute coefficients
        for (int l = 0; l < numBands; l++) {
            for (int m = -l; m <= l; m++) {
                int sphIdx = l * (l + 1) + m;
                sample.coeff[sphIdx] = SH(l, m, sample.sph.x, sample.sph.y);
            }
        }
    }

    std::vector<float> coefficients = ProjectPolarFunction(samples);
    std::cout << coefficients.size() << std::endl;

    int count = 0;
    int numNextLine = 1;

    for (auto &coefficient : coefficients) {
        std::cout << coefficient << ", ";
        count++;
        if (count == numNextLine) {
            std::cout << "\n";
            count = 0;
            numNextLine += 2;
        }
    }


    // Lets reconstruct the signal
    std::vector<SPHSample> testSamples(20);
    for (uint32_t sampleIdx = 0; sampleIdx < numSamplesSqrt * numSamplesSqrt; sampleIdx++) {
        const uint32_t a = sampleIdx / numSamplesSqrt;
        const uint32_t b = sampleIdx % numSamplesSqrt;

        const float oneOverSqrtN = 1.0f / numSamplesSqrt;
        const float e0 = (a + uniform01(generator)) * oneOverSqrtN;
        const float e1 = (b + uniform01(generator)) * oneOverSqrtN;


        auto &sample = samples[sampleIdx];
        sample.sph.x = 2.0f * std::acos(std::sqrt(1.0f - e0));
        sample.sph.y = 2.0f * std::numbers::pi_v<float> * e1;

        float reconstructedValue = 0.0f;
        // Precompute coefficients
        for (int l = 0; l < numBands; l++) {
            for (int m = -l; m <= l; m++) {
                int sphIdx = l * (l + 1) + m;
                reconstructedValue += coefficients[sphIdx] * SH(l, m, sample.sph.x, sample.sph.y);
            }
        }
        const float exactValue = LightFunction(sample.sph.x, sample.sph.y);

        std::cout << "Reconstructed: " << reconstructedValue << "\n";
        std::cout << "Exact value: " << exactValue << "\n";
        std::cout << "Abs Error: " << abs(reconstructedValue - exactValue) << std::endl;
    }

    std::cout << std::endl;

    return 0;
}