#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <PathGeneration.h>

namespace {
bool NearlyEqual(const Farlor::Vector3 &a, const Farlor::Vector3 &b)
{
    return std::abs(a.m_data[0] - b.m_data[0]) <= 1e-5f
          && std::abs(a.m_data[1] - b.m_data[1]) <= 1e-5f
          && std::abs(a.m_data[2] - b.m_data[2]) <= 1e-5f;
}
}

int main()
{
    const double ds = 1.0;

    // Endpoints exactly 2*ds apart -> ResolveTwoSegments must place the single
    // middle point at the midpoint and return immediately.
    std::vector<Farlor::Vector3> pointList { Farlor::Vector3(0.0f, 0.0f, 0.0f),
          Farlor::Vector3(99.0f, 99.0f, 99.0f), Farlor::Vector3(2.0f, 0.0f, 0.0f) };

    std::mt19937_64 rng(12345);
    twisty::PathGeneration::ResolveTwoSegments(pointList, 0, 2, ds, rng);

    const Farlor::Vector3 expected(1.0f, 0.0f, 0.0f);
    if (!NearlyEqual(pointList[1], expected)) {
        printf("FAILURE: ResolveTwoSegments placed %f %f %f, expected %f %f %f\n",
              pointList[1].m_data[0], pointList[1].m_data[1], pointList[1].m_data[2],
              expected.m_data[0], expected.m_data[1], expected.m_data[2]);
        return 1;
    }

    printf("SUCCESS: ResolveTwoSegments placed the middle point at the midpoint\n");
    return 0;
}