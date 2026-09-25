#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <PathGeneration.h>

namespace {
bool NearlyEqual(const glm::vec3 &a, const glm::vec3 &b)
{
    return std::abs(a.x - b.x) <= 1e-5f
          && std::abs(a.y - b.y) <= 1e-5f
          && std::abs(a.z - b.z) <= 1e-5f;
}
}

int main()
{
    const double ds = 1.0;

    // Endpoints exactly 2*ds apart -> ResolveTwoSegments must place the single
    // middle point at the midpoint and return immediately.
    std::vector<glm::vec3> pointList { glm::vec3(0.0f, 0.0f, 0.0f),
          glm::vec3(99.0f, 99.0f, 99.0f), glm::vec3(2.0f, 0.0f, 0.0f) };

    std::mt19937_64 rng(12345);
    twisty::PathGeneration::ResolveTwoSegments(pointList, 0, 2, ds, rng);

    const glm::vec3 expected(1.0f, 0.0f, 0.0f);
    if (!NearlyEqual(pointList[1], expected)) {
        printf("FAILURE: ResolveTwoSegments placed %f %f %f, expected %f %f %f\n",
              pointList[1].x, pointList[1].y, pointList[1].z, expected.x, expected.y,
              expected.z);
        return 1;
    }

    printf("SUCCESS: ResolveTwoSegments placed the middle point at the midpoint\n");
    return 0;
}