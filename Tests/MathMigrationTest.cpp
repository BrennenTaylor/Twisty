#include <cmath>
#include <iostream>

#include <glm/glm.hpp>

static bool nearlyEqual(float a, float b)
{
    return std::abs(a - b) < 1e-6f;
}

int main()
{
    const glm::vec3 a(1.0f, 2.0f, 3.0f);
    const glm::vec3 b(4.0f, -5.0f, 6.0f);

    // Dot — declared 1*4 + 2*(-5) + 3*6
    if (!nearlyEqual(glm::dot(a, b), 12.0f)) {
        std::cout << "FAILURE: glm::dot" << std::endl;
        return 1;
    }

    // Cross — (2*6 - 3*(-5), 3*4 - 1*6, 1*(-5) - 2*4)
    const glm::vec3 c = glm::cross(a, b);
    if (!nearlyEqual(c.x, 27.0f) || !nearlyEqual(c.y, 6.0f) || !nearlyEqual(c.z, -13.0f)) {
        std::cout << "FAILURE: glm::cross" << std::endl;
        return 1;
    }

    // Length / length2
    if (!nearlyEqual(glm::length(a), std::sqrt(14.0f)) || !nearlyEqual(glm::dot(a, a), 14.0f)) {
        std::cout << "FAILURE: glm::length/length2" << std::endl;
        return 1;
    }

    // Normalize — unit vector on the exact same line
    const glm::vec3 n = glm::normalize(a);
    if (!nearlyEqual(glm::length(n), 1.0f)) {
        std::cout << "FAILURE: glm::normalize" << std::endl;
        return 1;
    }

    // Component-wise + and scalar *
    const glm::vec3 sum = a + b;
    const glm::vec3 scaled = a * 2.0f;
    if (!nearlyEqual(sum.x, 5.0f) || !nearlyEqual(sum.y, -3.0f) || !nearlyEqual(sum.z, 9.0f)
        || !nearlyEqual(scaled.x, 2.0f) || !nearlyEqual(scaled.z, 6.0f)) {
        std::cout << "FAILURE: component-wise + / scalar *" << std::endl;
        return 1;
    }

    // Vector equality via glm::all(glm::equal(...))
    if (!glm::all(glm::equal(a, a)) || glm::all(glm::equal(a, b))) {
        std::cout << "FAILURE: vector equality" << std::endl;
        return 1;
    }

    std::cout << "SUCCESS: MathMigrationTest passes!" << std::endl;
    return 0;
}