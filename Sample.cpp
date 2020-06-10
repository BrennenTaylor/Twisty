#include "Sample.h"

#include <cmath>

namespace twisty
{
    Farlor::Vector3 Sample::UniformSphere(const float rand0, const float rand1)
    {
        float theta = 2.0f * 3.141592f * rand0;
        float phi = 3.141592f * rand1;
        float x = std::sin(phi) * std::cos(phi);
        float y = std::sin(phi) * std::sin(theta);
        float z = std::cos(phi);
        Farlor::Vector3 normal(x, y, z);
        return normal.Normalized();
    }
}