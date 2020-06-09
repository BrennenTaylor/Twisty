#include "Sample.h"

namespace twisty
{
    Farlor::Vector3 Sample::UniformSphere(const float rand0, const float rand1)
    {
        float theta = 2.0f * 3.141592f * rand0;
        float phi = 3.141592f * rand1;
        float x = sin(phi) * cos(phi);
        float y = sin(phi) * sin(theta);
        float z = cos(phi);
        Farlor::Vector3 normal(x, y, z);
        return normal.Normalized();
    }
}