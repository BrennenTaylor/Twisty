#include "Sample.h"

#define _USE_MATH_DEFINES
#include <math.h>

namespace twisty
{
    /*
        // Incorrect
        double theta = 2 * M_PI * uniform01(generator);
        double phi = M_PI * uniform01(generator);
        double x = sin(phi) * cos(theta);
        double y = sin(phi) * sin(theta);
        double z = cos(phi);
    


        // Correct
        double theta = 2 * M_PI * uniform01(generator);
        double phi = acos(1 - 2 * uniform01(generator));
        double x = sin(phi) * cos(theta);
        double y = sin(phi) * sin(theta);
        double z = cos(phi);
    */




    Farlor::Vector3 Sample::SampleUnitSphere(const float rand0, const float rand1)
    {
        float theta = 2.0f * M_PI * rand0;
        float phi = std::acos(1.0f - 2.0f * rand1);
        float x = std::sin(phi) * std::cos(theta);
        float y = std::sin(phi) * std::sin(theta);
        float z = std::cos(phi);
        Farlor::Vector3 normal(x, y, z);
        return normal.Normalized();
    }
}