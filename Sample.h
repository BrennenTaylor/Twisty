#pragma once

#include <FMath/Vector3.h>

namespace twisty
{
    class Sample
    {
    public:
        static Farlor::Vector3 SampleUnitSphere(const float rand0, const float rand1);
    };
}