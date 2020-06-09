#pragma once

#include <functional>

namespace twisty
{
    class Derivative
    {
    public:
        Derivative()
        {
        }

        template<typename T>
        T Derive(std::function<T(float)> evalFunction, const float val, const float shiftVal)
        {
            T leftVal = evalFunction(val - shiftVal);
            T rightVal = evalFunction(val + shiftVal);
            return (rightVal - leftVal) * (1.0f / (2.0f * shiftVal));
        }
    };
}