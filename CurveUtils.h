#pragma once

#include "Curve.h"

#include <cstdint>

namespace twisty
{
    class CurveUtils
    {
    public:
        struct CurveState
        {
            float k1;
            float k2;
            float k3;
            float t1;
            float t2;
            float t3;
        };

        static float CalculateCurveError(const Curve& curve);
        static float CalculateCurveMeasure(const Curve& curve);
    };
}