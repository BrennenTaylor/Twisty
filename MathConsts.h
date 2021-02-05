#pragma once

namespace twisty
{
    /**
     * @brief Constant defining pi value used in all twisty code
     * 
     */
    #define TwistyPi 3.141592653589793f

    inline float DegreeFromRadian(float radian)
    {
        return (radian * 180.0f) / TwistyPi;
    }

    inline float RadianFromDegree(float degree)
    {
        return (degree * TwistyPi) / 180.0f;
    }
}