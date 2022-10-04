/**
 * @file Integrate.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-03-14
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include <functional>

namespace twisty
{
    // TODO: This should be refactored into a free function.
    /**
     * @brief Class providing numerical integration functionality.
     * 
     */
    class Integrater
    {
    public:
        /**
         * @brief Construct a new Integrater object
         * 
         */
        Integrater()
        {
        }

        /**
         * @brief Integrate a given function from a minimum to maximum value with the specified number of steps and based off the distance function provided.
         * 
         * @tparam T Return type of evauation function
         * @param evalFunction 
         * @param minVal 
         * @param maxVal 
         * @param numSteps 
         * @param distanceFunc 
         * @return float 
         */
        template<typename T>
        float Integrate(std::function<T(float)> evalFunction, const float minVal, const float maxVal, const uint32_t numSteps, std::function<float(T)> distanceFunc)
        {
            // This is a euler approximation to the integral
            float stepSize = (maxVal - minVal) / numSteps;
            float value = 0.0f;
            for (uint32_t i = 0; i < numSteps; ++i)
            {
                float leftParam = i * stepSize;
                leftParam = std::max(leftParam, minVal);
                float rightParam = leftParam + stepSize;
                rightParam = std::min(rightParam, maxVal);

                T left = evalFunction(leftParam);
                T right = evalFunction(rightParam);
                T diff = right - left;
                value += distanceFunc(diff);
            }
            return value;
        }
    };
}