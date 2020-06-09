/**
 * @file BezierCurve.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-03-14
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "FMath/Vector3.h"

#include <cstdint>
#include <vector>

namespace twisty
{
    /**
     * @brief Three control point bezier curve. Exists for the parameter space t in [0, 1]
     *
     */
    class BezierCurve3
    {
    public:
        /**
         * @brief Construct a new Bezier Curve 3 object, initializes the control points to all 0.0f
         *
         */
        explicit BezierCurve3();

        /**
         * @brief Get the Position object
         *
         * @param t value between 0 and 1. t = 0 points to the first control point, t = 1 points to last control point
         * @return Farlor::Vector3 The position t distance along the bezier curve.
         */
        Farlor::Vector3 GetPosition(float t);

        /**
         * @brief Output the control point information to the console
         *
         */
        void PrintControlPts();

    public:
        /**
         * @brief Stores the n control points, public to allow for easy access
         *
         */
        std::vector<Farlor::Vector3> m_controlPts;
        /**
         * @brief Reflects the number of control points, 3 in this case
         *
         */
        static const uint32_t s_NumControlPts;
    };

    /**
     * @brief Four control point bezier curve. Exists for the parameter space t in [0, 1]
     *
     */
    class BezierCurve4
    {
    public:
        /**
         * @brief Construct a new Bezier Curve 4 object
         *
         */
        explicit BezierCurve4();

        /**
         * @brief Get the Position object
         *
         * @param t value between 0 and 1. t = 0 points to the first control point, t = 1 points to last control point
         * @return Farlor::Vector3 The position t distance along the bezier curve.
         */
        Farlor::Vector3 GetPosition(float t);

        /**
         * @brief Get the first derivative value, calculated through a derivitive curve.
         *
         * @param t value between 0 and 1. t = 0 points to the first control point, t = 1 points to last control point
         * @return Farlor::Vector3 The first derivative t distance along the bezier curve.
         */
        Farlor::Vector3 FirstDerivative(float t);

        /**
         * @brief Output the control point information to the console
         *
         */
        void PrintControlPts();

        /**
         * @brief Get the derivative curve, a bezier with n-1 control points
         *
         * @return BezierCurve3 Derivative curve
         */
        BezierCurve3 GetDerivativeCurve();

    public:
        /**
         * @brief Stores the n control points, public to allow for easy access
         *
         */
        std::vector<Farlor::Vector3> m_controlPts;
        /**
         * @brief Reflects the number of control points, 4 in this case
         *
         */
        static const uint32_t s_NumControlPts;
    };

    /**
     * @brief Five control point bezier curve. Exists for the parameter space t in [0, 1]
     *
     */
    class BezierCurve5
    {
    public:
        /**
         * @brief Construct a new Bezier Curve 5 object
         *
         */
        explicit BezierCurve5();

        /**
         * @brief Get the Position object
         *
         * @param t value between 0 and 1. t = 0 points to the first control point, t = 1 points to last control point
         * @return Farlor::Vector3 The position t distance along the bezier curve.
         */
        Farlor::Vector3 GetPosition(float t);
        /**
         * @brief Get the first derivative value, calculated through a derivitive curve.
         *
         * @param t value between 0 and 1. t = 0 points to the first control point, t = 1 points to last control point
         * @return Farlor::Vector3 The first derivative t distance along the bezier curve.
         */
        Farlor::Vector3 FirstDerivative(float t);
        /**
         * @brief Get the second derivative value, calculated through a derivitive curve.
         *
         * @param t value between 0 and 1. t = 0 points to the first control point, t = 1 points to last control point
         * @return Farlor::Vector3 The second derivative t distance along the bezier curve.
         */
        Farlor::Vector3 SecondDerivative(float t);
        /**
         * @brief Get the third derivative value, calculated through a derivitive curve.
         *
         * @param t value between 0 and 1. t = 0 points to the first control point, t = 1 points to last control point
         * @return Farlor::Vector3 The third derivative t distance along the bezier curve.
         */
        Farlor::Vector3 ThirdDerivative(float t);

        /**
         * @brief Calculates normalized tangent at t value along curve
         *
         * @param t Parameter between 0 and 1
         * @return Farlor::Vector3 Normalized tangent at t
         */
        Farlor::Vector3 Tangent(float t);
        /**
         * @brief Calculates normalized normal at t value along curve
         *
         * @param t Parameter between 0 and 1
         * @return Farlor::Vector3 Normalized normal t distance along curve
         */
        Farlor::Vector3 Normal(float t);
        /**
         * @brief Calcualtes binormal at t value along curve
         *
         * @param t Parameter between 0 and 1
         * @return Farlor::Vector3 Normalized binormal t distance along curve
         */
        Farlor::Vector3 Binormal(float t);

        /**
         * @brief Integrate along the bezier from a min to max t value, calculating arclength.
         * Verified with the formula at: http://tutorial.math.lamar.edu/Classes/CalcIII/VectorArcLength.aspx.
         *
         * @param minVal minimum t value
         * @param maxVal maximum t value
         * @return float arclength of the curve from min to max t values
         */
        float CalculateArclength(float minVal, float maxVal);

        /**
         * @brief This caches n arclength values along the curve
         *
         * @param numCachedValues
         * @return float
         */
        void CacheArclength(uint32_t numCachedValues);
        /**
         * @brief Use the cached bezier values from CacheArclength
         * If uncached, return 0.0f
         *
         * @param minVal
         * @param maxVal
         * @return float
         */
        float CalculateArclengthAlreadyCached(float minVal, float maxVal);

        /**
         * @brief Output the control point information to the console
         *
         */
        void PrintControlPts();
        /**
         * @brief Get the derivative curve, a bezier with n-1 control points
         *
         * @return Derivative curve
         */
        BezierCurve4 GetDerivativeCurve();

    public:
        /**
         * @brief Stores the n control points, public to allow for easy access
         *
         */
        std::vector<Farlor::Vector3> m_controlPts;
        /**
         * @brief CachedValues of t values set ds apart
         *
         */
        std::vector<std::pair<float, float>> m_cachedTLocations;
        /**
         * @brief Reflects the number of control points, 4 in this case
         *
         */
        static const uint32_t s_NumControlPts;
    };
}