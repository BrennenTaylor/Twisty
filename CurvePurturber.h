/**
 * @file CurvePurturber.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-03-18
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include "Curve.h"
#include "Range.h"

#include <optional>

namespace twisty
{
    /**
     * @brief Responsible for taking a curve and puturbing it, provides static methods for doing so
     * 
     */
    class CurvePuturber
    {
    public:
        /**
         * @brief Construct a new Curve Puturber object
         * 
         */
        CurvePuturber();

        /**
         * @brief Generate and return a curve purtivation candidate, doesnt necessarily need to be a valid candidate
         * 
         * @param curve Curve to purturb
         * @param kdsRange Range of allowed curvature * ds values, used for generation of new curvature values
         * @param tdsRange Range of allowed torsion * ds values, used for generation of new torsion values
         * @return std::unique_ptr<Curve> Generated purturbed curve
         */
        static std::unique_ptr<Curve> GetCurvePutrubation(const Curve &curve, const Range& kdsRange, const Range& tdsRange);
        /**
         * @brief Attempt to generate a valid curve purtivation candidate, user specifies the number of allowed attempts.
         * 
         * @param curve 
         * @param numAttempts 
         * @return std::optional<Curve> 
         */
        static std::optional<Curve> GetValidCurvePurtubation(const Curve& curve, uint32_t numAttempts);
    };
}