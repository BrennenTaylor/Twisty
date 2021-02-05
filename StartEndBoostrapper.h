/**
 * @file StartEndBoostrapper.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-03-19
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "BezierCurve.h"
#include "Bootstrapper.h"
#include "Range.h"

#include <FMath\Vector3.h>

#include <cstdint>

namespace twisty
{
    /**
     * @brief Bootstrapper which uses specified start and end directions and positions.
     *
     */
    class StartEndBoostrapper : public Bootstrapper
    {
    public:

        /**
         * @brief
         *
         */
        StartEndBoostrapper(const Vector3& startPos, const Vector3& startDir
            const Vector3& endPos, const Vector3& endDir, Range arclengthRange);


        /**
         * @brief Get the Target Position object
         *
         * @return Farlor::Vector3 The target position
         */
        virtual Farlor::Vector3 GetTargetPosition() override;
        /**
         * @brief Get the Target Normal object
         *
         * @return Farlor::Vector3 The target normal
         */
        virtual Farlor::Vector3 GetTargetNormal() override;

        /**
         * @brief Create a Curve object
         *
         * @param numSegments Number of segments in the discritized curve
         * @return std::unique_ptr<Curve> Discritized curve object
         */
        virtual std::unique_ptr<Curve> CreateCurve(uint32_t numSegments) override;

        std::unique_ptr<BezierInfo> GetBezierInfo();
        std::vector<Farlor::Vector3> GetTValuePositions();
        std::vector<Farlor::Matrix3x3> GetTValueFrames();

    protected:
        virtual void BeginReset() override;
        virtual void EndReset() override;

    private:
        std::unique_ptr<Curve> ToDiscreteFSCurve(uint32_t m_numSegments, BezierCurve5&);

    private:
        std::unique_ptr<BezierInfo> m_upCachedBezierInfo;

        std::unique_ptr<BezierCurve5> m_upCachedBezier;

        Farlor::Vector3 m_x0;
        Farlor::Vector3 m_n0;
        Farlor::Vector3 m_x1;
        Farlor::Vector3 m_n1;
        Range m_arclengthRange;
    };
}