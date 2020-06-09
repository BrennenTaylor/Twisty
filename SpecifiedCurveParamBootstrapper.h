#pragma once

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
    class SpecifiedCurveParamBootstrapper : public Bootstrapper
    {
    public:

        /**
         * @brief
         *
         */
        SpecifiedCurveParamBootstrapper(float initialCurvature, float initialTorsion, const Range& arclengthRange, uint32_t randomSeed);

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
        float m_initialCurvature;
        float m_initialTorsion;
    };
}