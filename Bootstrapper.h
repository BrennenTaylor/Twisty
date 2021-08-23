/**
 * @file Bootstrapper.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-03-14
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "Curve.h"

#include "BezierCurve.h"

#include <FMath/Vector3.h>

#include <memory>
#include <random>

namespace twisty
{
    /**
     * @brief Base bootstrapper class, responsible for creating a seed curve based off scene parameters.
     * The bootstrapper is a state based machine, once a curve has been generated, it remains cached until reset is called.
     * Until we call reset, it will always return the same cached seed bezier, control points, and curve
     */
    class Bootstrapper
    {
    public:
        struct BezierInfo
        {
            Farlor::Vector3 m_controlPt0;
            Farlor::Vector3 m_controlPt1;
            Farlor::Vector3 m_controlPt2;
            Farlor::Vector3 m_controlPt3;
            Farlor::Vector3 m_controlPt4;
        };

    public:
        /**
         * @brief Construct a new Bootstrapper object
         *
         * @param arclengthRange Range of allowed arclength values
         */
        Bootstrapper(float targetArclength, uint32_t randomSeed);
        /**
         * @brief Destroy the Bootstrapper object
         *
         */
        virtual ~Bootstrapper();

        /**
         * @brief Deletes any cached curve, allowing for a fresh curve to be generated.
         *
         */
        virtual void Reset();

        uint32_t GetBootstrapSeed() const;

        /**
         * @brief Get the Start Position object
         *
         * @return Farlor::Vector3 Start position
         */
        Farlor::Vector3 GetStartPosition() const;
        /**
         * @brief Get the Start Normal object
         *
         * @return Farlor::Vector3 Start normal
         */
        Farlor::Vector3 GetStartNormal() const;
        /**
         * @brief Get the Target Position object
         *
         * @return Farlor::Vector3 Target position
         */
        Farlor::Vector3 GetTargetPosition() const;
        /**
         * @brief Get the Target Normal object
         *
         * @return Farlor::Vector3 Target normal
         */
        Farlor::Vector3 GetTargetNormal() const;

        /**
         * @brief Get the Cached Curve object
         *
         * @return std::unique_ptr<Curve>
         */
        std::unique_ptr<Curve> GetCachedCurve();

        /**
         * @brief Get the Bezier Info object
         *
         * @return std::unique_ptr<BezierInfo>
         */
        std::unique_ptr<BezierInfo> GetBezierInfo() const;

        /**
         * @brief Create a Curve object
         *
         * @param numSegments Number of segments in the final discrete curve. Derived class can implement this method.
         * @return std::unique_ptr<Curve> Generated discrete curve
         */
        virtual std::unique_ptr<Curve> CreateCurve(uint32_t numSegments);

        virtual std::unique_ptr<Curve> CreateCurveGeometricSafe(uint32_t numSegments);
        /**
         * @brief Return positions spaced ds apart on the curve
         *
         * @return std::vector<Farlor::Vector3>
         */
        virtual std::vector<Farlor::Vector3> GetTValuePositions() const;
        /**
         * @brief Return frames spaced ds apart on the curve
         *
         * @return std::vector<Farlor::Matrix3x3>
         */
        virtual std::vector<Farlor::Vector3> GetTValueFrames() const;

    protected:
        /**
         * @brief Called at the beginning of a reset call. Allows derived classes to do anything they need before a reset takes place.
         *
         */
        virtual void BeginReset() = 0;
        /**
         * @brief Called at the end of a reset.
         *
         */
        virtual void EndReset() = 0;

    protected:
        std::unique_ptr<Curve> ToDiscreteFSCurve(uint32_t m_numSegments, BezierCurve5&);

    protected:
        Farlor::Vector3 m_startPos;
        Farlor::Vector3 m_startDir;
        Farlor::Vector3 m_endPos;
        Farlor::Vector3 m_endDir;
        float m_targetArclength;

        bool m_isCached;

        std::unique_ptr<Curve> m_upCachedCurve;
        std::vector<float> m_cachedTValues;
        std::vector<Farlor::Vector3> m_cachedSegmentPositions;
        std::vector<Farlor::Vector3> m_cachedSegmentFrames;
        std::unique_ptr<BezierInfo> m_upCachedBezierInfo;
        std::unique_ptr<BezierCurve5> m_upCachedBezier;
        std::mt19937_64 m_gen;
        uint32_t m_bootstrapSeed = 0;
    };
}