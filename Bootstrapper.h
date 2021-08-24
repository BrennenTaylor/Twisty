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
        Bootstrapper();
        virtual ~Bootstrapper();

        virtual void Reset();

        Farlor::Vector3 GetStartPosition() const;
        Farlor::Vector3 GetStartNormal() const;
        Farlor::Vector3 GetTargetPosition() const;
        Farlor::Vector3 GetTargetNormal() const;

        std::unique_ptr<Curve> GetCachedCurve();

        std::unique_ptr<BezierInfo> GetBezierInfo() const;

        virtual std::unique_ptr<Curve> CreateCurve(uint32_t numSegments, float targetArclength, uint32_t generationSeed);

        virtual std::unique_ptr<Curve> CreateCurveGeometricSafe(uint32_t numSegments, float targetArclength);
        virtual std::vector<Farlor::Vector3> GetTValuePositions() const;
        virtual std::vector<Farlor::Vector3> GetTValueFrames() const;

    protected:
        virtual void BeginReset() = 0;
        virtual void EndReset() = 0;

    protected:
        std::unique_ptr<Curve> ToDiscreteFSCurve(uint32_t m_numSegments, BezierCurve5&);

    protected:
        Farlor::Vector3 m_startPos;
        Farlor::Vector3 m_startDir;
        Farlor::Vector3 m_endPos;
        Farlor::Vector3 m_endDir;

        bool m_isCached;

        std::unique_ptr<Curve> m_upCachedCurve;
        std::vector<float> m_cachedTValues;
        std::vector<Farlor::Vector3> m_cachedSegmentPositions;
        std::vector<Farlor::Vector3> m_cachedSegmentFrames;
        std::unique_ptr<BezierInfo> m_upCachedBezierInfo;
        std::unique_ptr<BezierCurve5> m_upCachedBezier;
    };
}