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
        class Geometry
        {
        public:
            struct SampleRay
            {
                Farlor::Vector3 m_pos;
                Farlor::Vector3 m_dir;
            };

        public:
            explicit Geometry()
            {
            }

            virtual SampleRay GetSampleRay() const = 0;
        };

        class RayGeometry : public Geometry
        {
        public:
            explicit RayGeometry(Farlor::Vector3 start, Farlor::Vector3 dir);
            virtual Geometry::SampleRay GetSampleRay() const override;

        private:
            Farlor::Vector3 m_pos;
            Farlor::Vector3 m_dir;
        };

        class SphereGeometry : public Geometry
        {
        public:
            explicit SphereGeometry(Farlor::Vector3 pos, float radius, float fov);
            virtual Geometry::SampleRay GetSampleRay() const override;

        private:
            Farlor::Vector3 m_pos;
            float m_radius;
            float m_fov;
        };

    public:
        // Represents a bezier curve
        struct BezierInfo
        {
            Farlor::Vector3 m_controlPt0 = Farlor::Vector3(0.0, 0.0, 0.0);
            Farlor::Vector3 m_controlPt1 = Farlor::Vector3(0.0, 0.0, 0.0);
            Farlor::Vector3 m_controlPt2 = Farlor::Vector3(0.0, 0.0, 0.0);
            Farlor::Vector3 m_controlPt3 = Farlor::Vector3(0.0, 0.0, 0.0);
            Farlor::Vector3 m_controlPt4 = Farlor::Vector3(0.0, 0.0, 0.0);
        };

    public:
        Bootstrapper();
        // Samples from geometry
        Bootstrapper(const Geometry& emitterGeometry, const Geometry& recieverGeometry);
        ~Bootstrapper();

        Farlor::Vector3 GetStartPosition() const;
        Farlor::Vector3 GetStartNormal() const;
        Farlor::Vector3 GetTargetPosition() const;
        Farlor::Vector3 GetTargetNormal() const;

        std::unique_ptr<Curve> CreateCurve(uint32_t numSegments, float targetArclength, uint32_t generationSeed);
        std::unique_ptr<Curve> CreateCurveGeometricSafe(uint32_t numSegments, float targetArclength);

    private:
        std::unique_ptr<Curve> ToDiscreteFSCurve(uint32_t m_numSegments, BezierCurve5& curve);

    private:
        Farlor::Vector3 m_startPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        Farlor::Vector3 m_startDir = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        Farlor::Vector3 m_endPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
        Farlor::Vector3 m_endDir = Farlor::Vector3(0.0f, 0.0f, 0.0f);
    };
}