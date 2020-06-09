#pragma once

#include "BezierCurve.h"
#include "Bootstrapper.h"
#include "Range.h"

#include <FMath\Vector3.h>

#include <cstdint>

namespace twisty
{
    // Implements a bootstrapping method which takes a two geometric objects and generates a seed curve between them.
    class GeometryBootstrapper : public Bootstrapper
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

    public:
        explicit GeometryBootstrapper(const Geometry& emitterGeometry, const Geometry& recieverGeometry, Range arclengthRange, uint32_t randomSeed);

    protected:
        virtual void BeginReset() override;
        virtual void EndReset() override;
    };
}