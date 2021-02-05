#pragma once

#include "BezierCurve.h"
#include "Bootstrapper.h"
#include "Geometry.h"
#include "Range.h"

#include <FMath/Vector3.h>

#include <cstdint>

namespace twisty
{
    // Implements a bootstrapping method which takes a two geometric objects and generates a seed curve between them.
    class GeometryBootstrapper : public Bootstrapper
    {

    public:
        explicit GeometryBootstrapper(const Geometry& emitterGeometry, const Geometry& recieverGeometry, Range arclengthRange, uint32_t randomSeed);

    protected:
        virtual void BeginReset() override;
        virtual void EndReset() override;
    };
}