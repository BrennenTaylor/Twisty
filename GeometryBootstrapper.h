#pragma once

#include "BezierCurve.h"
#include "Bootstrapper.h"
#include "Geometry.h"

#include <FMath/Vector3.h>

#include <cstdint>

namespace twisty
{
    // Implements a bootstrapping method which takes a two geometric objects and generates a seed curve between them.
    class GeometryBootstrapper : public Bootstrapper
    {

    public:
        explicit GeometryBootstrapper(const Geometry& emitterGeometry, const Geometry& recieverGeometry);

    protected:
        virtual void BeginReset() override;
        virtual void EndReset() override;
    };
}