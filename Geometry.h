#pragma once

#include "GeometryBootstrapper.h"

namespace twisty
{
    class RayGeometry : public GeometryBootstrapper::Geometry
    {
    public:
        explicit RayGeometry(Farlor::Vector3 start, Farlor::Vector3 dir);
        virtual GeometryBootstrapper::Geometry::SampleRay GetSampleRay() const override;

    private:
        Farlor::Vector3 m_pos;
        Farlor::Vector3 m_dir;
    };

    class SphereGeometry : public GeometryBootstrapper::Geometry
    {
    public:
        explicit SphereGeometry(Farlor::Vector3 pos, float radius, float fov);
        virtual GeometryBootstrapper::Geometry::SampleRay GetSampleRay() const override;

    private:
        Farlor::Vector3 m_pos;
        float m_radius;
        float m_fov;
    };
}