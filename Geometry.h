#pragma once

#include <FMath/FMath.h>

namespace twisty
{
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
}