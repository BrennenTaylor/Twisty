#include "Geometry.h"

#include "Sample.h"

#include <random>

namespace twisty
{
    RayGeometry::RayGeometry(Farlor::Vector3 start, Farlor::Vector3 dir)
        : m_pos(start)
        , m_dir(dir)
    {
    }

    GeometryBootstrapper::Geometry::SampleRay RayGeometry::GetSampleRay() const
    {
        return SampleRay{ m_pos, m_dir };
    }

    SphereGeometry::SphereGeometry(Farlor::Vector3 pos, float radius, float fov)
        : m_pos{ pos }
        , m_radius{ radius }
        , m_fov{ fov }
    {
    }

    GeometryBootstrapper::Geometry::SampleRay SphereGeometry::GetSampleRay() const
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0f, 1.0f);
        float rand0 = static_cast<float>(dist(gen));
        float rand1 = static_cast<float>(dist(gen));
        Farlor::Vector3 sphereSample = Sample::UniformSphere(rand0, rand1);
        return SampleRay{ m_pos, sphereSample.Normalized() };
    }
}