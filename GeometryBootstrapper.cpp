#include "GeometryBootstrapper.h"

#include "Sample.h"

#include <algorithm>
#include <assert.h>
#include <math.h>
#include <random>
#include <fstream>
#include <filesystem>

namespace twisty
{
    GeometryBootstrapper::GeometryBootstrapper(const Geometry& emitterGeometry, const Geometry& recieverGeometry, float targetArclength, uint32_t randomSeed)
        : Bootstrapper(targetArclength, randomSeed)
    {
        auto sample0 = emitterGeometry.GetSampleRay();
        m_startPos = sample0.m_pos;
        m_startDir = sample0.m_dir;

        auto sample1 = recieverGeometry.GetSampleRay();
        m_endPos = sample1.m_pos;
        m_endDir = sample1.m_dir;
    }

    void GeometryBootstrapper::BeginReset()
    {
    }

    void GeometryBootstrapper::EndReset()
    {
    }
}