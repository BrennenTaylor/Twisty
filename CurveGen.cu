#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace twisty
{
    // These are side structures for the cuda side algorithm
    // TODO: Measure the optimal organization for this structure
    struct DeviceCurve
    {
        float m_x0x;
        float m_x0y;
        float m_x0z;

        float m_x1x;
        float m_x1y;
        float m_x1z;

        float m_n0x;
        float m_n0y;
        float m_n0z;

        float m_n1x;
        float m_n1y;
        float m_n1z;

        float m_arclength;
        uint32_t m_numSegments;
        float __pad[2];
    };

    // TODO: These should store the premultiplied matrices...?
    struct DeviceSegment
    {
        float m_curvature;
        float m_torsion;
        float __pad[2];
    };

    // Settings for the different stages
    struct Settings_RandomNodeSelect
    {
        uint32_t m_maxSegments;
        float __pad[3];
    };

    struct Settings_CurvetureMod
    {
        float m_minMod;
        float m_maxMod;
        float __pad[2];
    };

    __global__ void SelectRandomNodes();
}