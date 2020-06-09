#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace twisty
{
    struct DeviceBasis
    {
        float3 m_tangent;
        float3 m_normal;
        float3 m_binormal;
    };

    // These are side structures for the cuda side algorithm
    // TODO: Measure the optimal organization for this structure
    struct DeviceCurve
    {
        float3 m_basePos;
        float3 m_baseTangent;
        float3 m_baseNormal;
        float3 m_baseBinormal;
        float3 m_targetPos;
        float3 m_targetTangent;
        float m_arclength;
        float m_minCurvature;
        float m_maxCurvature;
        float m_minTorsion;
        float m_maxTorsion;
        uint32_t m_numSegments;

        //float __pad[0];
    };
    static_assert(sizeof(DeviceCurve) % 16 == 0, "sizeof(DeviceCurve) % 16 == 0");

    //// TODO: These should store the premultiplied matrices...?
    //struct DeviceSegment
    //{
    //    float m_curvature;
    //    float m_torsion;
    //    // Rotation Matrix
    //    float m_rotationU[3 * 3];
    //    
    //    // Required pad for alignment
    //    float __pad[1];
    //};
    //static_assert(sizeof(DeviceSegment) % 16 == 0, "sizeof(DeviceSegment) % 16 == 0");

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
}