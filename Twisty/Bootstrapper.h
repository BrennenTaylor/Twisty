// /**
//  * @file Bootstrapper.h
//  * @author Brennen Taylor (brtaylor1001@gmail.com)
//  * @brief
//  * @version 0.1
//  * @date 2019-03-14
//  *
//  * @copyright Copyright (c) 2019
//  *
//  */

// #pragma once

// #define _USE_MATH_DEFINES
// #include "Curve.h"

// #include "BezierCurve.h"
// #include "PerturbUtils.h"

// #include "CurvePerturbUtils.h"

// #include <FMath/Vector3.h>

// #include <memory>

// namespace twisty {
// /**
//  * @brief Base bootstrapper class, responsible for creating a seed curve based
//  * off scene parameters. The bootstrapper is a state based machine, once a curve
//  * has been generated, it remains cached until reset is called. Until we call
//  * reset, it will always return the same cached seed bezier, control points, and
//  * curve
//  */
// class Bootstrapper {
//    public:
//     class Geometry {
//        public:
//         struct SampleRay {
//             glm::vec3 m_pos;
//             glm::vec3 m_dir;
//         };

//        public:
//         Geometry() { }

//         virtual SampleRay GetSampleRay() const = 0;
//     };

//     class RayGeometry : public Geometry {
//        public:
//         RayGeometry(glm::vec3 start, glm::vec3 dir);
//         virtual Geometry::SampleRay GetSampleRay() const override;

//        private:
//         glm::vec3 m_pos;
//         glm::vec3 m_dir;
//     };

//     class SphereGeometry : public Geometry {
//        public:
//         SphereGeometry(glm::vec3 pos, float radius, float fov);
//         virtual Geometry::SampleRay GetSampleRay() const override;

//        private:
//         glm::vec3 m_pos;
//         float m_radius;
//         float m_fov;
//     };

//    public:
//     // Represents a bezier curve
//     struct BezierInfo {
//         glm::vec3 m_controlPt0 = glm::vec3(0.0, 0.0, 0.0);
//         glm::vec3 m_controlPt1 = glm::vec3(0.0, 0.0, 0.0);
//         glm::vec3 m_controlPt2 = glm::vec3(0.0, 0.0, 0.0);
//         glm::vec3 m_controlPt3 = glm::vec3(0.0, 0.0, 0.0);
//         glm::vec3 m_controlPt4 = glm::vec3(0.0, 0.0, 0.0);
//     };

//    public:
//     // Bootstrapper();
//     // Samples from geometry
//     Bootstrapper(const twisty::PerturbUtils::BoundaryConditions &problemGeoemtry);
//     ~Bootstrapper();

//     glm::vec3 GetStartPosition() const;
//     glm::vec3 GetStartNormal() const;
//     glm::vec3 GetTargetPosition() const;
//     glm::vec3 GetTargetNormal() const;

//     // std::unique_ptr<Curve> CreateCurve(
//     //       uint32_t numSegments, float targetArclength, uint32_t generationSeed) const;
//     std::unique_ptr<Curve> CreateCurveGeometricSafe(
//           uint32_t numSegments, float targetArclength) const;

//     static float CalculateMinimumArclength(const uint32_t numSegments,
//           const glm::vec3 &startPos, const glm::vec3 &endPos);

//    private:
//     // std::unique_ptr<Curve> ToDiscreteFSCurve(uint32_t m_numSegments, BezierCurve5 &curve) const;

//    private:
//     twisty::PerturbUtils::BoundaryConditions m_experimentGeometry;
// };
// }  // namespace twisty