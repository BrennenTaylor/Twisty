#pragma once

#include "Bootstrapper.h"

#include <cstdint>

namespace twisty
{
    namespace testing
    {
        class LinearBootstrapper : public Bootstrapper
        {
        public:
            explicit LinearBootstrapper();

            virtual std::unique_ptr<Curve> CreateCurve(uint32_t numSegments) override;

        protected:
            virtual void BeginReset() override;
            virtual void EndReset() override;
        };


        // Quadratic bootstrapper
        class QuadraticBootstrapper : public Bootstrapper
        {
        public:
            explicit QuadraticBootstrapper();

            virtual std::unique_ptr<Curve> CreateCurve(uint32_t numSegments) override;

        protected:
            virtual void BeginReset() override;
            virtual void EndReset() override;
        };

        // Quadratic bootstrapper
        class CircleBootstrapper : public Bootstrapper
        {
        public:
            explicit CircleBootstrapper();

            virtual std::unique_ptr<Curve> CreateCurve(uint32_t numSegments) override;

        protected:
            virtual void BeginReset() override;
            virtual void EndReset() override;

        private:
            float m_radius;
        };
    }
}