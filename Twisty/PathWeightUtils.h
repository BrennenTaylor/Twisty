#pragma once

#if defined(USE_CUDA)
#include <cuda_occupancy.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

#else
#define __device__
#define __host__
#endif

#include <boost/multiprecision/cpp_dec_float.hpp>

#include "Curve.h"

#include <fstream>
#include <functional>
#include <map>
#include <vector>

#define BigFloatMultiprecision

namespace twisty {
namespace PerturbUtils {
    struct BoundaryConditions;
}
}  // namespace twisty

namespace twisty {
enum class WeightingMethod : int32_t { RadiativeTransfer = 0, SimplifiedModel = 1 };

struct WeightingParameters {
    float mu = 0.1f;
    uint32_t numStepsInt = 2000;
    float minBound = 0.0f;
    float maxBound = 100.0f;
    float eps = 0.01f;

    float scatter = 0.0f;
    float absorption = 0.0f;

    uint32_t numCurvatureSteps = 10000;

    WeightingMethod weightingMethod = WeightingMethod::RadiativeTransfer;
};

namespace PathWeighting {
    // Used for the Path Integral Radiative Transfer Weighting
    float SimpleGaussianPhase(float evalLocation, float mu);
    float GaussianPhase(float evalLocation, float mu);

    // Simple Weight Function, for small segment work!
    float SimpleWeightFunction(float curvature);

    class BaseWeightLookupTable {
       public:
        BaseWeightLookupTable(const WeightingParameters &weightingParams, float ds,
              float minCurvature, float maxCurvature);
        virtual ~BaseWeightLookupTable();

        float InterpolateWeightValues(float curvature) const;

        const std::vector<float> &AccessLookupTable() const { return m_lookupTable; }

        float GetMinSegmentWeight() const { return m_minSegmentWeight; }

        float GetMaxSegmentWeight() const { return m_maxSegmentWeight; }

        float GetMinCurvature() const { return m_minCurvature; }

        float GetMaxCurvature() const { return m_maxCurvature; }

        float GetDs() const { return m_ds; }

        float GetCurvatureStepSize() const { return m_curvatureStepSize; }

        WeightingParameters GetWeightingParams() const { return m_weightingParams; }

        void ExportValues(const std::string &relativePath) const;

       protected:
        // virtual double Integrate(double curvature, const WeightingParameters&
        // weightingParams, double ds) const = 0;
        virtual const std::string ExportFilename() const = 0;

        float m_minCurvature;
        float m_maxCurvature;

        float m_minSegmentWeight = 0.0;
        float m_maxSegmentWeight = 0.0;

        float m_curvatureStepSize;
        std::vector<float> m_lookupTable;

        float m_ds;
        WeightingParameters m_weightingParams;
    };

    class WeightLookupTableIntegral : public BaseWeightLookupTable {
       public:
        WeightLookupTableIntegral(const WeightingParameters &weightingParams, float ds);
        virtual ~WeightLookupTableIntegral();

       protected:
        float Integrate(
              float curvature, const WeightingParameters &weightingParams, float ds) const;
        virtual const std::string ExportFilename() const override
        {
            return "WeightLookupTableIntegral_Values.csv";
        }
    };

    // The ICTT27 Weighting function integral
    class SimpleWeightLookupTable : public BaseWeightLookupTable {
       public:
        SimpleWeightLookupTable(const WeightingParameters &weightingParams, float ds);
        virtual ~SimpleWeightLookupTable();

        virtual const std::string ExportFilename() const override
        {
            return "SimpleWeightLookupTable_Values.csv";
        }

       protected:
    };

    struct MinMaxCurvature {
        float minCurvature = 0.0f;
        float maxCurvature = 0.0f;
    };
    MinMaxCurvature CalcMinMaxCurvature(const twisty::WeightingParameters &wp, float ds);

    namespace NormalizerStuff {
        typedef boost::multiprecision::cpp_dec_float_100 NormalizerDoubleType;

        class BaseNormalizer {
           public:
            BaseNormalizer() { }

            virtual ~BaseNormalizer() { }

            virtual NormalizerDoubleType eval(int order, float r) const { return 1.0; }
        };

        class FN : public BaseNormalizer {
           public:
            FN(int samples, int numIntegrationSamples, int orders, const double &zMin,
                  const double &zMax);
            FN(std::ifstream &inFile);

            ~FN() { }

            virtual NormalizerDoubleType eval(int order, float r) const override;

            float minimum() const { return zMin; }
            float maximum() const { return zMax; }
            int samples() const { return nb_samples; }
            int orders() const { return nb_orders; }

            void WriteToFile(std::ofstream &outFile);

           private:
            int32_t nb_samples;
            int32_t nb_orders;
            float zMin;
            float zMax;

            std::map<int, std::vector<NormalizerDoubleType>> fNsets;
            void init(int numIntegrationSamples);

            void init_fromFile(std::ifstream &inFile);
        };

        NormalizerDoubleType Norm(int numberOfSegments, float ds,
              twisty::PerturbUtils::BoundaryConditions boundaryConditions);

        std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> GetNormalizer(
              uint32_t numSegments);
    }  // namespace NormalizerStuff
}  // namespace PathWeighting
}  // namespace twisty