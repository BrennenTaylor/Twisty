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
    float bias = 1.0f;

    float scatter = 0.0f;
    float absorption = 0.0f;

    uint32_t numCurvatureSteps = 10000;

    WeightingMethod weightingMethod = WeightingMethod::RadiativeTransfer;

    std::pair<std::string, uint64_t> GenerateStringUUID() const
    {
        std::stringstream uuid;
        std::ios_base::fmtflags flags = uuid.flags();

        {
            uint32_t value = 0;
            memcpy(&value, &mu, 4);
            uuid << "mu_0x" << std::uppercase << std::setfill('0') << std::setw(8) << std::hex
                 << value;
        }
        uuid.flags(flags);
        uuid << "_numStepsInt_" << numStepsInt;

        {
            uint32_t value = 0;
            memcpy(&value, &minBound, 4);
            uuid << "_minBound_0x" << std::uppercase << std::setfill('0') << std::setw(8)
                 << std::hex << value;
        }
        {
            uint32_t value = 0;
            memcpy(&value, &maxBound, 4);
            uuid << "_maxBound_0x" << std::uppercase << std::setfill('0') << std::setw(8)
                 << std::hex << value;
        }
        {
            uint32_t value = 0;
            memcpy(&value, &eps, 4);
            uuid << "_eps_0x" << std::uppercase << std::setfill('0') << std::setw(8) << std::hex
                 << value;
        }
        {
            uint32_t value = 0;
            memcpy(&value, &scatter, 4);
            uuid << "_scatter_0x" << std::uppercase << std::setfill('0') << std::setw(8)
                 << std::hex << value;
        }
        {
            uint32_t value = 0;
            memcpy(&value, &absorption, 4);
            uuid << "_absorption_0x" << std::uppercase << std::setfill('0') << std::setw(8)
                 << std::hex << value;
        }
        {
            uint32_t value = 0;
            memcpy(&value, &bias, 4);
            uuid << "_bias_0x" << std::uppercase << std::setfill('0') << std::setw(8) << std::hex
                 << value;
        }
        uuid.flags(flags);
        uuid << "_numCurvatureSteps_" << numCurvatureSteps << "_weightingMethod_"
             << (int32_t)weightingMethod;

        return std::make_pair<std::string, uint64_t>(
              uuid.str(), std::hash<std::string> {}(uuid.str()));
    }
};

namespace PathWeighting {
    // Used for the Path Integral Radiative Transfer Weighting
    float GaussianPhase(float evalLocation, float mu);

    double IntegrandRT(const double p, const double kds, const double bds, double eps, double mu);

    // Simple Weight Function, for small segment work!
    float SimpleWeightFunction(float curvature);

    class BaseWeightLookupTable {
       public:
        BaseWeightLookupTable(const WeightingParameters &weightingParams, const float ds,
              const float minCurvature, const float maxCurvature);
        virtual ~BaseWeightLookupTable() = default;

        void InitializeFromStream(std::ifstream &fileStream);
        void StreamOut(std::ostream &outputStream) const;

        virtual void Initialize() = 0;

        const std::vector<float> &AccessLookupTable() const { return m_lookupTable; }

        float GetMinSegmentWeight() const { return m_minSegmentWeight; }
        float GetMaxSegmentWeight() const { return m_maxSegmentWeight; }

        float GetMinCurvature() const { return m_minCurvature; }
        float GetMaxCurvature() const { return m_maxCurvature; }

        float GetDs() const { return m_ds; }

        float GetCurvatureStepSize() const { return m_curvatureStepSize; }

        WeightingParameters GetWeightingParams() const { return m_weightingParams; }

        void ExportValues(const std::string &relativePath) const;
        void ExportValues(const std::string &relativePath, const std::string &filename) const;

        void UpdateUUID();

       protected:
        virtual const std::string ExportFilename() const = 0;

        WeightingParameters m_weightingParams;
        std::pair<std::string, uint64_t> m_wpUUID;

        float m_ds = 0.0f;
        std::pair<std::string, uint64_t> m_wtUUID;

        float m_minCurvature = 0.0f;
        float m_maxCurvature = 0.0f;
        float m_curvatureStepSize = 0.0f;

        float m_minSegmentWeight = 0.0f;
        float m_maxSegmentWeight = 0.0f;

        std::vector<float> m_lookupTable;
    };

    // Cached Multi-Arclength Weight Lookup Table
    class CachedMultiArclengthWeightLookupTable {
       public:
        CachedMultiArclengthWeightLookupTable(const WeightingParameters &weightingParams,
              float minDs, float maxDs, uint32_t numDsSteps);

        // Currently just gets the closest table, but could be improved to do bilinear interpolation
        BaseWeightLookupTable *GetWeightLookupTable(float ds) const;

        void Initialize();
        void InitializeFromStream(std::ifstream &fileStream);

        std::pair<std::string, uint64_t> GetUUID() const { return m_cwtUUID; }
        void UpdateUUID();

       private:
        void CacheWeightTable();

       private:
        WeightingParameters m_weightingParams;
        uint32_t m_numDsSteps;
        float m_minDs;
        float m_maxDs;

        std::pair<std::string, uint64_t> m_cwtUUID;
        std::vector<std::unique_ptr<BaseWeightLookupTable>> m_weightLookupTables;
    };

    class WeightLookupTableIntegral : public BaseWeightLookupTable {
       public:
        WeightLookupTableIntegral(const WeightingParameters &weightingParams, float ds);
        virtual ~WeightLookupTableIntegral();

        virtual void Initialize() override;

        virtual const std::string ExportFilename() const override { return "RT.csv"; }

       private:
        float Integrate(
              float curvature, const WeightingParameters &weightingParams, float ds) const;
    };

    // The ICTT27 Weighting function integral
    class SimpleWeightLookupTable : public BaseWeightLookupTable {
       public:
        SimpleWeightLookupTable(const WeightingParameters &weightingParams, float ds);
        virtual ~SimpleWeightLookupTable();

        virtual void Initialize() override;

        virtual const std::string ExportFilename() const override { return "Simple.csv"; }
    };

    struct MinMaxCurvature {
        float minCurvature = 0.0f;
        float maxCurvature = 0.0f;
    };
    MinMaxCurvature CalcMinMaxCurvatureRadiativeTransfer(
          const twisty::WeightingParameters &wp, float ds);

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
