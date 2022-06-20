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
    double mu = 0.1;
    uint32_t numStepsInt = 2000;
    double minBound = 0.0;
    double maxBound = 100.0;
    double eps = 0.01;

    double scatter = 0.0;
    std::vector<double> scatterValues;
    double absorbtion = 0.0;

    uint32_t numCurvatureSteps = 10000;

    WeightingMethod weightingMethod = WeightingMethod::RadiativeTransfer;

    WeightingParameters()
        : scatterValues()
    {
        scatterValues.push_back(0.0);
    }
};

namespace PathWeighting {
    // Used for the Path Integral Radiative Transfer Weighting
    double SimpleGaussianPhase(double evalLocation, double mu);
    double GaussianPhase(double evalLocation, double mu);

    // Simple Weight Function, for small segment work!
    double SimpleWeightFunction(double curvature);

    class BaseWeightLookupTable {
       public:
        BaseWeightLookupTable(const WeightingParameters &weightingParams, double ds,
              double minCurvature, double maxCurvature);
        virtual ~BaseWeightLookupTable();

        double InterpolateWeightValues(double curvature) const;

        const std::vector<double> &AccessLookupTable() const { return m_lookupTable; }

        double GetMinSegmentWeight() const { return m_minSegmentWeight; }

        double GetMaxSegmentWeight() const { return m_maxSegmentWeight; }

        double GetMinCurvature() const { return m_minCurvature; }

        double GetMaxCurvature() const { return m_maxCurvature; }

        double GetDs() const { return m_ds; }

        double GetCurvatureStepSize() const { return m_curvatureStepSize; }

        WeightingParameters GetWeightingParams() const { return m_weightingParams; }

        void ExportValues(const std::string &relativePath) const;

       protected:
        // virtual double Integrate(double curvature, const WeightingParameters&
        // weightingParams, double ds) const = 0;
        virtual const std::string ExportFilename() const = 0;

        double m_minCurvature;
        double m_maxCurvature;

        double m_minSegmentWeight = 0.0;
        double m_maxSegmentWeight = 0.0;

        double m_curvatureStepSize;
        std::vector<double> m_lookupTable;

        double m_ds;
        WeightingParameters m_weightingParams;
    };

    class WeightLookupTableIntegral : public BaseWeightLookupTable {
       public:
        WeightLookupTableIntegral(const WeightingParameters &weightingParams, double ds);
        virtual ~WeightLookupTableIntegral();

       protected:
        double Integrate(
              double curvature, const WeightingParameters &weightingParams, double ds) const;
        virtual const std::string ExportFilename() const override
        {
            return "WeightLookupTableIntegral_Values.csv";
        }
    };

    // The ICTT27 Weighting function integral
    class SimpleWeightLookupTable : public BaseWeightLookupTable {
       public:
        SimpleWeightLookupTable(const WeightingParameters &weightingParams, double ds);
        virtual ~SimpleWeightLookupTable();

        virtual const std::string ExportFilename() const override
        {
            return "SimpleWeightLookupTable_Values.csv";
        }

       protected:
    };

    struct MinMaxCurvature {
        double minCurvature = 0.0;
        double maxCurvature = 0.0;
    };
    MinMaxCurvature CalcMinMaxCurvature(const twisty::WeightingParameters &wp, double ds);

    namespace NormalizerStuff {
        typedef boost::multiprecision::cpp_dec_float_100 NormalizerDoubleType;

        class BaseNormalizer {
           public:
            BaseNormalizer() { }

            virtual ~BaseNormalizer() { }

            virtual NormalizerDoubleType eval(int order, double r) const { return 1.0; }
        };

        class FN : public BaseNormalizer {
           public:
            FN(int samples, int numIntegrationSamples, int orders, const double &zMin,
                  const double &zMax);
            FN(std::ifstream &inFile);

            ~FN() { }

            virtual NormalizerDoubleType eval(int order, double r) const override;

            double minimum() const { return zMin; }
            double maximum() const { return zMax; }
            int samples() const { return nb_samples; }
            int orders() const { return nb_orders; }

            void WriteToFile(std::ofstream &outFile);

           private:
            int nb_samples;
            int nb_orders;
            double zMin;
            double zMax;

            std::map<int, std::vector<NormalizerDoubleType>> fNsets;
            void init(int numIntegrationSamples);

            void init_fromFile(std::ifstream &inFile);
        };

        NormalizerDoubleType Norm(int numberOfSegments, double ds,
              twisty::PerturbUtils::BoundaryConditions boundaryConditions);

        std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> GetNormalizer(
              uint32_t numSegments);
    }  // namespace NormalizerStuff
}  // namespace PathWeighting
}  // namespace twisty