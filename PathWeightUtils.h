#pragma once

#include "Curve.h"
#include "PerturbUtils.h"

#include <boost/serialization/nvp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <fstream>
#include <functional>
#include <map>
#include <vector>

// TODO: Track down why this was added and add a useful comment.
#ifdef __CUDACC__
#define BOOST_PP_VARIADICS 0
#endif
//#include <boost/multiprecision/cpp_dec_float.hpp>

#define BigFloatMultiprecision

namespace twisty
{
    struct WeightingParameters
    {
        double mu = 0.1;
        uint32_t numStepsInt = 2000;
        double minBound = 0.0;
        double maxBound = 100.0;
        double eps = 0.01;

        double scatter = 0.0;
        double absorbtion = 0.0;

        uint32_t numCurvatureSteps = 10000;
    };

    namespace PathWeighting
    {
        class IntegralStrategy;

        double SimpleGaussianPhase(double evalLocation, double mu);
        double GaussianPhase(double evalLocation, double mu);

        class IntegralStrategy
        {
        public:
            IntegralStrategy(const WeightingParameters& weightingParams, double ds);
            virtual ~IntegralStrategy();

            virtual double Eval(double density, double absorbtion, double curvature) const;
            virtual double Integrate(double density, double curvature) const = 0;

            double GetDs() const
            {
                return m_ds;
            }

            const WeightingParameters& GetWeightingParams() const
            {
                return m_weightingParams;
            }

        protected:
            double m_ds;
            WeightingParameters m_weightingParams;
        };

        /*
            Numerically evaluate path weight integral, introducing an epsilon term to
            make the integration numerically feasible.

            Parameters
            ----------

            mu : float
            Gaussian phase function width parameter.
            step : float
            Step size used for integration.
            bounds : 2 - tuple or None
            Minimum and maximum of interval to step over when evaluating the
            integral.
            eps : float
            A small value.The smaller this value, the more accurate the
            integration, but breaks down numerically if the value is too small.
            '''
        */
        class RegularizedIntegral : public IntegralStrategy
        {
        public:
            RegularizedIntegral(const WeightingParameters& weightParams, double ds);
            virtual ~RegularizedIntegral();

            virtual double Integrate(double scattering, double curvature) const override;
        };

        class WeightLookupTableIntegral : public IntegralStrategy
        {
        public:
            WeightLookupTableIntegral(const WeightingParameters& weightingParams, double ds);
            virtual ~WeightLookupTableIntegral();

            virtual double Integrate(double scattering, double curvature) const override;
            
            const std::vector<double>& AccessLookupTable() const {
                return m_lookupTable;
            }

            double GetMinSegmentWeight() const
            {
                return m_minSegmentWeight;
            }

            double GetMaxSegmentWeight() const
            {
                return m_maxSegmentWeight;
            }

            void ExportValues(std::string relativePath);

        private:
            double m_minCurvature;
            double m_maxCurvature;
            RegularizedIntegral m_regularizedIntegral;

            double m_minSegmentWeight = 0.0;
            double m_maxSegmentWeight = 0.0;

            double m_curvatureStepSize;
            std::vector<double> m_lookupTable;
        };

        void CalcMinMaxCurvature(double& minCurvature, double& maxCurvature, double ds);

        // Given a vector of curvatures, 1 per segement of a path, weight the path and return the long10 of the weight
        // TODO: We want to use span here, but currently not supported in compiler (is, but have to force latest verison).
        // Assumes that integral matches weighting params
        double WeightCurveViaCurvatureLog10(float* pCurvatureStart, uint32_t numCurvatures, const WeightLookupTableIntegral& weightIntegral);

        namespace NormalizerStuff
        {
            typedef boost::multiprecision::cpp_dec_float_100 NormalizerDoubleType;

            NormalizerDoubleType f2_formula(const double z);

            class FN
            {
            public:
                FN(int samples, int numIntegrationSamples, int orders, const double &rmn, const double &rmx)
                    : nb_samples(samples)
                    , nb_orders(orders)
                    , rmin(rmn)
                    , rmax(rmx)
                {
                    init(numIntegrationSamples);
                }

                FN(std::ifstream& inFile)
                    : nb_samples(0)
                    , nb_orders(0)
                    , rmin(0)
                    , rmax(0)
                {
                    init_fromFile(inFile);
                }

                ~FN() {}

                NormalizerDoubleType eval(int order, double r) const;

                double minimum() const { return rmin; }
                double maximum() const { return rmax; }
                int samples() const { return nb_samples; }
                int orders() const { return nb_orders; }

                void WriteToFile(std::ofstream& outFile);

            private:
                int nb_samples;
                int nb_orders;
                double rmin;
                double rmax;

                std::map<int, std::vector<NormalizerDoubleType>> fNsets;
                void init(int numIntegrationSamples);

                void init_fromFile(std::ifstream& inFile);
            };

            Farlor::Vector3 makeVector(double a, double b, double c);

            NormalizerDoubleType psd_one(const FN &fn, int M, const double s, const Farlor::Vector3& X, const Farlor::Vector3& N, const Farlor::Vector3& Np, const Farlor::Vector3& beta);

            NormalizerDoubleType Norm(const FN &fn, int M, double z, double s);

            NormalizerDoubleType psd_n(const FN &fn, int M, const double s, const Farlor::Vector3& X, const Farlor::Vector3& N, const Farlor::Vector3& Np,
                const std::vector<Farlor::Vector3>& beta);

            NormalizerDoubleType likelihood(const FN &fn, int numSegments, const double arclength, const Farlor::Vector3& endPosition, const Farlor::Vector3& startPosition,
                const Farlor::Vector3& N, const Farlor::Vector3& Np, const std::vector<Farlor::Vector3> &beta0, const std::vector<Farlor::Vector3> &betastar);

            NormalizerDoubleType CalculateLikelihood(const FN& fn, int numSegments, const twisty::PerturbUtils::BoundrayConditions& boundaryConditions,
                std::vector<Farlor::Vector3>& oldTangents, std::vector<Farlor::Vector3>& newTangents);


            std::unique_ptr<PathWeighting::NormalizerStuff::FN> GetNormalizer(uint32_t numSegments);
        }
    }
}