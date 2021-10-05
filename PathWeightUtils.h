#pragma once

#include "Curve.h"
#include "PerturbUtils.h"

#include <boost/serialization/nvp.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <fstream>
#include <functional>
#include <map>
#include <vector>

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
        // Used for the Path Integral Radiative Transfer Weighting
        double SimpleGaussianPhase(double evalLocation, double mu);
        double GaussianPhase(double evalLocation, double mu);



        // Simple Weight Function, for small segment work!
        double SimpleWeightFunction(double curvature);


        // TODO: Describe this with a useful comment
        //class IntegralStrategy
        //{
        //public:
        //    IntegralStrategy(const WeightingParameters& weightingParams, double ds);
        //    virtual ~IntegralStrategy();

        //    virtual double Integrate(double curvature) const = 0;

        //    double GetDs() const
        //    {
        //        return m_ds;
        //    }

        //    const WeightingParameters& GetWeightingParams() const
        //    {
        //        return m_weightingParams;
        //    }

        //protected:

        //    double m_ds;
        //    WeightingParameters m_weightingParams;
        //};

        class BaseWeightLookupTable
        {
        public:
            BaseWeightLookupTable(const WeightingParameters& weightingParams, double ds, double minCurvature, double maxCurvature);
            virtual ~BaseWeightLookupTable();

            double InterpolateWeightValues(double curvature) const;

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

            double GetMinCurvature() const
            {
                return m_minCurvature;
            }

            double GetMaxCurvature() const
            {
                return m_maxCurvature;
            }

            double GetDs() const {
                return m_ds;
            }

            WeightingParameters GetWeightingParams() const {
                return m_weightingParams;
            }

            void ExportValues(std::string relativePath);

        protected:
            //virtual double Integrate(double curvature, const WeightingParameters& weightingParams, double ds) const = 0;
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

        class WeightLookupTableIntegral : public BaseWeightLookupTable
        {
        public:
            WeightLookupTableIntegral(const WeightingParameters& weightingParams, double ds);
            virtual ~WeightLookupTableIntegral();

        protected:
            double Integrate(double curvature, const WeightingParameters& weightingParams, double ds) const;
            virtual const std::string ExportFilename() const override {
                return "WeightLookupTableIntegral_Values.csv";
            }

            //RegularizedIntegral m_regularizedIntegral;
        };

        // The ICTT27 Weighting function integral
        class SimpleWeightLookupTable : public BaseWeightLookupTable
        {
        public:
            SimpleWeightLookupTable(const WeightingParameters& weightingParams, double ds);
            virtual ~SimpleWeightLookupTable();

            //virtual double Integrate(double curvature, const WeightingParameters& weightingParams, double ds) const override;

            virtual const std::string ExportFilename() const override{
                return "SimpleWeightLookupTable_Values.csv";
            }

        protected:

        };

        void CalcMinMaxCurvature(double& minCurvature, double& maxCurvature, double ds);

        // Given a vector of curvatures, 1 per segement of a path, weight the path and return the long10 of the weight
        // TODO: We want to use span here, but currently not supported in compiler (is, but have to force latest verison).
        // Assumes that integral matches weighting params
        double WeightCurveViaCurvatureLog10(float* pCurvatureStart, uint32_t numCurvatures, const BaseWeightLookupTable& weightIntegral);


        double SimpleWeightCurveViaTangentDotProductLog10(Farlor::Vector3* pTangents, uint32_t numCurvatures, const BaseWeightLookupTable& weightIntegral);

        namespace NormalizerStuff
        {
            typedef boost::multiprecision::cpp_dec_float_100 NormalizerDoubleType;

            NormalizerDoubleType f2_formula(const double z);

            class BaseNormalizer
            {
            public:
                BaseNormalizer()
                {
                }

                virtual ~BaseNormalizer()
                {
                }

                virtual NormalizerDoubleType eval(int order, double r) const
                {
                    return 1.0;
                }
            };

            class FN : public BaseNormalizer
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

                virtual NormalizerDoubleType eval(int order, double r) const override;

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

            NormalizerDoubleType psd_one(const BaseNormalizer &fn, int M, const double s, const Farlor::Vector3& X, const Farlor::Vector3& N, const Farlor::Vector3& Np, const Farlor::Vector3& beta);

            NormalizerDoubleType Norm(const BaseNormalizer &fn, int M, double z, double s);

            NormalizerDoubleType psd_n(const BaseNormalizer&fn, int M, const double s, const Farlor::Vector3& X, const Farlor::Vector3& N, const Farlor::Vector3& Np,
                const std::vector<Farlor::Vector3>& beta);

            NormalizerDoubleType likelihood(const BaseNormalizer&fn, int numSegments, const double arclength, const Farlor::Vector3& endPosition, const Farlor::Vector3& startPosition,
                const Farlor::Vector3& N, const Farlor::Vector3& Np, const std::vector<Farlor::Vector3> &beta0, const std::vector<Farlor::Vector3> &betastar);

            NormalizerDoubleType CalculateLikelihood(const BaseNormalizer& fn, int numSegments, const twisty::PerturbUtils::BoundrayConditions& boundaryConditions,
                std::vector<Farlor::Vector3>& oldTangents, std::vector<Farlor::Vector3>& newTangents);


            std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> GetNormalizer(uint32_t numSegments);
        }
    }
}