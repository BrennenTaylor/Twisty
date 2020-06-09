#pragma once

#include "Curve.h"

#include <boost\multiprecision\cpp_dec_float.hpp>

#include <functional>
#include <vector>

// TODO: Track down why this was added and add a useful comment.
#ifdef __CUDACC__
#define BOOST_PP_VARIADICS 0
#endif
//#include <boost/multiprecision/cpp_dec_float.hpp>

#define BigFloatMultiprecision

namespace twisty
{
    using TwistyWeight = double;
    // TODO: Try with cpp_dec_float_50
    
#ifdef BigFloatMultiprecision
    using BigFloat = boost::multiprecision::cpp_dec_float_100;
#else
    using BigFloat = double;
#endif

    namespace PathSpaceUtils
    {
        class IntegralStrategy;

        double SimpleGaussianPhase(double evalLocation, double mu);
        double GaussianPhase(double evalLocation, double mu);

        //BigFloat WeightPathBigFloat(const Curve& path
        //    , std::function<double(Farlor::Vector3 pos)> absorbtionFunc
        //    , std::function<double(Farlor::Vector3 pos)> scatteringFunc
        //    , const IntegralStrategy& strategy);

        //TwistyWeight WeightPathLogWeight(const Curve& path
        //    , std::function<double(Farlor::Vector3 pos)> absorbtionFunc
        //    , std::function<double(Farlor::Vector3 pos)> scatteringFunc
        //    , const IntegralStrategy& strategy);

        //TwistyWeight WeightSegment(float segmentCurvature
        //    , const Farlor::Vector3& segmentPos
        //    , const Farlor::Vector3& segmentTan
        //    , std::function<double(Farlor::Vector3 pos)> absorbtionFunc
        //    , std::function<double(Farlor::Vector3 pos)> scatteringFunc
        //    , const IntegralStrategy& strategy
        //    , double minClip = -FLT_MAX
        //    , double maxClip = FLT_MAX);

        class IntegralStrategy
        {
        public:
            IntegralStrategy(double ds);
            virtual ~IntegralStrategy();

            virtual TwistyWeight Eval(double density, double absorbtion, double curvature) const;
            virtual TwistyWeight Integrate(double density, double curvature) const = 0;

        protected:
            double m_ds;
        };

        //class LogIntegralStrategy : public IntegralStrategy
        //{
        //public:
        //    LogIntegralStrategy(double ds);
        //    virtual ~LogIntegralStrategy();

        //    virtual TwistyWeight Eval(double density, double absorbtion, double curvature) const override final;
        //    virtual TwistyWeight Integrate(double density, double curvature) const = 0;
        //};

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
            RegularizedIntegral(double ds, double mu, uint32_t numStepsInt, double minBound, double maxBound, double eps);
            virtual ~RegularizedIntegral();

            virtual TwistyWeight Integrate(double scattering, double curvature) const override;

        private:
            double m_mu;
            uint32_t m_numStepsInt;
            double m_minBound;
            double m_maxBound;
            double m_eps;
        };

        class WeightLookupTableIntegral : public IntegralStrategy
        {
        public:
            WeightLookupTableIntegral(double ds, double mu, uint32_t numStepsInt, double minBound, double maxBound, double eps,
                double minCurvature, double maxCurvature, uint32_t numCurvatureSteps, double scattering);
            virtual ~WeightLookupTableIntegral();

            virtual TwistyWeight Integrate(double scattering, double curvature) const override;
            
            const std::vector<TwistyWeight>& AccessLookupTable() const {
                return m_lookupTable;
            }

            TwistyWeight GetMinSegmentWeight() const
            {
                return m_minSegmentWeight;
            }

            TwistyWeight GetMaxSegmentWeight() const
            {
                return m_maxSegmentWeight;
            }

        private:
            double m_mu;
            uint32_t m_numStepsInt;
            double m_minBound;
            double m_maxBound;
            double m_eps;
            double m_minCurvature;
            double m_maxCurvature;
            uint32_t m_numCurvatureSteps;
            RegularizedIntegral m_regularizedIntegral;

            TwistyWeight m_minSegmentWeight = 0.0;
            TwistyWeight m_maxSegmentWeight = 0.0;

            double m_curvatureStepSize;
            std::vector<TwistyWeight> m_lookupTable;
        };

        /*class LogWeightLookupTableIntegral : public LogIntegralStrategy
        {
        public:
            LogWeightLookupTableIntegral(double ds, double mu, uint32_t numStepsInt, double minBound, double maxBound, double eps,
                double minCurvature, double maxCurvature, uint32_t numCurvatureSteps, double scattering);
            virtual ~LogWeightLookupTableIntegral();

            virtual TwistyWeight Integrate(double scattering, double curvature) const override;

            const std::vector<TwistyWeight>& AccessLookupTable() const {
                return m_lookupTable;
            }

            TwistyWeight GetMinSegmentWeight() const
            {
                return m_minSegmentWeight;
            }

            TwistyWeight GetMaxSegmentWeight() const
            {
                return m_maxSegmentWeight;
            }

        private:
            double m_mu;
            uint32_t m_numStepsInt;
            double m_minBound;
            double m_maxBound;
            double m_eps;
            double m_minCurvature;
            double m_maxCurvature;
            uint32_t m_numCurvatureSteps;
            RegularizedIntegral m_regularizedIntegral;

            TwistyWeight m_minSegmentWeight = 0.0;
            TwistyWeight m_maxSegmentWeight = 0.0;

            double m_curvatureStepSize;
            std::vector<TwistyWeight> m_lookupTable;
        };*/
    }
}