#include "PathWeightUtils.h"

#include "math_constants.h"

#include <algorithm>
#include <assert.h>

namespace twisty
{
    namespace PathSpaceUtils
    {
        // Parameterized simple gaussian function
        double SimpleGaussianPhase(double evalLocation, double mu)
        {
            double val = -mu * evalLocation * evalLocation;
            val *= 0.5;
            return std::exp(val);
        }

        // Parameterized gaussian function
        double GaussianPhase(double evalLocation, double mu)
        {
            double a = std::sqrt(CUDART_PI * mu * 0.5);
            double b = SimpleGaussianPhase(evalLocation, mu);
            double c = 1.0 - std::exp(-2.0 / mu);
            return (a * b) / c;
        }

        // Utility method for evaluating the weight of a single path
        //TwistyWeight WeightPathLogWeight(const Curve& path
        //    , std::function<double(Farlor::Vector3 pos)> absorbtionFunc
        //    , std::function<double(Farlor::Vector3 pos)> scatteringFunc
        //    , const IntegralStrategy& strategy)
        //{
        //    // We start the path weight at 0.0, as we add each segment's weight within the path into this for the final path weight value.
        //    TwistyWeight pathWeight = 0.0f;
        //    for (uint32_t segIdx = 0; segIdx < path.m_numSegments; ++segIdx)
        //    {
        //        auto segmentWeight = WeightSegment(path.m_curvatures[segIdx], path.m_positions[segIdx], path.m_tangents[segIdx], absorbtionFunc, scatteringFunc, strategy);
        //        // We are in log compressed weight form, and thus need to add here
        //        //std::cout << "Segment " << segIdx << " log weight: " << segmentWeight << std::endl;
        //        pathWeight += segmentWeight;
        //    }
        //    return pathWeight;
        //}

        // Utility method for evaluating the weight of a single path
        //BigFloat WeightPathBigFloat(const Curve& path
        //    , std::function<double(Farlor::Vector3 pos)> absorbtionFunc
        //    , std::function<double(Farlor::Vector3 pos)> scatteringFunc
        //    , const IntegralStrategy& strategy)
        //{
        //    // We start the path weight at 1.0, as we multiply each segment's weight within the path into this for the final path weight value.
        //    BigFloat pathWeight = 1.0f;
        //    for (uint32_t segIdx = 0; segIdx < path.m_numSegments; ++segIdx)
        //    {
        //        auto segmentWeight = WeightSegment(path.m_curvatures[segIdx], path.m_positions[segIdx], path.m_tangents[segIdx], absorbtionFunc, scatteringFunc, strategy);
        //        //std::cout << "Segment " << segIdx << " weight: " << segmentWeight << std::endl;
        //        pathWeight *= segmentWeight;
        //    }
        //    return pathWeight;
        //}

//        // Utility method for evaluating the weights of all nodes in a path
//        TwistyWeight WeightSegment(float segmentCurvature
//            , const Farlor::Vector3& segmentPos
//            , const Farlor::Vector3& segmentTan
//            , std::function<double(Farlor::Vector3 pos)> absorbtionFunc
//            , std::function<double(Farlor::Vector3 pos)> scatteringFunc
//            , const IntegralStrategy& strategy
//            , double minClip
//            , double maxClip)
//        {
//            // Pass in the location in world space we sample the absorbion and scattering from.
//            // Currently, this should probably be the start (or end) position of the segment.
//            double absorb = absorbtionFunc(segmentPos);
//            double density = scatteringFunc(segmentTan);
//            double curvature = segmentCurvature;
//            // Do we need min and max clip here?
////            BigFloat segWeight = strategy.Eval(density, absorb, curvature, ds, minClip, maxClip);
//            TwistyWeight segWeight = strategy.Eval(density, absorb, curvature);
//            //std::cout << "Weight Segment weight: " << segWeight << std::endl;
//            return segWeight;
//        }

        // Base Integral Strategy
        IntegralStrategy::IntegralStrategy(double ds)
            : m_ds(ds)
        {
        }

        IntegralStrategy::~IntegralStrategy()
        {
        }

        TwistyWeight IntegralStrategy::Eval(double density, double absorbtion, double curvature) const
        {
            TwistyWeight c = density + absorbtion;
            TwistyWeight constant = std::exp(-c * m_ds) / (2.0 * CUDART_PI_F * CUDART_PI_F);
            return constant * Integrate(density, curvature);
        }

        //// Base Log Integral Strategy
        //LogIntegralStrategy::LogIntegralStrategy(double ds)
        //    : IntegralStrategy(ds)
        //{
        //}

        //LogIntegralStrategy::~LogIntegralStrategy()
        //{
        //}

        //TwistyWeight LogIntegralStrategy::Eval(double density, double absorbtion, double curvature) const
        //{
        //    TwistyWeight c = density + absorbtion;
        //    TwistyWeight constant = std::exp(-c * m_ds) / (2.0 * CUDART_PI_F * CUDART_PI_F);
        //    return log(constant) + Integrate(density, curvature);
        //}

        // Regularized Integral
        RegularizedIntegral::RegularizedIntegral(double ds, double mu, uint32_t numStepsInt, double minBound, double maxBound, double eps)
            : IntegralStrategy(ds)
            , m_mu(mu)
            , m_numStepsInt(numStepsInt)
            , m_minBound(minBound)
            , m_maxBound(maxBound)
            , m_eps(eps)
        {
        }

        RegularizedIntegral::~RegularizedIntegral()
        {
        }

        TwistyWeight RegularizedIntegral::Integrate(double scattering, double curvature) const
        {
            double kds = curvature * m_ds;
            double bds = scattering * m_ds;

            auto Integrand = [this](double p, double kds, double bds) -> double
            {
                // Handle the case where curvature drops to 0
                /*if (kds == 0.0)
                {
                    return p;
                }*/

                double phaseFunction = GaussianPhase(p, m_mu);

                double scatteringTerm = p * std::exp(
                    bds * phaseFunction // scatter piece
                    - 1.0 * (m_eps * m_eps * p * p) / 2.0 // regularizer
                );
                //double regularizer = exp(-1.0 * (m_eps * m_eps * p  * p) / 2.0);

                double sinTerm = 0.0;
                if (kds != 0.0)
                {
                    sinTerm = sin(kds * p) / kds;
                }
                else
                {
                    sinTerm = p;
                }

                return scatteringTerm * sinTerm;// *regularizer;
            };

            // Perform first integration
            double firstVal = 0.0f;
            {
                double stepSize = (m_maxBound - m_minBound) / m_numStepsInt;
                for (uint32_t i = 0; i <= m_numStepsInt; ++i)
                {
                    double p = i * stepSize;
                    double left = Integrand(p, kds, bds);
                    //std::cout << "Integrand eval: " << left << std::endl;
                    firstVal += left * stepSize;
                }
            }

            // Note: This is added back in because it seems this allows the base kds == 0 case to work
            // Perform second integration
            double secondVal = 0.0f;
            {
                double stepSize = (m_maxBound - m_minBound) / m_numStepsInt;
                for (uint32_t i = 0; i <= m_numStepsInt; ++i)
                {
                    double p = i * stepSize;
                    double left = Integrand(p, 0.0, bds);
                    secondVal += left * stepSize;
                }
            }

            return firstVal / secondVal;
        }


        // Lookup table integrand
        WeightLookupTableIntegral::WeightLookupTableIntegral(double ds, double mu, uint32_t numStepsInt, double minBound, double maxBound, double eps,
            double minCurvature, double maxCurvature, uint32_t numCurvatureSteps, double scattering)
            : IntegralStrategy(ds)
            , m_mu(mu)
            , m_numStepsInt(numStepsInt)
            , m_minBound(minBound)
            , m_maxBound(maxBound)
            , m_eps(eps)
            , m_minCurvature(minCurvature)
            , m_maxCurvature(maxCurvature)
            , m_numCurvatureSteps(numCurvatureSteps)
            , m_regularizedIntegral(ds, mu, numStepsInt, minBound, maxBound, eps)
            , m_curvatureStepSize(0.0f)
            , m_lookupTable()
        {
            std::cout << "Calcuating path weight integral lookup table" << std::endl;

            m_curvatureStepSize = (m_maxCurvature - m_minCurvature) / m_numCurvatureSteps;
            m_lookupTable.clear();
            m_lookupTable.resize(m_numCurvatureSteps + 1u);

            // Handle first case
            {
                double value = m_regularizedIntegral.Integrate(scattering, m_minCurvature);
                m_lookupTable[0] = value;
            }

            double min = m_lookupTable[0];
            double max = m_lookupTable[0];

            uint32_t numInvalid = 0;

            for (uint32_t i = 1; i <= m_numCurvatureSteps; ++i)
            {
                double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
                double value = m_regularizedIntegral.Integrate(scattering, curvatureEval);

                if (value < 0.0)
                {
                    numInvalid++;
                    value = 0.0;
                }

                m_lookupTable[i] = value;

                if (value < min)
                {
                    min = value;
                }
                if (value > max)
                {
                    max = value;
                }
            }

            if (numInvalid > 0)
            {
                std::cout << "We calculated " << numInvalid << " negative table values and clamped them to zero" << std::endl;
            }

            m_minSegmentWeight = min;
            m_maxSegmentWeight = max;

            std::cout << "Finished path weight integral lookup table" << std::endl;
            std::cout << "\tMin Possible Weight Value: " << min << std::endl;
            std::cout << "\tMax Possible Weight Value: " << max << std::endl;
            //Parameters
            std::cout << "\tTable construction params: " << std::endl;
            std::cout << "\t\tmu: " << m_mu << std::endl;
            std::cout << "\t\tnumStepsInt: " << m_numStepsInt << std::endl;
            std::cout << "\t\tm_minBound: " << m_minBound << std::endl;
            std::cout << "\t\tm_maxBound: " << m_maxBound << std::endl;
            std::cout << "\t\tm_eps: " << m_eps << std::endl;
            std::cout << "\t\tm_minCurvature: " << m_minCurvature << std::endl;
            std::cout << "\t\tm_maxCurvature: " << m_maxCurvature << std::endl;
            std::cout << "\t\tm_numCurvatureSteps: " << m_numCurvatureSteps << std::endl;
        }

        WeightLookupTableIntegral::~WeightLookupTableIntegral()
        {
        }

        TwistyWeight WeightLookupTableIntegral::Integrate(double scattering, double curvature) const
        {
            assert(curvature >= m_minCurvature);

            if (curvature > m_maxCurvature)
            {
                std::cout << "Clamping to max curvature" << std::endl;
                curvature = m_maxCurvature;
            }

            double distance = curvature - m_minCurvature;
            double realIdx = distance / m_curvatureStepSize;
            uint32_t leftIdx = floor(realIdx);
            uint32_t rightIdx = leftIdx + 1;

            double leftLookup = m_lookupTable[leftIdx];
            double rightLookup = m_lookupTable[rightIdx];

            double leftDist = distance - (leftIdx * m_curvatureStepSize);

            double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);

            return interpolatedResult;
        }

        //// Lookup table integrand
        //LogWeightLookupTableIntegral::LogWeightLookupTableIntegral(double ds, double mu, uint32_t numStepsInt, double minBound, double maxBound, double eps,
        //    double minCurvature, double maxCurvature, uint32_t numCurvatureSteps, double scattering)
        //    : LogIntegralStrategy(ds)
        //    , m_mu(mu)
        //    , m_numStepsInt(numStepsInt)
        //    , m_minBound(minBound)
        //    , m_maxBound(maxBound)
        //    , m_eps(eps)
        //    , m_minCurvature(minCurvature)
        //    , m_maxCurvature(maxCurvature)
        //    , m_numCurvatureSteps(numCurvatureSteps)
        //    , m_regularizedIntegral(ds, mu, numStepsInt, minBound, maxBound, eps)
        //    , m_curvatureStepSize(0.0f)
        //    , m_lookupTable()
        //{
        //    std::cout << "Calcuating path weight integral lookup table" << std::endl;

        //    m_curvatureStepSize = (m_maxCurvature - m_minCurvature) / m_numCurvatureSteps;
        //    m_lookupTable.clear();
        //    m_lookupTable.resize(m_numCurvatureSteps + 1u);

        //    // Handle first case
        //    {
        //        const double value = m_regularizedIntegral.Integrate(scattering, m_minCurvature);
        //        // This is just like the normal cached version, except we cache the log version of the weights
        //        const double logValue = std::log(value);
        //        m_lookupTable[0] = logValue;
        //    }

        //    double min = m_lookupTable[0];
        //    double max = m_lookupTable[0];

        //    for (uint32_t i = 1; i <= m_numCurvatureSteps; ++i)
        //    {
        //        const double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
        //        const double value = m_regularizedIntegral.Integrate(scattering, curvatureEval);
        //        m_lookupTable[i] = value;

        //        if (value < min)
        //        {
        //            min = value;
        //        }
        //        if (value > max)
        //        {
        //            max = value;
        //        }
        //    }

        //    m_minSegmentWeight = min;
        //    m_maxSegmentWeight = max;

        //    std::cout << "Finished log path weight integral lookup table" << std::endl;
        //    std::cout << "\tMin Possible Log Weight Value: " << min << std::endl;
        //    std::cout << "\tMax Possible Log Weight Value: " << max << std::endl;
        //}

        //LogWeightLookupTableIntegral::~LogWeightLookupTableIntegral()
        //{
        //}

        //TwistyWeight LogWeightLookupTableIntegral::Integrate(double scattering, double curvature) const
        //{
        //    // Should not execute
        //    assert(false);
        //    assert(curvature >= m_minCurvature);
        //    assert(curvature <= m_maxCurvature);

        //    double distance = curvature - m_minCurvature;
        //    double realIdx = distance / m_curvatureStepSize;
        //    uint32_t leftIdx = floor(realIdx);
        //    uint32_t rightIdx = leftIdx + 1;

        //    double leftLookup = m_lookupTable[leftIdx];
        //    double rightLookup = m_lookupTable[rightIdx];

        //    double leftDist = distance - (leftIdx * m_curvatureStepSize);

        //    

        //    // TODO: Whats the point even of doing logs if this is necessary
        //    double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
        //    //std::cout << "Interpolated result: " << interpolatedResult << std::endl;
        //    double interpolatedResultLog = log(interpolatedResult);
        //    //std::cout << "Log of interpolated result: " << interpolatedResultLog << std::endl;

        //    // FIXME: Magic number atm.
        //    return interpolatedResultLog;
        //}
    }
}