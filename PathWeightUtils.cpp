#include "PathWeightUtils.h"

#include "MathConsts.h"

#include <algorithm>
#include <assert.h>
#include <filesystem>
#include <fstream>

namespace twisty
{
    namespace PathWeighting
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
            double a = std::sqrt(TwistyPi * mu * 0.5);
            double b = SimpleGaussianPhase(evalLocation, mu);
            double c = 1.0 - std::exp(-2.0 / mu);
            return (a * b) / c;
        }

        //// Base Integral Strategy
        //IntegralStrategy::IntegralStrategy(const WeightingParameters& weightingParams, double ds)
        //    : m_ds(ds)
        //    , m_weightingParams(weightingParams)
        //{
        //}

        //IntegralStrategy::~IntegralStrategy()
        //{
        //}

        // Lookup table integrand
        BaseWeightLookupTable::BaseWeightLookupTable(const WeightingParameters& weightingParams, double ds, double minCurvature, double maxCurvature)
            : m_minCurvature(minCurvature)
            , m_maxCurvature(maxCurvature)
            , m_curvatureStepSize(0.0)
            , m_lookupTable(weightingParams.numCurvatureSteps + 1) // Leave room for one past the last curvature, weird boundary stuff
            , m_weightingParams(weightingParams)
            , m_ds(ds)
        {
            m_curvatureStepSize = (m_maxCurvature - m_minCurvature) / (weightingParams.numCurvatureSteps - 1);
        }

        BaseWeightLookupTable::~BaseWeightLookupTable()
        {
        }

        double BaseWeightLookupTable::InterpolateWeightValues(double curvature) const
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

        void BaseWeightLookupTable::ExportValues(std::string directoryFileName)
        {
            const std::filesystem::path exprDirectory = directoryFileName;
            const std::string TableValuesFilename = ExportFilename();
            const std::filesystem::path outputFilePath = exprDirectory / TableValuesFilename;

            std::cout << "Exporting table of size: " << m_lookupTable.size() << std::endl;

            std::ofstream outputFile(outputFilePath.string());
            for (uint32_t i = 0; i < m_lookupTable.size(); ++i)
            {
                const double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
                outputFile << curvatureEval << ", " << m_lookupTable[i] << std::endl;
            }
            outputFile.close();
        }


        // Lookup table integrand
        WeightLookupTableIntegral::WeightLookupTableIntegral(const WeightingParameters& weightingParams, double ds)
            : BaseWeightLookupTable(weightingParams, ds, 0.0, 0.0)
        {
            std::cout << "Calcuating WeightLookupTableIntegral lookup table" << std::endl;
            twisty::PathWeighting::CalcMinMaxCurvature(m_minCurvature, m_maxCurvature, ds);

            m_curvatureStepSize = (m_maxCurvature - m_minCurvature) / (weightingParams.numCurvatureSteps - 1);

            // Handle first case
            {
                double value = Integrate(m_minCurvature, weightingParams, ds);
                m_lookupTable[0] = value;
            }

            double min = m_lookupTable[0];
            double max = m_lookupTable[0];
            for (uint32_t i = 1; i <= weightingParams.numCurvatureSteps; ++i)
            {
                double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
                double value = Integrate(curvatureEval, weightingParams, ds);
                m_lookupTable[i] = value;

                if (min < 0.0)
                {
                    if (value > 0.0)
                    {
                        min = value;
                    }
                    else
                    {
                        // Do nothing because we dont want a negative min in the end, thus we dont need to update to a min
                    }
                }
                else
                {
                    // We only want values greater than zero in this case, no negatives
                    if (value > 0.0)
                    {
                        if (value < min)
                        {
                            min = value;
                        }
                    }
                }

                if (value > max)
                {
                    max = value;
                }
            }

            uint32_t numInvalid = 0;
            for (uint32_t i = 0; i <= weightingParams.numCurvatureSteps; ++i)
            {
                double value = m_lookupTable[i];

                if (value <= 0.0)
                {
                    numInvalid++;
                    value = min;
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
            std::cout << "\t\tmu: " << weightingParams.mu << std::endl;
            std::cout << "\t\tnumStepsInt: " << weightingParams.numStepsInt << std::endl;
            std::cout << "\t\tm_minBound: " << weightingParams.minBound << std::endl;
            std::cout << "\t\tm_maxBound: " << weightingParams.maxBound << std::endl;
            std::cout << "\t\tm_eps: " << weightingParams.eps<< std::endl;
            std::cout << "\t\tm_minCurvature: " << m_minCurvature << std::endl;
            std::cout << "\t\tm_maxCurvature: " << m_maxCurvature << std::endl;
            std::cout << "\t\tm_numCurvatureSteps: " << weightingParams.numCurvatureSteps << std::endl;
        }

        WeightLookupTableIntegral::~WeightLookupTableIntegral()
        {
        }


        double WeightLookupTableIntegral::Integrate(double curvature, const WeightingParameters& weightingParams, double ds) const
        {
            double kds = curvature * ds;
            double bds = weightingParams.scatter * ds;

            auto Integrand = [this, weightingParams](double p, double kds, double bds) -> double
            {
                double phaseFunction = GaussianPhase(p, weightingParams.mu);

                double scatteringTerm = p * std::exp(
                    bds * phaseFunction // scatter piece
                    - 1.0 * (weightingParams.eps * weightingParams.eps * p * p) / 2.0 // regularizer
                );

                double sinTerm = 0.0;
                if (kds != 0.0)
                {
                    sinTerm = sin(kds * p) / kds;
                }
                else
                {
                    sinTerm = p;
                }

                return scatteringTerm * sinTerm;
            };

            // Perform first integration
            double firstVal = 0.0f;
            {
                double stepSize = (weightingParams.maxBound - weightingParams.minBound) / weightingParams.numStepsInt;
                for (uint32_t i = 0; i <= weightingParams.numStepsInt; ++i)
                {
                    double p = i * stepSize;
                    double left = Integrand(p, kds, bds);
                    firstVal += left * stepSize;
                }
            }

            // Note: This is added back in because it seems this allows the base kds == 0 case to work
            // Perform second integration
            double secondVal = 0.0f;
            {
                double stepSize = (weightingParams.maxBound - weightingParams.minBound) / weightingParams.numStepsInt;
                for (uint32_t i = 0; i <= weightingParams.numStepsInt; ++i)
                {
                    double p = i * stepSize;
                    double left = Integrand(p, 0.0, bds);
                    secondVal += left * stepSize;
                }
            }

            double c = weightingParams.absorbtion + weightingParams.scatter;
            double constant = std::exp(-c * ds) / (2.0 * TwistyPi * TwistyPi);
            return constant * (firstVal / secondVal);
        }

        // Lookup table integrand
        SimpleWeightLookupTable::SimpleWeightLookupTable(const twisty::WeightingParameters& weightingParams, double ds)
            : BaseWeightLookupTable(weightingParams, ds, -1.0, 1.0)
        {
            std::cout << "Calcuating path weight integral lookup table: " << weightingParams.numCurvatureSteps << std::endl;
            //twisty::PathWeighting::CalcMinMaxCurvature(m_minCurvature, m_maxCurvature, ds);
            m_minCurvature = -1.0;
            m_maxCurvature = 1.0;

            m_curvatureStepSize = (m_maxCurvature - m_minCurvature) / (weightingParams.numCurvatureSteps - 1);

            auto CalculateSimpleWeightValue = [weightingParams, ds](double curvature) -> double {
                const double alpha = 1.0 / (weightingParams.scatter * ds * weightingParams.mu);
                const double leftComponent = std::exp(-1.0 * alpha);
                const double rightComponent = std::exp(alpha * curvature);
                return leftComponent * rightComponent;
            };

            // Handle first case
            {
                double value = CalculateSimpleWeightValue(m_minCurvature);
                m_lookupTable[0] = value;
            }

            double min = m_lookupTable[0];
            double max = m_lookupTable[0];
            for (uint32_t i = 1; i <= weightingParams.numCurvatureSteps; ++i)
            {
                double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
                double value = CalculateSimpleWeightValue(curvatureEval);

                m_lookupTable[i] = value;
                if (min < 0.0)
                {
                    if (value > 0.0)
                    {
                        min = value;
                    }
                    else
                    {
                        // Do nothing because we dont want a negative min in the end, thus we dont need to update to a min
                    }
                }
                else
                {
                    // We only want values greater than zero in this case, no negatives
                    if (value > 0.0)
                    {
                        if (value < min)
                        {
                            min = value;
                        }
                    }
                }

                if (value > max)
                {
                    max = value;
                }
            }

            uint32_t numInvalid = 0;
            for (uint32_t i = 0; i <= weightingParams.numCurvatureSteps; ++i)
            {
                double value = m_lookupTable[i];

                if (value <= 0.0)
                {
                    numInvalid++;
                    value = min;
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
            std::cout << "\t\tmu: " << weightingParams.mu << std::endl;
            std::cout << "\t\tnumStepsInt: " << weightingParams.numStepsInt << std::endl;
            std::cout << "\t\tm_minBound: " << weightingParams.minBound << std::endl;
            std::cout << "\t\tm_maxBound: " << weightingParams.maxBound << std::endl;
            std::cout << "\t\tm_eps: " << weightingParams.eps << std::endl;
            std::cout << "\t\tm_minCurvature: " << m_minCurvature << std::endl;
            std::cout << "\t\tm_maxCurvature: " << m_maxCurvature << std::endl;
            std::cout << "\t\tm_numCurvatureSteps: " << weightingParams.numCurvatureSteps << std::endl;
        }

        SimpleWeightLookupTable::~SimpleWeightLookupTable()
        {
        }

        // Todo: figure out why max is this, should have documented
        void CalcMinMaxCurvature(double& minCurvature, double& maxCurvature, double ds)
        {
            minCurvature = 0.0;
            maxCurvature = (3.47 / ds) * 2.0;
        }

        // Assume we have good pointers
        double WeightCurveViaCurvatureLog10(float* pCurvatureStart, uint32_t numCurvatures, const twisty::PathWeighting::BaseWeightLookupTable& weightIntegral)
        {
            if (!pCurvatureStart || (numCurvatures == 0))
            {
                return 0.0;
            }

            double ds = weightIntegral.GetDs();
            const auto& weightingParams = weightIntegral.GetWeightingParams();
            double minCurvature = 0.0;
            double maxCurvature = 0.0;
            twisty::PathWeighting::CalcMinMaxCurvature(minCurvature, maxCurvature, ds);
            const double curvatureStepSize = (maxCurvature - minCurvature) / weightingParams.numCurvatureSteps;
            auto& lookupTable = weightIntegral.AccessLookupTable();

            const float c = weightingParams.scatter + weightingParams.absorbtion;
            const float absorbtionConst = std::exp(-c * ds) / (2.0 * TwistyPi * TwistyPi);
            const float absorbtionConstLog10 = std::log10(absorbtionConst);

            // Calculate value
            double runningPathWeightLog10 = 0.0;
            for (int64_t segIdx = 0; segIdx < numCurvatures; ++segIdx)
            {
                // Extract curvature
                double curvature = pCurvatureStart[segIdx];
                double distance = curvature - minCurvature;
                double realIdx = distance / curvatureStepSize;
                int64_t leftIdx = floor(realIdx);
                int64_t rightIdx = leftIdx + 1;

                double leftLookup = lookupTable[leftIdx];
                double rightLookup = lookupTable[rightIdx];

                double leftDist = distance - (leftIdx * curvatureStepSize);

                double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                // Take the natural log of the interpolated results
                double interpolatedResultLog10 = std::log10(interpolatedResult);
                // Lets do weights as doubles for now
                double segmentWeightLog10 = interpolatedResultLog10;

                // Take natural log of this constant
                segmentWeightLog10 += absorbtionConstLog10;

                // Update the running path weight. We also want to cache the segment weights
                runningPathWeightLog10 += segmentWeightLog10;
            }
            return runningPathWeightLog10;
        }


        // We have to calculate the "curvature" a different way for the simple weight case.
        // The actual value used is the dot product between two neighboring tangents
        double SimpleWeightCurveViaTangentDotProductLog10(Farlor::Vector3* pTangentsStart, uint32_t numSegments, const BaseWeightLookupTable& weightIntegral)
        {
            if (!pTangentsStart || (numSegments == 0))
            {
                return 0.0;
            }

            uint32_t numScatterEvents = numSegments - 1;

            double ds = weightIntegral.GetDs();
            const auto& weightingParams = weightIntegral.GetWeightingParams();
            double minCurvature = weightIntegral.GetMinCurvature();
            double maxCurvature = weightIntegral.GetMaxCurvature();
            const double curvatureStepSize = (maxCurvature - minCurvature) / weightingParams.numCurvatureSteps;
            auto& lookupTable = weightIntegral.AccessLookupTable();

            const float c = weightingParams.scatter + weightingParams.absorbtion;
            const float absorbtionConst = std::exp(-c * ds) / (2.0 * TwistyPi * TwistyPi);
            const float absorbtionConstLog10 = std::log10(absorbtionConst);

            // Calculate value
            double runningPathWeightLog10 = 0.0;
            for (int64_t segIdx = 0; segIdx < numScatterEvents; ++segIdx)
            {
                // Extract curvature
                Farlor::Vector3 leftTangent = pTangentsStart[segIdx].Normalized();
                Farlor::Vector3 rightTangent = pTangentsStart[segIdx + 1].Normalized();

                float curvature = leftTangent.Dot(rightTangent);
                curvature = std::min(curvature, 1.0f);
                curvature = std::max(curvature, -1.0f);

                double distance = curvature - minCurvature;
                double realIdx = distance / curvatureStepSize;
                int64_t leftIdx = floor(realIdx);
                int64_t rightIdx = leftIdx + 1;
                if (leftIdx == lookupTable.size() - 1)
                {
                    rightIdx--; // Bump it left 1, it doesnt really matter anymore anyways.
                }

                double leftLookup = lookupTable[leftIdx];

                double rightLookup = lookupTable[rightIdx];

                double leftDist = distance - (leftIdx * curvatureStepSize);

                double interpolatedResult = leftLookup * (1.0f - leftDist) + (rightLookup * leftDist);
                // Take the natural log of the interpolated results
                double interpolatedResultLog10 = std::log10(interpolatedResult);
                // Lets do weights as doubles for now
                double segmentWeightLog10 = interpolatedResultLog10;

                // Take natural log of this constant
                segmentWeightLog10 += absorbtionConstLog10;

                // Update the running path weight. We also want to cache the segment weights
                runningPathWeightLog10 += segmentWeightLog10;
            }
            return runningPathWeightLog10;
        }



        namespace NormalizerStuff
        {
            // Why is this?
            NormalizerDoubleType f2_formula(const double z)
            {
                double result = (z < 2.0) ? 1.0 : 0.0;
                return result;
            }

            void FN::WriteToFile(std::ofstream& outFile)
            {
                outFile << nb_samples << std::endl;
                outFile << nb_orders << std::endl;
                outFile << rmin << std::endl;
                outFile << rmax << std::endl;

                for (uint32_t order = 2; order <= nb_orders; ++order)
                {
                    std::cout << "Writing out order: " << order << std::endl;
                    auto& samples = fNsets[order];
                    outFile << order << std::endl;
                    for (uint32_t sample = 0; sample < nb_samples; ++sample)
                    {
                        outFile << samples[sample] << std::endl;
                    }
                }
            }

            void FN::init_fromFile(std::ifstream& inFile)
            {
                inFile >> nb_samples;
                inFile >> nb_orders;
                inFile >> rmin;
                inFile >> rmax;

                uint32_t numValues = 0;
                for (uint32_t order = 2; order <= nb_orders; ++order)
                {
                    // Make sure we are reading the correct order
                    uint32_t readOrder = 0;
                    inFile >> readOrder;
                    assert(order == readOrder);

                    // Read in all the samples
                    std::vector<NormalizerDoubleType> fSamples(nb_samples);
                    for (uint32_t sample = 0; sample < nb_samples; ++sample)
                    {
                        inFile >> fSamples[sample];
                        numValues++;
                    }
                    fNsets[order] = fSamples;
                }

                //assert(numValues == (nb_samples * nb_orders));
            }

            void FN::init(int numIntegrationSamples)
            {
                // order 2

                std::cout << "# order=" << 2 << std::endl;
                std::vector<NormalizerDoubleType> f2(nb_samples);

                // Assume we have a range of possible r values, we want to go ahead and calculate and store them
                double dr = (rmax - rmin) / (nb_samples - 1);
                // For each r-value sample, calculate the base order pair (M = 2, r)
                for (int i = 0; i < nb_samples; i++)
                {
                    double r = rmin + i * dr;
                    f2[i] = f2_formula(r);
                }
                fNsets[2] = f2;

                // Now we continue on and calculate the next orders up until maxorder
                // This is currently 200 in our case
                for (int o = 3; o <= nb_orders; o++)
                {
                    if ((o % 1) == 0)
                    {
                        std::cout << "# order=" << o << std::endl;
                    }

                    std::vector<NormalizerDoubleType> f(nb_samples);
#pragma omp parallel for
                    for (int i = 0; i < nb_samples; i++)
                    {
                        double r = rmin + i * dr;
                        double lower = std::fabs(r - 1.0);
                        double upper = r + 1.0;

                        // we use the same number of samples as our integral number of samples
                        double ddr = (upper - lower) / (numIntegrationSamples - 1);
                        NormalizerDoubleType accum = 0.0;
                        for (int i = 0; i < numIntegrationSamples; i++)
                        {
                            // Evaluation point is: rp
                            // Current order is: o
                            // Order below is: o - 1
                            double rp = lower + i * ddr;
                            // Add in bar representing ddr * r *f_{M-1}(r)
                            accum += ddr * rp * eval(o - 1, rp);
                        }
                        f[i] = accum;
                    }
                    fNsets[o] = f;
                }
            }

            // Evaluate f of a given order at an r position
            NormalizerDoubleType FN::eval(int order, double r) const
            {
                // Get the stored dataset for the specific order
                const std::vector<NormalizerDoubleType>& data = fNsets.at(order);
                // Next, find the bucket tp the left
                long double rrr = (r - rmin) * (nb_samples - 1) / (rmax - rmin);
                int ii = rrr;
                long double weight = rrr - ii;
                // Bind weights to the known table
                if (ii < 0)
                {
                    //std::cout << "Warning: Left clamped up to 0: " << r << std::endl;
                    ii = 0;
                }
                if (ii >= nb_samples)
                {
                    //std::cout << "Warning: Left clamped down to " << (nb_samples - 1) << ": " << r << std::endl;
                    ii = nb_samples - 1;
                }

                // Calculate right side value and bind to table as well
                int iii = ii + 1;
                if (iii < 0)
                {
                    //std::cout << "Warning: Right clamped up to 0: " << r << std::endl;
                    iii = 0;
                }
                if (iii >= nb_samples)
                {
                    //std::cout << "Warning: Right clamped down to " << (nb_samples - 1) << ": " << r << std::endl;
                    iii = nb_samples - 1;
                }

                // Finally, our datapoint is a bilinear interpolation
                return data[ii] * (1.0 - weight) + data[iii] * weight;
            }

            NormalizerDoubleType psd_one(const BaseNormalizer& fn, int M, const double s, const Farlor::Vector3& X, const Farlor::Vector3& N, const Farlor::Vector3& Np,
                const Farlor::Vector3& beta)
            {
                Farlor::Vector3 Z = X * (M + 1) * (1.0 / s) - N - Np;
                double z = Z.Magnitude();
                double zb = (Z - beta).Magnitude();
                NormalizerDoubleType fmb = fn.eval(M - 1, zb);
                NormalizerDoubleType fm = fn.eval(M, z);
                if (fm == 0.0)
                {
                    return 0.0;
                }
                NormalizerDoubleType result = fmb / fm;
                result *= (z / zb);
                result /= (2.0 * 3.14159265);
                result *= std::pow((double)(M) / (double)(M + 1), 3);
                return result;
            }

            NormalizerDoubleType Norm(const BaseNormalizer& fn, int M, double z, double s)
            {
                std::cout << "M: " << M << std::endl;
                std::cout << "z: " << z << std::endl;
                std::cout << "s: " << s << std::endl;

                NormalizerDoubleType fneval = fn.eval(M, z);
                NormalizerDoubleType piPower = boost::multiprecision::pow(
                    boost::multiprecision::cpp_dec_float_100(2.0) * boost::multiprecision::cpp_dec_float_100(3.14150265),
                    M - 1
                );
                NormalizerDoubleType invZ = 1.0 / z;
                NormalizerDoubleType mPower = boost::multiprecision::pow(
                    boost::multiprecision::cpp_dec_float_100(M + 1) / s,
                    3
                );

                std::cout << "fneval: " << fneval << std::endl;
                std::cout << "piPower: " << piPower << std::endl;
                std::cout << "invZ: " << invZ << std::endl;
                std::cout << "mPower: " << mPower << std::endl;

                NormalizerDoubleType result = fneval * piPower * invZ * mPower;
                std::cout << "result: " << result << std::endl;

                return result;
            }

            NormalizerDoubleType psd_n(const BaseNormalizer& fn, int M, const double s, const Farlor::Vector3& X, const Farlor::Vector3& N,
                const Farlor::Vector3& Np, const std::vector<Farlor::Vector3>& beta)
            {
                Farlor::Vector3 Zm = X * (M + 1) * (1 / s) - N - Np;
                double zm = Zm.Magnitude();
                // Subtract off all the betas
                for (size_t n = 0; n < beta.size(); n++)
                {
                    Zm -= beta[n];
                }
                double rmn = Zm.Magnitude();

                NormalizerDoubleType result = (fn.eval(M - (int)beta.size(), rmn) * zm) / (fn.eval(M, zm) * rmn);
                result /= boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(2.0 * 3.14159265), beta.size());
                result *= std::pow((double)(M - beta.size() + 1) / (double)(M + 1), 3);
                return result;
            }

            NormalizerDoubleType likelihood(const BaseNormalizer& fn, int M, const double arclength, const Farlor::Vector3& endPosition, const Farlor::Vector3& startPosition,
                const Farlor::Vector3& N, const Farlor::Vector3& Np, const std::vector<Farlor::Vector3>& oldBetas, const std::vector < Farlor::Vector3> & newBetas)
            {
                Farlor::Vector3 Z0 = (endPosition - startPosition) * (M + 1) * (1.0 / arclength);
                //std::cout << "Z0: " << Z0 << std::endl;

                Farlor::Vector3 Zstar = Z0;
                for (size_t n = 0; n < oldBetas.size(); n++)
                {
                    Z0 -= oldBetas[n];
                    Zstar -= newBetas[n];
                }

                //std::cout << "Zstar: " << Zstar << std::endl;

                double z0 = Z0.Magnitude();
                double zstar = Zstar.Magnitude();

                //std::cout << "z0: " << z0 << std::endl;
                //std::cout << "zstar: " << zstar << std::endl;

                NormalizerDoubleType topFN = fn.eval(M - (int)newBetas.size(), zstar);
                NormalizerDoubleType bottomFN = fn.eval(M - (int)oldBetas.size(), z0);
                double zRational = z0 / zstar;

                //std::cout << "topFN: " << topFN << std::endl;
                //std::cout << "bottomFN: " << bottomFN << std::endl;
                //std::cout << "zRational: " << zRational << std::endl;

                if ((bottomFN == 0.0) || (zstar == 0.0))
                {
                    return 0.0;
                }
                NormalizerDoubleType result = (topFN / bottomFN) * zRational;
                return result;
            }

            Farlor::Vector3 makeVector(double a, double b, double c)
            {
                return Farlor::Vector3(a, b, c);
            }

            NormalizerDoubleType CalculateLikelihood(const BaseNormalizer& fn, int numSegments, const twisty::PerturbUtils::BoundrayConditions& boundaryConditions,
                std::vector<Farlor::Vector3>& oldBetas, std::vector<Farlor::Vector3>& newBetas)
            {
                // We store 1 tangent per segement plus an addition one for end.
                // The first and last are boundary, so we want the middle M-1
                assert(oldBetas.size() == newBetas.size());

                return likelihood(fn, numSegments - 1, boundaryConditions.arclength, boundaryConditions.m_endPos, boundaryConditions.m_startPos,
                    boundaryConditions.m_endDir, boundaryConditions.m_startDir, oldBetas, newBetas);
            }

            std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> GetNormalizer(uint32_t numSegments)
            {
                const std::string fnFilename = "SavedFN.fnd";
                const std::filesystem::path fnFilePath = std::filesystem::current_path() / fnFilename;

                if (std::filesystem::exists(fnFilePath))
                {
                    std::cout << "Using cached fd file at: " << fnFilePath << std::endl;
                    std::ifstream inFile(fnFilePath);
                    std::unique_ptr<PathWeighting::NormalizerStuff::FN> upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(inFile);
                    inFile.close();
                    return upFN;
                }

                // This is the max M value
                const int maxorder = numSegments;

                // Generate the fn data table
                const int numZSamples = 5000;
                const int numIntegrationSamples = 5000;

                // Arbitrarily set min and max |r_vec| values.
                // Why this specific max bound, I do not know
                const double rMin = 0.0;
                const double rMax = 200.0;
                std::unique_ptr<PathWeighting::NormalizerStuff::FN> upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(numZSamples, numIntegrationSamples, maxorder, rMin, rMax);

                std::ofstream outFile(fnFilePath);
                upFN->WriteToFile(outFile);
                outFile.close();
                return upFN;
            }
        }
    }
}