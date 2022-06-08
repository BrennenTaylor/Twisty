#include "PathWeightUtils.h"

#include "MathConsts.h"

#include "CurvePerturbUtils.h"

#include <algorithm>
#include <assert.h>
#include <filesystem>
#include <fstream>

namespace twisty {
namespace PathWeighting {
    // Parameterized simple gaussian function
    double SimpleGaussianPhase(double p, double mu)
    {
        double val = -mu * p * p * 0.5;
        val *= 0.5;
        return std::exp(val);
    }

    // Parameterized gaussian function
    double GaussianPhase(double evalLocation, double mu)
    {
        double Np = std::sqrt(TwistyPi * mu * 0.5) / (1.0 - std::exp(-2.0 / mu));
        return Np * SimpleGaussianPhase(evalLocation, mu);
    }

    // Lookup table integrand
    BaseWeightLookupTable::BaseWeightLookupTable(const WeightingParameters &weightingParams,
          double ds, double minCurvature, double maxCurvature)
        : m_minCurvature(minCurvature)
        , m_maxCurvature(maxCurvature)
        , m_curvatureStepSize((maxCurvature - minCurvature) / weightingParams.numCurvatureSteps)
        , m_lookupTable(
                weightingParams.numCurvatureSteps + 1)  // We include 0, thus need that last spot
        , m_ds(ds)
        , m_weightingParams(weightingParams)
    {
    }

    BaseWeightLookupTable::~BaseWeightLookupTable() { }

    double BaseWeightLookupTable::InterpolateWeightValues(double curvature) const
    {
        if (curvature < m_minCurvature) {
            std::cout << "Clamping to min curvature" << std::endl;
            curvature = m_minCurvature;
        }

        if (curvature <= m_minCurvature) {
            return m_lookupTable[0];
        }

        if (curvature > m_maxCurvature) {
            std::cout << "Clamping to max curvature" << std::endl;
            curvature = m_maxCurvature;
        }

        if (curvature >= m_maxCurvature) {
            return m_lookupTable[m_weightingParams.numCurvatureSteps];
        }

        double distanceFromMin = curvature - m_minCurvature;
        double realIdx = distanceFromMin / m_curvatureStepSize;
        int32_t leftIdx = floor(realIdx);
        int32_t rightIdx = leftIdx + 1;

        double leftLookup = m_lookupTable[leftIdx];
        double rightLookup = m_lookupTable[rightIdx];

        double interpDist = distanceFromMin - (leftIdx * m_curvatureStepSize);
        double interpolatedResult = leftLookup * (1.0f - interpDist) + (rightLookup * interpDist);
        return interpolatedResult;
    }

    void BaseWeightLookupTable::ExportValues(const std::string &directoryFileName) const
    {
        std::filesystem::path exportDirectory = directoryFileName;
        exportDirectory = exportDirectory / "WeightLookupTable/";
        if (!std::filesystem::exists(exportDirectory)) {
            std::filesystem::create_directory(exportDirectory);
        }

        std::cout << "Exporting table of size: " << m_lookupTable.size() << std::endl;
        std::ofstream outputFile(exportDirectory.string() + ExportFilename());
        if (!outputFile.is_open()) {
            throw std::runtime_error("Failed to export weight table");
        }
        for (uint32_t i = 0; i < m_lookupTable.size(); ++i) {
            const double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
            outputFile << curvatureEval << ", " << m_lookupTable[i] << std::endl;
        }
        outputFile.close();
    }

    // Lookup table integrand
    WeightLookupTableIntegral::WeightLookupTableIntegral(
          const WeightingParameters &weightingParams, double ds)
        : BaseWeightLookupTable(weightingParams, ds, 0.0, 0.0)
    {
        std::cout << "Calcuating WeightLookupTableIntegral lookup table" << std::endl;
        MinMaxCurvature minMax = twisty::PathWeighting::CalcMinMaxCurvature(weightingParams, ds);
        m_minCurvature = minMax.minCurvature;
        m_maxCurvature = minMax.maxCurvature;

        m_curvatureStepSize
              = (m_maxCurvature - m_minCurvature) / (weightingParams.numCurvatureSteps - 1);

        // Handle first case
        {
            double value = Integrate(m_minCurvature, weightingParams, ds);
            m_lookupTable[0] = value;
        }
        double min = m_lookupTable[0];
        double max = m_lookupTable[0];

        for (uint32_t i = 1; i <= weightingParams.numCurvatureSteps; ++i) {
            double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
            double value = Integrate(curvatureEval, weightingParams, ds);

            // Running min?
            if (value <= 0.0) {
                value = min;
            }
            m_lookupTable[i] = value;

            if (value < min) {
                min = value;
            }

            if (value > max) {
                max = value;
            }
        }

        uint32_t numInvalid = 0;
        for (uint32_t i = 0; i <= weightingParams.numCurvatureSteps; ++i) {
            double value = m_lookupTable[i];

            if (value <= 0.0) {
                numInvalid++;
                value = min;
            }
        }

        if (numInvalid > 0) {
            std::cout << "We calculated " << numInvalid
                      << " negative table values and clamped them to zero" << std::endl;
        }

        m_minSegmentWeight = min;
        m_maxSegmentWeight = max;

        std::cout << "Finished path weight integral lookup table" << std::endl;
        std::cout << "\tMin Possible Weight Value: " << min << std::endl;
        std::cout << "\tMax Possible Weight Value: " << max << std::endl;
        // Parameters
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

    WeightLookupTableIntegral::~WeightLookupTableIntegral() { }

    double WeightLookupTableIntegral::Integrate(
          double curvature, const WeightingParameters &weightingParams, double ds) const
    {
        double kds = curvature * ds;
        double bds = weightingParams.scatter * ds;

        auto Integrand = [this, weightingParams](double p, double kds, double bds) -> double {
            double phaseFunction = GaussianPhase(p, weightingParams.mu);

            double scatteringTerm = p
                  * std::exp(bds * phaseFunction  // scatter piece
                        - 1.0
                              * (weightingParams.eps * weightingParams.eps * p * p
                                    * 0.5)  // regularizer
                  );

            double sinTerm = 0.0;
            // TODO: Should we implement this as if (kds < smallAngleThreshold?)
            if (kds != 0.0) {
                sinTerm = sin(kds * p) / kds;
            }
            // With small angle apprimation, we have that kds is very small
            // As we approch, we have that sin(kds * p) => p
            // as well as                  1/kds -> 1/inf
            else {
                sinTerm = p;
            }

            return scatteringTerm * sinTerm;
        };

        // Perform integration. Per disertation page 34, when bn is constant, wn peaks
        // at kds. As a result, normalizeation is used. This is due to an overflow.
        // However, as we already are using a big float library, do we even need to
        // deal with this?
        double firstVal = 0.0f;
        double normalizerWithZeroCurvature = 0.0f;
        {
            double stepSize = (weightingParams.maxBound - weightingParams.minBound)
                  / (weightingParams.numStepsInt - 1);
            for (uint32_t i = 0; i <= weightingParams.numStepsInt; ++i) {
                double p = i * stepSize;
                firstVal += Integrand(p, kds, bds) * stepSize;
                normalizerWithZeroCurvature += Integrand(p, 0.0, bds) * stepSize;
            }
        }

        // We put the exponential falloff due to C here.
        double c = weightingParams.absorbtion + weightingParams.scatter;
        double constant = std::exp(-c * ds) / (2.0 * TwistyPi * TwistyPi);
        // TODO: For now, lets try not including this
        return constant * firstVal / normalizerWithZeroCurvature;
    }

    // Lookup table integrand
    SimpleWeightLookupTable::SimpleWeightLookupTable(
          const twisty::WeightingParameters &weightingParams, double ds)
        : BaseWeightLookupTable(weightingParams, ds, -1.0, 1.0)
    {
        std::cout << "Calcuating path weight integral lookup table: "
                  << weightingParams.numCurvatureSteps << std::endl;
        MinMaxCurvature minMax = twisty::PathWeighting::CalcMinMaxCurvature(weightingParams, ds);
        m_minCurvature = minMax.minCurvature;
        m_maxCurvature = minMax.maxCurvature;

        m_curvatureStepSize
              = (m_maxCurvature - m_minCurvature) / (weightingParams.numCurvatureSteps - 1);

        auto CalculateSimpleWeightValue = [weightingParams, ds](double curvature) -> double {
            curvature *= -1.0;  // All curvatures in this mode are negated
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
        for (uint32_t i = 1; i <= weightingParams.numCurvatureSteps; ++i) {
            double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
            double value = CalculateSimpleWeightValue(curvatureEval);
            // Running min?
            if (value <= 0.0) {
                value = min;
            }
            m_lookupTable[i] = value;

            if (value < min) {
                min = value;
            }

            if (value > max) {
                max = value;
            }
        }

        uint32_t numInvalid = 0;
        for (uint32_t i = 0; i <= weightingParams.numCurvatureSteps; ++i) {
            double value = m_lookupTable[i];

            if (value <= 0.0) {
                numInvalid++;
                value = min;
            }
        }

        if (numInvalid > 0) {
            std::cout << "We calculated " << numInvalid
                      << " negative table values and clamped them to zero" << std::endl;
        }

        m_minSegmentWeight = min;
        m_maxSegmentWeight = max;

        std::cout << "Finished path weight integral lookup table" << std::endl;
        std::cout << "\tMin Possible Weight Value: " << min << std::endl;
        std::cout << "\tMax Possible Weight Value: " << max << std::endl;
        // Parameters
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

    SimpleWeightLookupTable::~SimpleWeightLookupTable() { }

    MinMaxCurvature CalcMinMaxCurvature(const twisty::WeightingParameters &wp, double ds)
    {
        switch (wp.weightingMethod) {
            case WeightingMethod::RadiativeTransfer: {
                // Our curvature lookups are using finite difference
                // ki = (ti+1 - ti-1) / (2.0 * ds)
                // Our max and min are calculated as follows

                MinMaxCurvature result;
                result.minCurvature = 0.0;
                result.maxCurvature = 2.3 / (ds);
                return result;
            } break;
            case WeightingMethod::SimplifiedModel: {
                // This is a dot prodcuct version
                // Curvatures are calculated in the following:
                // tr.dot(tl) = [-1, 1]
                MinMaxCurvature result;
                result.minCurvature = -1.0;
                result.maxCurvature = 1.0;
                return result;
            } break;
            default: {
                std::cout << "Error, invalid weighting model selected" << std::endl;
                MinMaxCurvature result;
                return result;
            } break;
        }
    }

    // // We have to calculate the "curvature" a different way for the simple weight
    // case.
    // // The actual value used is the dot product between two neighboring tangents
    // double SimpleWeightCurveViaTangentDotProductLog10(Farlor::Vector3*
    // pTangentsStart, uint32_t numSegments, const BaseWeightLookupTable&
    // weightIntegral)
    // {
    //     if (!pTangentsStart || (numSegments == 0))
    //     {
    //         return 0.0;
    //     }

    //     uint32_t numScatterEvents = numSegments - 1;

    //     double ds = weightIntegral.GetDs();
    //     const auto& weightingParams = weightIntegral.GetWeightingParams();
    //     MinMaxCurvature minMax =
    //     twisty::PathWeighting::CalcMinMaxCurvature(weightingParams, ds); const
    //     double curvatureStepSize = (minMax.maxCurvature - minMax.minCurvature) /
    //     weightingParams.numCurvatureSteps; auto& lookupTable =
    //     weightIntegral.AccessLookupTable();

    //     // Calculate value
    //     double runningPathWeightLog10 = 0.0;
    //     for (int64_t segIdx = 0; segIdx < numScatterEvents; ++segIdx)
    //     {
    //         // Extract curvature
    //         Farlor::Vector3 leftTangent = pTangentsStart[segIdx].Normalized();
    //         Farlor::Vector3 rightTangent = pTangentsStart[segIdx +
    //         1].Normalized();

    //         float curvature = leftTangent.Dot(rightTangent);
    //         curvature = std::min(curvature, 1.0f);
    //         curvature = std::max(curvature, -1.0f);

    //         double distance = curvature - minMax.minCurvature;
    //         double realIdx = distance / curvatureStepSize;
    //         int64_t leftIdx = floor(realIdx);
    //         int64_t rightIdx = leftIdx + 1;
    //         if (leftIdx == lookupTable.size() - 1)
    //         {
    //             rightIdx--; // Bump it left 1, it doesnt really matter anymore
    //             anyways.
    //         }

    //         double leftLookup = lookupTable[leftIdx];

    //         double rightLookup = lookupTable[rightIdx];

    //         double leftDist = distance - (leftIdx * curvatureStepSize);

    //         double interpolatedResult = leftLookup * (1.0f - leftDist) +
    //         (rightLookup * leftDist);
    //         // Take the log10 of the interpolated results
    //         double interpolatedResultLog10 = std::log10(interpolatedResult);
    //         // Lets do weights as doubles for now
    //         double segmentWeightLog10 = interpolatedResultLog10;

    //         // Update the running path weight. We also want to cache the segment
    //         weights runningPathWeightLog10 += segmentWeightLog10;
    //     }
    //     return runningPathWeightLog10;
    // }

    namespace NormalizerStuff {
        // Why is this?
        NormalizerDoubleType f2_formula(const double z)
        {
            double result = (z < 2.0) ? 1.0 : 0.0;
            return result;
        }

        void FN::WriteToFile(std::ofstream &outFile)
        {
            outFile << nb_samples << std::endl;
            outFile << nb_orders << std::endl;
            outFile << rmin << std::endl;
            outFile << rmax << std::endl;

            for (uint32_t order = 2; order <= nb_orders; ++order) {
                std::cout << "Writing out order: " << order << std::endl;
                auto &samples = fNsets[order];
                outFile << order << std::endl;
                for (uint32_t sample = 0; sample < nb_samples; ++sample) {
                    outFile << samples[sample] << std::endl;
                }
            }
        }

        void FN::init_fromFile(std::ifstream &inFile)
        {
            inFile >> nb_samples;
            inFile >> nb_orders;
            inFile >> rmin;
            inFile >> rmax;

            uint32_t numValues = 0;
            for (uint32_t order = 2; order <= nb_orders; ++order) {
                // Make sure we are reading the correct order
                uint32_t readOrder = 0;
                inFile >> readOrder;
                assert(order == readOrder);

                // Read in all the samples
                std::vector<NormalizerDoubleType> fSamples(nb_samples);
                for (uint32_t sample = 0; sample < nb_samples; ++sample) {
                    inFile >> fSamples[sample];
                    numValues++;
                }
                fNsets[order] = fSamples;
            }

            // assert(numValues == (nb_samples * nb_orders));
        }

        void FN::init(int numIntegrationSamples)
        {
            // order 2

            std::cout << "# order=" << 2 << std::endl;
            std::vector<NormalizerDoubleType> f2(nb_samples);

            // Assume we have a range of possible r values, we want to go ahead and
            // calculate and store them
            double dr = (rmax - rmin) / (nb_samples - 1);
            // For each r-value sample, calculate the base order pair (M = 2, r)
            for (int i = 0; i < nb_samples; i++) {
                double r = rmin + i * dr;
                f2[i] = f2_formula(r);
            }
            fNsets[2] = f2;

            // Now we continue on and calculate the next orders up until maxorder
            // This is currently 200 in our case
            for (int o = 3; o <= nb_orders; o++) {
                if ((o % 1) == 0) {
                    std::cout << "# order=" << o << std::endl;
                }

                std::vector<NormalizerDoubleType> f(nb_samples);
#pragma omp parallel for
                for (int i = 0; i < nb_samples; i++) {
                    double r = rmin + i * dr;
                    double lower = std::fabs(r - 1.0);
                    double upper = r + 1.0;

                    // we use the same number of samples as our integral number of samples
                    double ddr = (upper - lower) / (numIntegrationSamples - 1);
                    NormalizerDoubleType accum = 0.0;
                    for (int i = 0; i < numIntegrationSamples; i++) {
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
            const std::vector<NormalizerDoubleType> &data = fNsets.at(order);
            const long double dr = (rmax - rmin) / (nb_samples - 1);
            // Next, find the bucket tp the left
            int leftIdx = (int)(r - rmin) / dr;
            // Bind weights to the known table
            leftIdx = std::min(leftIdx, nb_samples - 1);
            leftIdx = std::max(leftIdx, 0);

            // Calculate right side value and bind to table as well
            int rightIdx = leftIdx + 1;
            rightIdx = std::min(rightIdx, nb_samples - 1);
            rightIdx = std::max(rightIdx, 0);

            double distance = r - rmin;
            double leftDist = distance - (leftIdx * dr);

            // Finally, our datapoint is a bilinear interpolation
            return data[leftIdx] * (1.0f - leftDist) + data[rightIdx] * leftDist;
        }

        NormalizerDoubleType psd_one(const BaseNormalizer &fn, int M, const double s,
              const Farlor::Vector3 &X, const Farlor::Vector3 &N, const Farlor::Vector3 &Np,
              const Farlor::Vector3 &beta)
        {
            Farlor::Vector3 Z = X * (M + 1) * (1.0 / s) - N - Np;
            double z = Z.Magnitude();
            double zb = (Z - beta).Magnitude();
            NormalizerDoubleType fmb = fn.eval(M - 1, zb);
            NormalizerDoubleType fm = fn.eval(M, z);
            NormalizerDoubleType zero(0.0);
            if (fm == zero) {
                return zero;
            }
            NormalizerDoubleType result = fmb / fm;
            result *= (z / zb);
            result /= (2.0 * 3.14159265);
            result *= std::pow((double)(M) / (double)(M + 1), 3);
            return result;
        }

        NormalizerDoubleType Norm(const BaseNormalizer &fn, int M, double z, double s)
        {
            std::cout << "M: " << M << std::endl;
            std::cout << "z: " << z << std::endl;
            std::cout << "s: " << s << std::endl;

            NormalizerDoubleType fneval = fn.eval(M, z);
            NormalizerDoubleType piPower
                  = boost::multiprecision::pow(boost::multiprecision::cpp_dec_float_100(2.0)
                              * boost::multiprecision::cpp_dec_float_100(3.14150265),
                        M - 1);
            NormalizerDoubleType invZ = 1.0 / z;
            NormalizerDoubleType mPower = boost::multiprecision::pow(
                  boost::multiprecision::cpp_dec_float_100(M + 1) / s, 3);

            std::cout << "fneval: " << fneval << std::endl;
            std::cout << "piPower: " << piPower << std::endl;
            std::cout << "invZ: " << invZ << std::endl;
            std::cout << "mPower: " << mPower << std::endl;

            NormalizerDoubleType result = fneval * piPower * invZ * mPower;
            std::cout << "result: " << result << std::endl;

            return result;
        }

        NormalizerDoubleType psd_n(const BaseNormalizer &fn, int M, const double s,
              const Farlor::Vector3 &X, const Farlor::Vector3 &N, const Farlor::Vector3 &Np,
              const std::vector<Farlor::Vector3> &beta)
        {
            Farlor::Vector3 Zm = X * (M + 1) * (1 / s) - N - Np;
            double zm = Zm.Magnitude();
            // Subtract off all the betas
            for (size_t n = 0; n < beta.size(); n++) {
                Zm -= beta[n];
            }
            double rmn = Zm.Magnitude();

            NormalizerDoubleType result
                  = (fn.eval(M - (int)beta.size(), rmn) * zm) / (fn.eval(M, zm) * rmn);
            result /= boost::multiprecision::pow(
                  boost::multiprecision::cpp_dec_float_100(2.0 * 3.14159265), beta.size());
            result *= std::pow((double)(M - beta.size() + 1) / (double)(M + 1), 3);
            return result;
        }

        NormalizerDoubleType likelihood(const BaseNormalizer &fn, int M, const double arclength,
              const Farlor::Vector3 &endPosition, const Farlor::Vector3 &startPosition,
              const Farlor::Vector3 &N, const Farlor::Vector3 &Np,
              const std::vector<Farlor::Vector3> &oldBetas,
              const std::vector<Farlor::Vector3> &newBetas)
        {
            Farlor::Vector3 Z0 = (endPosition - startPosition) * (M + 1) * (1.0 / arclength);
            // std::cout << "Z0: " << Z0 << std::endl;

            Farlor::Vector3 Zstar = Z0;
            for (size_t n = 0; n < oldBetas.size(); n++) {
                Z0 -= oldBetas[n];
                Zstar -= newBetas[n];
            }

            // std::cout << "Zstar: " << Zstar << std::endl;

            double z0 = Z0.Magnitude();
            double zstar = Zstar.Magnitude();

            // std::cout << "z0: " << z0 << std::endl;
            // std::cout << "zstar: " << zstar << std::endl;

            NormalizerDoubleType topFN = fn.eval(M - (int)newBetas.size(), zstar);
            NormalizerDoubleType bottomFN = fn.eval(M - (int)oldBetas.size(), z0);
            double zRational = z0 / zstar;

            // std::cout << "topFN: " << topFN << std::endl;
            // std::cout << "bottomFN: " << bottomFN << std::endl;
            // std::cout << "zRational: " << zRational << std::endl;

            NormalizerDoubleType zero(0.0);
            if ((bottomFN == zero) || (zstar == zero)) {
                return 0.0;
            }
            NormalizerDoubleType result = (topFN / bottomFN) * zRational;
            return result;
        }

        NormalizerDoubleType CalculateLikelihood(const BaseNormalizer &fn, int numSegments,
              const twisty::PerturbUtils::BoundaryConditions &boundaryConditions,
              std::vector<Farlor::Vector3> &oldBetas, std::vector<Farlor::Vector3> &newBetas)
        {
            // We store 1 tangent per segement plus an addition one for end.
            // The first and last are boundary, so we want the middle M-1
            assert(oldBetas.size() == newBetas.size());

            return likelihood(fn, numSegments - 1, boundaryConditions.arclength,
                  boundaryConditions.m_endPos, boundaryConditions.m_startPos,
                  boundaryConditions.m_endDir, boundaryConditions.m_startDir, oldBetas, newBetas);
        }

        std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> GetNormalizer(
              uint32_t numSegments)
        {
            const std::string fnFilename = "SavedFN.fnd";
            const std::filesystem::path fnFilePath = std::filesystem::current_path() / fnFilename;

            if (std::filesystem::exists(fnFilePath)) {
                std::cout << "Using cached fd file at: " << fnFilePath << std::endl;
                std::ifstream inFile(fnFilePath);
                std::unique_ptr<PathWeighting::NormalizerStuff::FN> upFN
                      = std::make_unique<PathWeighting::NormalizerStuff::FN>(inFile);
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
            std::unique_ptr<PathWeighting::NormalizerStuff::FN> upFN
                  = std::make_unique<PathWeighting::NormalizerStuff::FN>(
                        numZSamples, numIntegrationSamples, maxorder, rMin, rMax);

            std::ofstream outFile(fnFilePath);
            upFN->WriteToFile(outFile);
            outFile.close();
            return upFN;
        }
    }  // namespace NormalizerStuff
}  // namespace PathWeighting
}  // namespace twisty