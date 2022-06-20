#include "PathWeightUtils.h"

#include "ExperimentRunner.h"
#include "FMath/Vector3.h"
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
        const double val = (-mu * p * p) * 0.5;
        return std::exp(val);
    }

    // Parameterized gaussian function
    // Switched to version found in "A leading order approximation of the path integral for radiative transfer"
    double GaussianPhase(double evalLocation, double mu)
    {
        // 8.0 is 2.0^3
        const double Np
              = std::sqrt(8.0 * TwistyPi * TwistyPi * TwistyPi * mu) / (1.0 - std::exp(-2.0 / mu));
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
        const double kds = curvature * ds;
        const double bds = weightingParams.scatter * ds;

        const double cds = (weightingParams.absorbtion + weightingParams.scatter) * ds;
        const double transmissionFalloff = std::exp(-cds) / (2.0 * TwistyPi * TwistyPi);

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

        const double stepSize = (weightingParams.maxBound - weightingParams.minBound)
              / (weightingParams.numStepsInt);

        double firstVal = 0.0f;
        double normalizerWithZeroCurvature = 0.0f;
        {
            for (uint32_t i = 0; i <= weightingParams.numStepsInt; ++i) {
                const double p = weightingParams.minBound + (i * stepSize);
                firstVal += Integrand(p, kds, bds) * stepSize;
                normalizerWithZeroCurvature += Integrand(p, 0.0, bds) * stepSize;
            }
        }
        return transmissionFalloff * firstVal / normalizerWithZeroCurvature;
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

    namespace NormalizerStuff {
        FN::FN(int samples, int numIntegrationSamples, int order, const double &zMin,
              const double &zMax)
            : nb_samples(samples)
            , nb_orders(order)
            , zMin(zMax)
            , zMax(zMax)
        {
            init(numIntegrationSamples);
        }

        FN::FN(std::ifstream &inFile)
            : nb_samples(0)
            , nb_orders(0)
            , zMin(0)
            , zMax(0)
        {
            init_fromFile(inFile);
        }


        void FN::WriteToFile(std::ofstream &outFile)
        {
            outFile << nb_samples << std::endl;
            outFile << nb_orders << std::endl;
            outFile << zMin << std::endl;
            outFile << zMax << std::endl;

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
            inFile >> zMin;
            inFile >> zMax;

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
        }

        void FN::init(int numIntegrationSamples)
        {
            //             std::cout << "# order=" << 4 << std::endl;
            //             std::vector<NormalizerDoubleType> f4(nb_samples);

            //             // Assume we have a range of possible r values, we want to go ahead and
            //             // calculate and store them
            //             const double dr = (zMax - zMin) / (nb_samples - 1);
            //             // For each r-value sample, calculate the base order pair (M = 4, r)
            //             for (int i = 0; i < nb_samples; i++) {
            //                 double r = zMin + i * dr;
            //                 f4[i] = F4(r);
            //             }
            //             fNsets[4] = f4;

            //             // Now we continue on and calculate the next orders up until maxorder
            //             // This is currently 200 in our case
            //             for (int o = 3; o <= nb_orders; o++) {
            //                 if ((o % 1) == 0) {
            //                     std::cout << "# order=" << o << std::endl;
            //                 }

            //                 std::vector<NormalizerDoubleType> f(nb_samples);
            // #pragma omp parallel for
            //                 for (int i = 0; i < nb_samples; i++) {
            //                     double r = zMin + i * dr;
            //                     double lower = std::fabs(r - 1.0);
            //                     double upper = r + 1.0;

            //                     // we use the same number of samples as our integral number of samples
            //                     double ddr = (upper - lower) / (numIntegrationSamples - 1);
            //                     NormalizerDoubleType accum = 0.0;
            //                     for (int i = 0; i < numIntegrationSamples; i++) {
            //                         // Evaluation point is: rp
            //                         // Current order is: o
            //                         // Order below is: o - 1
            //                         double rp = lower + i * ddr;
            //                         // Add in bar representing ddr * r *f_{M-1}(r)
            //                         accum += ddr * rp * eval(o - 1, rp);
            //                     }
            //                     f[i] = accum;
            //                 }
            //                 fNsets[o] = f;
            //             }
        }

        // Evaluate f of a given order at an z magnitude
        NormalizerDoubleType FN::eval(int order, double z) const
        {
            // // Get the stored dataset for the specific order
            // const std::vector<NormalizerDoubleType> &data = fNsets.at(order);
            // const long double dz = (zMax - zMin) / (nb_samples - 1);
            // // Next, find the bucket tp the left
            // int leftIdx = (int)(z - zMin) / dz;
            // // Bind weights to the known table
            // leftIdx = std::min(leftIdx, nb_samples - 1);
            // leftIdx = std::max(leftIdx, 0);

            // // Calculate right side value and bind to table as well
            // int rightIdx = leftIdx + 1;
            // rightIdx = std::min(rightIdx, nb_samples - 1);
            // rightIdx = std::max(rightIdx, 0);

            // double distance = z - zMin;
            // double leftDist = distance - (leftIdx * dz);

            // // Finally, our datapoint is a bilinear interpolation
            // return data[leftIdx] * (1.0f - leftDist) + data[rightIdx] * leftDist;
            return 0.0;
        }

        // The smallest order of F we can numerically evaluate
        NormalizerDoubleType K4(const double z, const double ds)
        {
            double result = (z < 2.0) ? 1.0 : 0.0;
            result /= z;
            result *= 2.0 * TwistyPi;
            const double invDs = (1.0 / ds);
            result *= (invDs * invDs * invDs);
            return result;
        }

        NormalizerDoubleType F5(
              const Farlor::Vector3 &betaVec, const Farlor::Vector3 &zVec, const double ds)
        {
            const Farlor::Vector3 rVec = zVec + betaVec;
            const double rMag = rVec.Magnitude();
            // std::cout << "rMag: " << rMag << std::endl;
            return K4(rMag, ds);
        }

        NormalizerDoubleType K5(const Farlor::Vector3 &zVec, const double ds)
        {
            std::cout << "K5" << std::endl;
            NormalizerDoubleType result = 0.0;
            const uint32_t numPhiSteps = 10000;
            const uint32_t numThetaSteps = 10000;

            const double phiMin = -TwistyPi;
            const double phiMax = -TwistyPi;
            const double deltaPhi = (phiMax - phiMin) / numThetaSteps;

            const double thetaMin = 0.0;
            const double thetaMax = TwistyPi;
            const double deltaTheta = (thetaMax - thetaMin) / numThetaSteps;

            for (uint32_t phiIdx = 0; phiIdx < numPhiSteps; phiIdx++) {
                for (uint32_t thetaIdx = 0; thetaIdx < numThetaSteps; thetaIdx++) {
                    const double phi = phiMin + phiIdx * deltaPhi;
                    const double theta = thetaMin + thetaIdx * deltaTheta;
                    const Farlor::Vector3 betaEval(std::sin(phi) * std::cos(theta),
                          std::sin(phi) * std::sin(theta), std::cos(phi));
                    // std::cout << "Beta: " << betaEval << std::endl;
                    // std::cout << "Zvec: " << zVec << std::endl;
                    const NormalizerDoubleType val = F5(betaEval.Normalized(), zVec, ds);
                    // std::cout << "Calcualted F5: " << val << std::endl;
                    result += val * (deltaTheta * deltaPhi);
                }
            }

            return result;
        }

        // Calculate the overall normalization
        NormalizerDoubleType Norm(int numberOfSegments, double ds,
              twisty::PerturbUtils::BoundaryConditions boundaryConditions)
        {
            // For M <= 3, the normalization is undefined or has a delta function
            assert(numberOfSegments > 3);
            std::cout << "Number of Segments: " << numberOfSegments << std::endl;
            std::cout << "ds: " << ds << std::endl;

            const Farlor::Vector3 zVec
                  = (boundaryConditions.m_endPos - boundaryConditions.m_startPos) * (1.0f / ds)
                  - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
            const float z = zVec.Magnitude();

            // Special case
            if (numberOfSegments == 4) {
                return K4(z, ds);
            }

            if (numberOfSegments == 5) {
                return K5(zVec, ds);
            }

            std::cout << "Invalid number of segments, currently cannot calculate normalizer"
                      << std::endl;
            return 0.0;
        }
    }  // namespace PathWeighting
}  // namespace twisty
}