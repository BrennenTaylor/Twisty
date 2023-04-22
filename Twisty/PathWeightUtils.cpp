#include "PathWeightUtils.h"

#include "ExperimentRunner.h"
#include "FMath/Vector3.h"
#include "MathConsts.h"

#include "CurvePerturbUtils.h"

#include <omp.h>

#include <algorithm>
#include <assert.h>
#include <filesystem>
#include <fstream>

namespace twisty {
namespace PathWeighting {
    // Parameterized simple gaussian function
    float SimpleGaussianPhase(float p, float mu)
    {
        const float val = (-mu * p * p) * 0.5f;
        return expf(val);
    }

    // TODO: We need to verify this. It's not clear if this is the correct form.
    // I cannot find the documented reference anywhere... I dont know which work this is from.
    // For now, I will revert to that found in Pauls dissertation and
    // "ACCELERATED PATH GENERATION AND VISUALIZATION FOR NUMERICAL INTEGRATION OF FEYNMAN PATH INTEGRALS FOR RADIATIVE TRANSFER"
    float GaussianPhase(float evalLocation, float mu)
    {
        const float Np = sqrtf((TwistyPi * mu) * 0.5f) / (1.0f - expf(-2.0f / mu));
        return Np * SimpleGaussianPhase(evalLocation, mu);
    }

    // Parameterized gaussian function
    // Switched to version found in "A leading order approximation of the path integral for radiative transfer"
    // float GaussianPhase(float evalLocation, float mu)
    // {
    //     const float Np
    //           = sqrtf(8.0f * TwistyPi * TwistyPi * TwistyPi * mu) / (1.0f - expf(-2.0f / mu));
    //     return Np * SimpleGaussianPhase(evalLocation, mu);
    // }


    // Lookup table integrand
    BaseWeightLookupTable::BaseWeightLookupTable(const WeightingParameters &weightingParams,
          const float ds, const float minCurvature, const float maxCurvature)
        : m_weightingParams(weightingParams)
        , m_wpUUID(weightingParams.GenerateStringUUID())
        , m_ds(ds)
        , m_wtUUID(std::make_pair<std::string, uint64_t>("", 0))
        , m_minCurvature(minCurvature)
        , m_maxCurvature(maxCurvature)
        , m_curvatureStepSize(
                (m_maxCurvature - m_minCurvature) / (weightingParams.numCurvatureSteps - 1))
        , m_minSegmentWeight(0.0f)
        , m_maxSegmentWeight(0.0f)
        , m_lookupTable(
                weightingParams.numCurvatureSteps)  // We include 0, thus need that last spot
    {
        UpdateUUID();
    }

    void BaseWeightLookupTable::ExportValues(const std::string &directoryFileName) const
    {
        ExportValues(directoryFileName, ExportFilename());
    }

    void BaseWeightLookupTable::ExportValues(
          const std::string &directoryFileName, const std::string &filename) const
    {
        std::filesystem::path exportDirectory = directoryFileName;
        if (!std::filesystem::exists(exportDirectory)) {
            std::filesystem::create_directories(exportDirectory);
        }

        std::stringstream dsSS;
        std::ios_base::fmtflags flags = dsSS.flags();
        dsSS << std::fixed << std::setprecision(10) << m_ds;
        dsSS.flags(flags);

        std::filesystem::path exportFile = exportDirectory.string();
        exportFile /= (dsSS.str() + "_" + std::to_string(m_wtUUID.second) + "_" + filename);

        // std::cout << "Exporting table of size: " << m_lookupTable.size() << std::endl;
        std::ofstream outputFile(exportFile.c_str());
        if (!outputFile.is_open()) {
            throw std::runtime_error("Failed to export weight table");
        }
        for (uint32_t i = 0; i < m_lookupTable.size(); ++i) {
            const double curvatureEval = m_minCurvature + i * m_curvatureStepSize;
            outputFile << curvatureEval << ", " << m_lookupTable[i] << '\n';
        }
        outputFile.close();
    }

    void BaseWeightLookupTable::InitializeFromStream(std::ifstream &ifstream)
    {
        m_lookupTable.resize(m_weightingParams.numCurvatureSteps);
        ifstream.read(reinterpret_cast<char *>(m_lookupTable.data()),
              m_weightingParams.numCurvatureSteps * sizeof(float));

        ifstream.read(reinterpret_cast<char *>(&m_minSegmentWeight), sizeof(float));
        ifstream.read(reinterpret_cast<char *>(&m_maxSegmentWeight), sizeof(float));

        // std::cout << "Finished path weight integral lookup table" << '\n';
        // std::cout << "\tMin Possible Weight Value: " << m_minSegmentWeight << '\n';
        // std::cout << "\tMax Possible Weight Value: " << m_maxSegmentWeight << '\n';
        // // Parameters
        // std::cout << "\tTable construction params: " << '\n';
        // std::cout << "\t\tmu: " << m_weightingParams.mu << '\n';
        // std::cout << "\t\tnumStepsInt: " << m_weightingParams.numStepsInt << '\n';
        // std::cout << "\t\tm_minBound: " << m_weightingParams.minBound << '\n';
        // std::cout << "\t\tm_maxBound: " << m_weightingParams.maxBound << '\n';
        // std::cout << "\t\tm_eps: " << m_weightingParams.eps << '\n';
        // std::cout << "\t\tm_minCurvature: " << m_minCurvature << '\n';
        // std::cout << "\t\tm_maxCurvature: " << m_maxCurvature << '\n';
        // std::cout << "\t\tm_numCurvatureSteps: " << m_weightingParams.numCurvatureSteps
        //           << std::endl;
    }

    void BaseWeightLookupTable::StreamOut(std::ostream &outputStream) const
    {
        // Write out table
        outputStream.write(reinterpret_cast<const char *>(m_lookupTable.data()),
              sizeof(float) * m_lookupTable.size());
        outputStream.write(reinterpret_cast<const char *>(&m_minSegmentWeight), sizeof(float));
        outputStream.write(reinterpret_cast<const char *>(&m_maxSegmentWeight), sizeof(float));
    }

    void BaseWeightLookupTable::UpdateUUID()
    {
        m_wpUUID = m_weightingParams.GenerateStringUUID();
        std::stringstream uuid;
        std::ios_base::fmtflags flags = uuid.flags();
        uuid << std::fixed << std::setprecision(10) << "ds_" << m_ds;
        uuid.flags(flags);
        uuid << "wp_" << m_wpUUID.first;
        m_wtUUID = std::make_pair<std::string, uint64_t>(
              uuid.str(), std::hash<std::string> {}(uuid.str()));
    }

    CachedMultiArclengthWeightLookupTable::CachedMultiArclengthWeightLookupTable(
          const WeightingParameters &weightingParams, float minDs, float maxDs, uint32_t numDsSteps)
        : m_weightingParams(weightingParams)
        , m_minDs(minDs)
        , m_maxDs(maxDs)
        , m_numDsSteps(numDsSteps)
    {
        UpdateUUID();

        // Search Local Directory for existing tables
        std::filesystem::path exportDirectory
              = std::filesystem::current_path() / "CachedWeightTables/";
        std::filesystem::path searchFilename
              = exportDirectory / (std::to_string(m_cwtUUID.second) + ".cwt");
        if (std::filesystem::exists(searchFilename)) {
            // Initialize from filename
            std::cout << "Found cached table, loading: " << searchFilename.string() << std::endl;
            std::ifstream inputFile(searchFilename.string(), std::ios::binary);
            if (!inputFile.is_open()) {
                throw std::runtime_error("Failed to open cached weight table");
            }
            InitializeFromStream(inputFile);
        } else {
            std::cout << "No cached table found, we need to compute." << std::endl;
            Initialize();
            // Write out the tables
            std::filesystem::path writeTablesPath
                  = exportDirectory / std::to_string(m_cwtUUID.second);
            if (!std::filesystem::exists(writeTablesPath)) {
                std::filesystem::create_directories(writeTablesPath);
            }
            for (auto &table : m_weightLookupTables) {
                table->ExportValues(writeTablesPath.string());
            }
        }
    }

    void CachedMultiArclengthWeightLookupTable::Initialize()
    {
        const float dsStepSize = (m_maxDs - m_minDs) / (m_numDsSteps - 1);

        std::chrono::high_resolution_clock::time_point startTime
              = std::chrono::high_resolution_clock::now();

        for (uint32_t i = 0; i < m_numDsSteps; ++i) {
            const float ds = m_minDs + i * dsStepSize;
            if (m_weightingParams.weightingMethod == WeightingMethod::RadiativeTransfer) {
                m_weightLookupTables.push_back(std::move(
                      std::make_unique<WeightLookupTableIntegral>(m_weightingParams, ds)));
            } else {
                m_weightLookupTables.push_back(
                      std::move(std::make_unique<SimpleWeightLookupTable>(m_weightingParams, ds)));
            }
            // We initialize since we are computing
            m_weightLookupTables[i]->Initialize();
            std::ios_base::fmtflags flags = std::cout.flags();
            std::cout << "\rPercent Complete: " << std::fixed << std::setprecision(2)
                      << (100.0f * i) / m_numDsSteps << "%";

            // Estimated time to completion
            std::chrono::high_resolution_clock::time_point currentTime
                  = std::chrono::high_resolution_clock::now();
            // Elapsed time in seconds
            const float elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(
                  currentTime - startTime)
                                               .count();
            const float secondsPerStep = elapsedSeconds / (i + 1);
            const float secondsRemaining = secondsPerStep * (m_numDsSteps - i);

            const float mins = std::floor(secondsRemaining / 60.0f);
            const float secs = secondsRemaining - mins * 60.0f;

            std::cout << " Time Remaining: " << std::fixed << std::setprecision(0) << mins << "m"
                      << secs << "s\t\t" << std::flush;
            std::cout.flags(flags);
        }
        CacheWeightTable();
    }

    void CachedMultiArclengthWeightLookupTable::CacheWeightTable()
    {
        std::filesystem::path exportDirectory
              = std::filesystem::current_path() / "CachedWeightTables";
        if (!std::filesystem::exists(exportDirectory)) {
            std::filesystem::create_directories(exportDirectory);
        }
        std::filesystem::path cachedTableFilename
              = exportDirectory / (std::to_string(m_cwtUUID.second) + ".cwt");
        std::ofstream outputFile(cachedTableFilename, std::ios::binary);
        if (!outputFile.is_open()) {
            throw std::runtime_error("Failed to export weight table");
        }
        for (const auto &table : m_weightLookupTables) {
            table->StreamOut(outputFile);
        }
        outputFile.close();
    }

    // Currently just gets the closest table, but could be improved to do bilinear interpolation
    BaseWeightLookupTable *CachedMultiArclengthWeightLookupTable::GetWeightLookupTable(
          float ds) const
    {
        float minDist = std::numeric_limits<float>::max();
        BaseWeightLookupTable *closestTable = nullptr;
        for (const auto &table : m_weightLookupTables) {
            float dist = std::abs(table->GetDs() - ds);
            if (dist < minDist) {
                minDist = dist;
                closestTable = table.get();
            }
        }
        return closestTable;
    }

    void CachedMultiArclengthWeightLookupTable::UpdateUUID()
    {
        std::stringstream uuid;
        std::ios_base::fmtflags flags = uuid.flags();
        uuid << std::fixed << std::setprecision(10) << "minDs_" << m_minDs << "maxDs_" << m_maxDs;
        uuid.flags(flags);
        uuid << "numDsSteps_" << m_numDsSteps << "wp_"
             << m_weightingParams.GenerateStringUUID().first;
        m_cwtUUID = std::make_pair<std::string, uint64_t>(
              uuid.str(), std::hash<std::string> {}(uuid.str()));
    }

    void CachedMultiArclengthWeightLookupTable::InitializeFromStream(std::ifstream &filestream)
    {
        const float dsStepSize = (m_maxDs - m_minDs) / (m_numDsSteps - 1);
        for (uint32_t i = 0; i < m_numDsSteps; ++i) {
            const float ds = m_minDs + i * dsStepSize;
            if (m_weightingParams.weightingMethod == WeightingMethod::RadiativeTransfer) {
                m_weightLookupTables.push_back(std::move(
                      std::make_unique<WeightLookupTableIntegral>(m_weightingParams, ds)));
            } else {
                m_weightLookupTables.push_back(
                      std::move(std::make_unique<SimpleWeightLookupTable>(m_weightingParams, ds)));
            }
            m_weightLookupTables[i]->InitializeFromStream(filestream);
        }
    }

    // Lookup table integrand
    WeightLookupTableIntegral::WeightLookupTableIntegral(
          const WeightingParameters &weightingParams, float ds)
        : BaseWeightLookupTable(weightingParams, ds, 0.0f, (2.3f / ds) * 1.5f)
    {
        m_minCurvature = 0.0f;
        m_maxCurvature = (2.3f / ds) * 1.5f;
        m_curvatureStepSize
              = (m_maxCurvature - m_minCurvature) / (m_weightingParams.numCurvatureSteps - 1);
    }

    WeightLookupTableIntegral::~WeightLookupTableIntegral() { }

    void WeightLookupTableIntegral::Initialize()
    {
        std::vector<float> localLookupTable(m_weightingParams.numCurvatureSteps);

        // Handle first case
        {
            float value = Integrate(m_minCurvature, m_weightingParams, m_ds);
            localLookupTable[0] = value;
        }
        float min = localLookupTable[0];
        float max = localLookupTable[0];

        const int numThreads = omp_get_max_threads();
        std::vector<float> minValues(numThreads);
        std::vector<float> maxValues(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            minValues[i] = min;
            maxValues[i] = max;
        }

#pragma omp parallel for num_threads(numThreads) default(none)                                     \
      shared(localLookupTable, minValues, maxValues)
        for (int64_t i = 1; i < m_weightingParams.numCurvatureSteps; ++i) {
            const float curvatureEval = m_minCurvature + i * m_curvatureStepSize;
            const float value = Integrate(curvatureEval, m_weightingParams, m_ds);
            const int threadIdx = omp_get_thread_num();

            // Unique thread idx
            localLookupTable[i] = value;

            if (value < minValues[threadIdx] && value >= 0.0f) {
                minValues[threadIdx] = value;
            }

            if (value > maxValues[threadIdx] && value >= 0.0f) {
                maxValues[threadIdx] = value;
            }
        }

        m_lookupTable = std::move(localLookupTable);

        min = *std::min_element(minValues.begin(), minValues.end());
        max = *std::max_element(maxValues.begin(), maxValues.end());

        for (uint32_t i = 0; i < m_weightingParams.numCurvatureSteps; ++i) {
            float &value = m_lookupTable[i];
            if (value <= 0.0) {
                value = min;
            }
        }

        m_minSegmentWeight = min;
        m_maxSegmentWeight = max;

        // std::cout << "Finished path weight integral lookup table" << '\n';
        // std::cout << "\tMin Possible Weight Value: " << min << '\n';
        // std::cout << "\tMax Possible Weight Value: " << max << '\n';
        // // Parameters
        // std::cout << "\tTable construction params: " << '\n';
        // std::cout << "\t\tmu: " << m_weightingParams.mu << '\n';
        // std::cout << "\t\tnumStepsInt: " << m_weightingParams.numStepsInt << '\n';
        // std::cout << "\t\tm_minBound: " << m_weightingParams.minBound << '\n';
        // std::cout << "\t\tm_maxBound: " << m_weightingParams.maxBound << '\n';
        // std::cout << "\t\tm_eps: " << m_weightingParams.eps << '\n';
        // std::cout << "\t\tm_minCurvature: " << m_minCurvature << '\n';
        // std::cout << "\t\tm_maxCurvature: " << m_maxCurvature << '\n';
        // std::cout << "\t\tm_numCurvatureSteps: " << m_weightingParams.numCurvatureSteps
        //           << std::endl;
    }

    float WeightLookupTableIntegral::Integrate(
          float curvature, const WeightingParameters &weightingParams, float ds) const
    {
        const float kds = curvature * ds;
        const float bds = weightingParams.scatter * ds;

        const float transmissionFalloff = std::exp(-bds) / (2.0 * TwistyPi * TwistyPi);

        auto Integrand = [this, weightingParams](
                               const double p, const double kds, const double bds) -> double {
            double phaseFunction = GaussianPhase(p, weightingParams.mu);

            float scatteringTerm = p
                  * std::exp(bds * phaseFunction  // scatter piece
                        - 1.0
                              * (weightingParams.eps * weightingParams.eps * p * p
                                    * 0.5)  // regularizer
                  );

            float sinTerm = 0.0f;
            // TODO: Should we implement this as if (kds < smallAngleThreshold?)
            if (kds != 0.0) {
                sinTerm = sinf(kds * p) / kds;
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
          const twisty::WeightingParameters &weightingParams, float ds)
        : BaseWeightLookupTable(weightingParams, ds, -1.0f, 1.0f)
    {
        std::cout << "Calcuating path weight integral lookup table: "
                  << weightingParams.numCurvatureSteps << std::endl;
        m_minCurvature = -1.0f;
        m_maxCurvature = 1.0f;
        m_curvatureStepSize
              = (m_maxCurvature - m_minCurvature) / (weightingParams.numCurvatureSteps - 1);
    }

    SimpleWeightLookupTable::~SimpleWeightLookupTable() { }

    void SimpleWeightLookupTable::Initialize()
    {
        auto CalculateSimpleWeightValue
              = [](float curvature, WeightingParameters wp, float ds) -> float {
            const float alpha = 1.0 / (wp.scatter * ds * wp.mu);
            const float leftComponent = std::exp(-1.0 * alpha) / (ds * ds * ds);
            const float rightComponent = std::exp(alpha * curvature);
            return leftComponent * rightComponent;
        };

        // Handle first case
        {
            float value = CalculateSimpleWeightValue(m_minCurvature, m_weightingParams, m_ds);
            m_lookupTable[0] = value;
        }

        float min = m_lookupTable[0];
        float max = m_lookupTable[0];
        for (uint32_t i = 1; i <= m_weightingParams.numCurvatureSteps; ++i) {
            float curvatureEval = m_minCurvature + i * m_curvatureStepSize;
            float value = CalculateSimpleWeightValue(curvatureEval, m_weightingParams, m_ds);

            m_lookupTable[i] = value;

            if (value < min) {
                min = value;
            }

            if (value > max) {
                max = value;
            }
        }

        m_minSegmentWeight = min;
        m_maxSegmentWeight = max;

        // std::cout << "Finished path weight integral lookup table" << '\n';
        // std::cout << "\tMin Possible Weight Value: " << min << '\n';
        // std::cout << "\tMax Possible Weight Value: " << max << '\n';
        // // Parameters
        // std::cout << "\tTable construction params: " << '\n';
        // std::cout << "\t\tmu: " << m_weightingParams.mu << '\n';
        // std::cout << "\t\tnumStepsInt: " << m_weightingParams.numStepsInt << '\n';
        // std::cout << "\t\tm_minBound: " << m_weightingParams.minBound << '\n';
        // std::cout << "\t\tm_maxBound: " << m_weightingParams.maxBound << '\n';
        // std::cout << "\t\tm_eps: " << m_weightingParams.eps << '\n';
        // std::cout << "\t\tm_minCurvature: " << m_minCurvature << '\n';
        // std::cout << "\t\tm_maxCurvature: " << m_maxCurvature << '\n';
        // std::cout << "\t\tm_numCurvatureSteps: " << m_weightingParams.numCurvatureSteps
        //           << std::endl;
    }

    MinMaxCurvature CalcMinMaxCurvature(const twisty::WeightingParameters &wp, float ds)
    {
        switch (wp.weightingMethod) {
            case WeightingMethod::RadiativeTransfer: {
                // Our curvature lookups are using finite difference
                // ki = (ti+1 - ti-1) / (2.0 * ds)
                // Our max and min are calculated as follows

                MinMaxCurvature result;
                result.minCurvature = 0.0f;
                result.maxCurvature = (2.3f / ds) * 1.5f;
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
            outFile << nb_samples << '\n';
            outFile << nb_orders << '\n';
            outFile << zMin << '\n';
            outFile << zMax << '\n';

            for (uint32_t order = 2; order <= nb_orders; ++order) {
                std::cout << "Writing out order: " << order << std::endl;
                auto &samples = fNsets[order];
                outFile << order << std::endl;
                for (uint32_t sample = 0; sample < nb_samples; ++sample) {
                    outFile << samples[sample] << '\n';
                }
            }
            outFile.flush();
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
        NormalizerDoubleType FN::eval(int order, float z) const
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
        NormalizerDoubleType K4(const float z, const float ds)
        {
            float result = (z < 2.0) ? 1.0 : 0.0;
            result /= z;
            result *= 2.0 * TwistyPi;
            const float invDs = (1.0 / ds);
            result *= (invDs * invDs * invDs);
            return result;
        }

        NormalizerDoubleType F5(const float x, const float &z, const float ds)
        {
            const float rMag = std::sqrt((z * z) + 1.0 - (2.0 * z * x));
            return K4(rMag, ds);
        }

#define F(N, M)                                                                                    \
    NormalizerDoubleType F##N(const float x, const float &z, const float ds)                       \
    {                                                                                              \
        const float rMag = std::sqrt((z * z) + 1.0 - (2.0 * z * x));                               \
                                                                                                   \
        NormalizerDoubleType result = 0.0;                                                         \
        const uint32_t numSteps = 10000;                                                           \
                                                                                                   \
        const float xMin = -1.0;                                                                   \
        const float xMax = 1.0;                                                                    \
        const float dx = (xMax - xMin) / numSteps;                                                 \
                                                                                                   \
        /* Phi is vertical Component\
         Theta is angle around the axis*/                                                             \
                                                                                                   \
        for (uint32_t xIdx = 0; xIdx < numSteps; xIdx++) {                                         \
            const float xVal = xMin + xIdx * dx;                                                   \
                                                                                                   \
            const NormalizerDoubleType val = F##M(xVal, rMag, ds);                                 \
            result += val;                                                                         \
        }                                                                                          \
        result *= (2.0 * TwistyPi) * dx;                                                           \
                                                                                                   \
        return result;                                                                             \
    }

        F(6, 5)
        F(7, 6)
        F(8, 7)

#define K(N)                                                                                       \
    NormalizerDoubleType K##N(const double &z, const double ds)                                    \
    {                                                                                              \
        std::cout << "K" << N << std::endl;                                                        \
        NormalizerDoubleType result = 0.0;                                                         \
        const uint32_t numSteps = 10000;                                                           \
                                                                                                   \
        const float xMin = -1.0;                                                                   \
        const float xMax = 1.0;                                                                    \
        const float dx = (xMax - xMin) / numSteps;                                                 \
                                                                                                   \
        /* Phi is vertical Component\
        Theta is angle around the axis*/                                                             \
                                                                                                   \
        for (uint32_t xIdx = 0; xIdx < numSteps; xIdx++) {                                         \
            const float xVal = xMin + xIdx * dx;                                                   \
            const NormalizerDoubleType val = F##7(xVal, z, ds);                                    \
            result += val;                                                                         \
        }                                                                                          \
        result *= (2.0 * TwistyPi) * dx;                                                           \
                                                                                                   \
        return result;                                                                             \
    }

        K(5)
        K(6)
        K(7)
        K(8)
        K(9)
        K(10)

        // Calculate the overall normalization
        NormalizerDoubleType Norm(int numberOfSegments, float ds,
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


            switch (numberOfSegments) {
                case 4: {
                    return K4(z, ds);
                } break;
                case 5: {
                    return K5(z, ds);
                } break;
                case 6: {
                    return K6(z, ds);
                } break;
                case 7: {
                    return K7(z, ds);
                } break;
                case 8: {
                    return K8(z, ds);
                } break;
                case 9: {
                    return K9(z, ds);
                } break;
                case 10: {
                    return K10(z, ds);
                } break;
                default: {
                    std::cout << "Invalid number of segments, currently cannot calculate normalizer"
                              << std::endl;
                    return 0.0;
                } break;
            };
        }
    }  // namespace PathWeighting
}  // namespace twisty
}