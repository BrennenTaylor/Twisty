#include <cstdint>

#include <PathWeightUtils.h>

int main()
{
    const uint32_t numSegmentsPerCurve = 66;
    const float minArclength = 10.0f;
    const float maxArclength = 20.0f;
    const float minDs = minArclength / numSegmentsPerCurve;
    const float maxDs = maxArclength / numSegmentsPerCurve;
    const uint32_t numArclengths = 10;

    // "weightFunction" : 0, "mu" : 0.1, "eps" : 0.075, "numStepsInt" : 20000,
    //       "numCurvatureSteps" : 10000, "absorption" : 0.004,
    //       "scatter" : 0.1

    twisty::WeightingParameters weightingParams;
    weightingParams.weightingMethod = twisty::WeightingMethod::RadiativeTransfer;
    weightingParams.mu = 0.1f;
    weightingParams.eps = 0.075f;
    weightingParams.numStepsInt = 20000;
    weightingParams.numCurvatureSteps = 10000;
    weightingParams.absorption = 0.04f;
    weightingParams.scatter = 0.1f;
    weightingParams.minBound = 0.0f;
    weightingParams.maxBound = 10.0f / weightingParams.eps;

    twisty::PathWeighting::CachedMultiArclengthWeightLookupTable computedLookupTable(
          weightingParams, minDs, maxDs, numArclengths);

    twisty::PathWeighting::CachedMultiArclengthWeightLookupTable loadedLookupTable(
          weightingParams, minDs, maxDs, numArclengths);

    // Check if the tables are equal
    if (computedLookupTable.GetUUID() != loadedLookupTable.GetUUID()) {
        std::cout << "FAILURE: Tables are not equal!" << std::endl;
        return 1;
    }

    // For each arclength, check if the tables are equal
    for (uint32_t dsIdx = 0; dsIdx < numArclengths; ++dsIdx) {
        const float ds = minDs + (maxDs - minDs) * dsIdx / (numArclengths - 1);
        twisty::PathWeighting::BaseWeightLookupTable *computedTableEntry
              = computedLookupTable.GetWeightLookupTable(ds);
        twisty::PathWeighting::BaseWeightLookupTable *loadedTableEntry
              = loadedLookupTable.GetWeightLookupTable(ds);


        //         const std::vector<float> &AccessLookupTable() const { return m_lookupTable; }

        // float GetMinSegmentWeight() const { return m_minSegmentWeight; }
        // float GetMaxSegmentWeight() const { return m_maxSegmentWeight; }

        // float GetMinCurvature() const { return m_minCurvature; }
        // float GetMaxCurvature() const { return m_maxCurvature; }

        // float GetDs() const { return m_ds; }

        // float GetCurvatureStepSize() const { return m_curvatureStepSize; }

        // WeightingParameters GetWeightingParams() const { return m_weightingParams; }

        if (computedTableEntry->GetWeightingParams().GenerateStringUUID().first
              != loadedTableEntry->GetWeightingParams().GenerateStringUUID().first) {
            std::cout << "FAILURE: Loaded are not equal: uuid_first!" << std::endl;
            return 1;
        }

        if (computedTableEntry->GetWeightingParams().GenerateStringUUID().second
              != loadedTableEntry->GetWeightingParams().GenerateStringUUID().second) {
            std::cout << "FAILURE: Loaded are not equal: uuid_second!" << std::endl;
            return 1;
        }

        if (std::abs(computedTableEntry->GetDs() - loadedTableEntry->GetDs()) > 0.0001f) {
            std::cout << "FAILURE: Loaded are not equal: ds!" << std::endl;
            std::cout << "computedTableEntry->GetDs() = " << computedTableEntry->GetDs()
                      << std::endl;
            std::cout << "loadedTableEntry->GetDs() = " << loadedTableEntry->GetDs() << std::endl;
            return 1;
        }

        if (std::abs(computedTableEntry->GetMinSegmentWeight()
                  - loadedTableEntry->GetMinSegmentWeight())
              > 0.0001f) {
            std::cout << "FAILURE: Loaded are not equal: min segment weight!" << std::endl;
            return 1;
        }

        if (std::abs(computedTableEntry->GetMaxSegmentWeight()
                  - loadedTableEntry->GetMaxSegmentWeight())
              > 0.0001f) {
            std::cout << "FAILURE: Loaded are not equal: max segment weight!" << std::endl;
            return 1;
        }

        if (std::abs(computedTableEntry->GetMinCurvature() - loadedTableEntry->GetMinCurvature())
              > 0.0001f) {
            std::cout << "FAILURE: Loaded are not equal: min curvature!" << std::endl;
            return 1;
        }

        if (std::abs(computedTableEntry->GetMaxCurvature() - loadedTableEntry->GetMaxCurvature())
              > 0.0001f) {
            std::cout << "FAILURE: Loaded are not equal: max curvature!" << std::endl;
            return 1;
        }

        if (std::abs(computedTableEntry->GetCurvatureStepSize()
                  - loadedTableEntry->GetCurvatureStepSize())
              > 0.0001f) {
            std::cout << "FAILURE: Loaded are not equal: curvature step size!" << std::endl;
            return 1;
        }

        const std::vector<float> &computedLookupTable = computedTableEntry->AccessLookupTable();
        const std::vector<float> &loadedLookupTable = loadedTableEntry->AccessLookupTable();
        // Check if the tables are equal
        if (computedLookupTable.size() != loadedLookupTable.size()) {
            std::cout << "FAILURE: Loaded are not equal: lookup table size!" << std::endl;
            return 1;
        }
        // Compare tables element by element
        for (uint32_t i = 0; i < computedLookupTable.size(); ++i) {
            if (std::abs(computedLookupTable[i] - loadedLookupTable[i]) > 0.0001f) {
                std::cout << "FAILURE: Loaded are not equal: lookup table element!" << std::endl;
                return 1;
            }
        }
    }

    std::cout << "SUCCESS: All tests pass!" << std::endl;
}