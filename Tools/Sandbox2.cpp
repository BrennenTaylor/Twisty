#include <boost/multiprecision/cpp_dec_float.hpp>

#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

const uint32_t WeightsPerBatch = 1000000;

const double MaxDoubleLog10 = 300;
const double PathBatchOffsetConstant = 6.0;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Use format: " << argv[0] << " filename" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    std::ifstream inFile(filename);
    if (!inFile.is_open())
    {
        std::cout << "Failed to open: " << filename << std::endl;
        return 1;
    }

    uint32_t numWeights = 0;
    inFile >> numWeights;

    const uint32_t numBatches = numWeights / WeightsPerBatch;

    uint32_t numWeightsRemaining = numWeights;

    std::vector<double> batchShiftedTotals(numBatches);
    std::vector<double> batchDifferences(numBatches);

    for (uint32_t batchIdx = 0; batchIdx < numBatches; ++batchIdx)
    {
        const uint32_t weightsInCurrentBatch = std::min(WeightsPerBatch, numWeightsRemaining);
        // Assume we will handle the number of current weights in batch
        numWeightsRemaining -= weightsInCurrentBatch;

        std::vector<boost::multiprecision::cpp_dec_float_100> weights(weightsInCurrentBatch);
        for (uint32_t i = 0; i < weightsInCurrentBatch; ++i)
        {
            inFile >> weights[i];
        }

        // Do the ground truth total weight calculation
        {
            boost::multiprecision::cpp_dec_float_100 totalWeight = 0.0;
            for (uint32_t weightIdx = 0; weightIdx < weightsInCurrentBatch; ++weightIdx)
            {
                totalWeight += weights[weightIdx];
            }
            std::cout << "Total weight: " << totalWeight << std::endl;

        }

        // Assume this is what we have weight size
        std::vector<double> log10Weights(weightsInCurrentBatch);
        for (uint32_t i = 0; i < weightsInCurrentBatch; ++i)
        {
            boost::multiprecision::cpp_dec_float_100 log10Weight = boost::multiprecision::log10(weights[i]);
            log10Weights[i] = log10Weight.convert_to<double>();
        }

        // Tracks the current largets log10 of weight for this batch
        double maxWeightLog10 = log10Weights[0];
        // Tracks the amount needed to shift the current max weight * number of paths in batch to the top of double space log10
        double actingDifference = MaxDoubleLog10 - (maxWeightLog10 + PathBatchOffsetConstant);

        // Tracks the running weight in shifted space for this batch
        double runningTotal = std::pow(10, (log10Weights[0] + actingDifference));
        for (uint32_t weightIdx = 1; weightIdx < weightsInCurrentBatch; ++weightIdx)
        {
            double currentWeightLog10 = log10Weights[weightIdx];

            // If this checks out, we have the same maximum and thus can simply scale the new weight up by the current
            // difference and add it to the running total.
            if (maxWeightLog10 > currentWeightLog10)
            {
                runningTotal += std::pow(10, (currentWeightLog10 + actingDifference));
                continue;
            }

            // If we are past, then we have a new maximum and need to adjust
            // New difference
            double newDifference = MaxDoubleLog10 - (currentWeightLog10 + PathBatchOffsetConstant);

            // If we have a new max, the difference is smaller, and thus things get shifted down
            double differenceDelta = newDifference - actingDifference;

            // We convert the running total back to log10, then shift by the difference in delta to realign with the new maximum value.
            // Finally, we write this back to the running total
            double log10RunningTotal = std::log10(runningTotal);
            runningTotal = std::pow(10.0, (log10RunningTotal + differenceDelta));

            // Update
            maxWeightLog10 = currentWeightLog10;
            actingDifference = newDifference;

            runningTotal += std::pow(10, (currentWeightLog10 + actingDifference));
        }


        // Cache these for recombination of all batches later
        batchShiftedTotals[batchIdx] = runningTotal;
        batchDifferences[batchIdx] = actingDifference;

        std::cout << "Total weight double: " << runningTotal << std::endl;

        boost::multiprecision::cpp_dec_float_100 bigFloatTotalWeight = runningTotal;
        boost::multiprecision::cpp_dec_float_100 log10BigFloatWeight = boost::multiprecision::log10(bigFloatTotalWeight);
        boost::multiprecision::cpp_dec_float_100 adjustedLog10BigFloatWeight = log10BigFloatWeight - actingDifference;
        boost::multiprecision::cpp_dec_float_100 finalWeight = boost::multiprecision::pow(10, adjustedLog10BigFloatWeight);

        std::cout << "Final Weight: " << finalWeight << std::endl;
    }

    std::cout << "\n\nBatch recombination section: " << std::endl;

    for (uint32_t i = 0; i < numBatches; ++i)
    {
        std::cout << "Weight: " << batchShiftedTotals[i] << " - Difference: " << batchDifferences[i] << std::endl;
    }

    // This is what we shift by for batch recombination stuff
    const double MaxBatchConstant = std::log10(numBatches);

    double maxBatchUnshiftedTotalLog10 = std::log10(batchShiftedTotals[0]) - batchDifferences[0];
    double actingDifference = MaxDoubleLog10 - (maxBatchUnshiftedTotalLog10 + MaxBatchConstant);
    double runningBatchTotal = std::pow(10.0, (maxBatchUnshiftedTotalLog10 + actingDifference));

    std::cout << "Running Batch Total: " << runningBatchTotal << std::endl;
    {
        boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
        std::cout << "Unshifted batch total: " << boost::multiprecision::pow(10.0, (boost::multiprecision::log10(bigFloatRunningBatchTotal) - actingDifference)) << std::endl;
    }

    for (uint32_t batchIdx = 1; batchIdx < numBatches; ++batchIdx)
    {
        double currentShiftedBatchWeightLog10 = std::log10(batchShiftedTotals[batchIdx]);
        double currentUnshiftedBatchWeightLog10 = currentShiftedBatchWeightLog10 - batchDifferences[batchIdx];
        // If this checks out, we have the same maximum and thus can just adjust things up
        // TODO: Make this equal
        if (maxBatchUnshiftedTotalLog10 > currentUnshiftedBatchWeightLog10)
        {
            runningBatchTotal += std::pow(10, (currentUnshiftedBatchWeightLog10 + actingDifference));

            std::cout << "No new max: " << std::endl;
            std::cout << "Running Batch Total: " << runningBatchTotal << std::endl;
            {
                boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
                std::cout << "Unshifted batch total: " << boost::multiprecision::pow(10.0, (boost::multiprecision::log10(bigFloatRunningBatchTotal) - actingDifference)) << std::endl;
            }

            continue;
        }


        // If we are past, then we have a new maximum and need to adjust
        // New difference

        std::cout << "Previous actingDifference: " << actingDifference << std::endl;
        std::cout << "Previous maxBatchUnshiftedTotalLog10: " << maxBatchUnshiftedTotalLog10 << std::endl;

        double adjustedLog10RunningTotal = std::log10(runningBatchTotal) - actingDifference;
        actingDifference = MaxDoubleLog10 - (currentUnshiftedBatchWeightLog10 + MaxBatchConstant);
        runningBatchTotal = std::pow(10.0, (adjustedLog10RunningTotal + actingDifference));

        // Update
        maxBatchUnshiftedTotalLog10 = currentUnshiftedBatchWeightLog10;

        runningBatchTotal += std::pow(10, (maxBatchUnshiftedTotalLog10 + actingDifference));

        std::cout << "New actingDifference: " << actingDifference << std::endl;
        std::cout << "New maxBatchUnshiftedTotalLog10: " << maxBatchUnshiftedTotalLog10 << std::endl;

        std::cout << "Running Batch Total: " << runningBatchTotal << std::endl;
        {
            boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
            std::cout << "Unshifted batch total: " << boost::multiprecision::pow(10.0, (boost::multiprecision::log10(bigFloatRunningBatchTotal) - actingDifference)) << std::endl;
        }
    }

    std::cout << "Final Batch Total: " << runningBatchTotal << std::endl;
    {
        boost::multiprecision::cpp_dec_float_100 bigFloatRunningBatchTotal = runningBatchTotal;
        std::cout << "Unshifted final batch total: " << boost::multiprecision::pow(10.0, (boost::multiprecision::log10(bigFloatRunningBatchTotal) - actingDifference)) << std::endl;
    }
}