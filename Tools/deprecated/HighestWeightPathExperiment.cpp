#include <boost/multiprecision/cpp_dec_float.hpp>

//typedef boost::multiprecision::number < boost::multiprecision::cpp_dec_float<1000>> cpp_dec_float_custom;
typedef boost::multiprecision::cpp_dec_float_100 cpp_dec_float_custom;

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("Call as: %s dataFilename percentFromMax compressedWeightsLog10", argv[0]);
        return false;
    }

    std::string dataFilename = argv[1];
    dataFilename += ".txt";

    double percentFromMax = std::stod(argv[2]);

    std::stringstream outputSS;
    outputSS << argv[1];
    outputSS << "_HighestWeightPathInfo";
    outputSS << "_";
    outputSS << percentFromMax;
    outputSS << ".txt";

    bool compressedWeightsLog10 = (std::stoi(argv[3]) == 0) ? false : true;
    std::cout << "Compressed weights: " << compressedWeightsLog10 << std::endl;

    std::ifstream inDataFile(dataFilename);
    uint32_t numWeightValues = 0;
    inDataFile >> numWeightValues;

    std::ofstream outFile(outputSS.str());

    // Path Counts
    outFile << "Number of weight values: " << numWeightValues << std::endl;

    cpp_dec_float_custom weightMin = 0.0;
    cpp_dec_float_custom weightMax = 0.0;


    for (uint32_t weightIdx = 0; weightIdx < numWeightValues; ++weightIdx)
    {
        // Read in the current value
        cpp_dec_float_custom currentVal = 0.0;
        inDataFile >> currentVal;

        if (weightIdx == 0)
        {
            weightMin = currentVal;
            weightMax = currentVal;
        }
        else
        {
            if (currentVal < weightMin)
            {
                weightMin = currentVal;
            }

            if (currentVal > weightMax)
            {
                weightMax = currentVal;
            }
        }

    }

    outFile << "Min, Max Log10: " << weightMin << ", " << weightMax << std::endl;
    outFile << "Min, Max Value: " << boost::multiprecision::pow(10.0, weightMin) << ", " << boost::multiprecision::pow(10.0, weightMax) << std::endl;
    inDataFile.close();


    // Calculate 10% away under log 10 max weight as cutoff
    cpp_dec_float_custom maxValueBigFloat = weightMax;
    if (compressedWeightsLog10)
    {
        maxValueBigFloat = boost::multiprecision::pow(10.0, maxValueBigFloat);
    }

    double allowedAmount = (100.0 - percentFromMax) / 100.0;
    const cpp_dec_float_custom cutoff = maxValueBigFloat * allowedAmount;
    const cpp_dec_float_custom cutoffLog10 = boost::multiprecision::log10(cutoff);

    outFile << "Allowed amount: " << allowedAmount << std::endl;
    outFile << "Cutoff: " << cutoff << std::endl;
    outFile << "Cutoff Log10: " << cutoffLog10 << std::endl;

    // We want to reopen the data file
    inDataFile.open(dataFilename);
    numWeightValues = 0;
    inDataFile >> numWeightValues;


    uint32_t numMeetingCutoff = 0;

    for (uint32_t weightIdx = 0; weightIdx < numWeightValues; ++weightIdx)
    {
        cpp_dec_float_custom currentVal = 0.0;
        inDataFile >> currentVal;

        // If they are stored as big floats, we want to expand it out
        if (compressedWeightsLog10)
        {
            if (currentVal >= cutoffLog10)
            {
                numMeetingCutoff++;
            }
        }
        // Not compressed, compare directly
        else
        {
            if (currentVal >= cutoff)
            {
                numMeetingCutoff++;
            }
        }
    }
    inDataFile.close();
    
    outFile << "Num Meeting, Num Not-Meeting Counts: " << numMeetingCutoff<< ", " << (numWeightValues-numMeetingCutoff) << std::endl;
    outFile << "Num Meeting, Num Not-Meeting Percents: " << (double)numMeetingCutoff / numWeightValues * 100.0 << ", " << (double)(numWeightValues - numMeetingCutoff) / numWeightValues * 100.0 << std::endl;
}