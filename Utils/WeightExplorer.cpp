#include "PathWeightUtils.h"

#include "ExperimentRunner.h"

#include "libconfig.h++"

bool LoadConfigFile(const std::string& filename, libconfig::Config& experimentConfig)
{
    try
    {
        experimentConfig.readFile(filename);
    }
    catch (const libconfig::FileIOException& fioex)
    {
        std::cout << "I/O error while reading file." << std::endl;
        return false;
    }
    catch (const libconfig::ParseException& pex)
    {
        std::cout << "Parse error at " << pex.getFile() << ":" << pex.getLine()
            << " - " << pex.getError() << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
        return 1;
    }
    std::string configFilename(argv[1]);
    libconfig::Config config;
    if (!LoadConfigFile(configFilename, config)) {
        std::cout << "Failed to load config file: " << configFilename << std::endl;
        return false;
    }

    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    try
    {
        experimentParams.numSegmentsPerCurve = (int)config.lookup("experiment.experimentParams.numSegments");
        experimentParams.arclength = config.lookup("experiment.experimentParams.arclength");

        // Weighting parameter stuff
        experimentParams.weightingParameters.mu = config.lookup("experiment.experimentParams.weighting.mu");
        experimentParams.weightingParameters.eps = config.lookup("experiment.experimentParams.weighting.eps");
        experimentParams.weightingParameters.numStepsInt = (int)config.lookup("experiment.experimentParams.weighting.numStepsInt");
        experimentParams.weightingParameters.numCurvatureSteps = (int)config.lookup("experiment.experimentParams.weighting.numCurvatureSteps");
        experimentParams.weightingParameters.absorbtion = config.lookup("experiment.experimentParams.weighting.absorbtion");

        std::vector<double> scatterValues;
        auto& scatterValuesLookup = config.lookup("experiment.experimentParams.weighting.scatterValues");
        experimentParams.weightingParameters.scatter = scatterValuesLookup[0];

        // TODO: Should these be configurable in the file?
        experimentParams.weightingParameters.minBound = 0.0;
        experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;

    }
    catch (const libconfig::SettingNotFoundException& ex)
    {
        std::cerr << "Parse error at " << ex.getPath() << ":" << ex.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }

 
    const float ds = experimentParams.arclength / experimentParams.numSegmentsPerCurve;

    twisty::PathWeighting::WeightLookupTableIntegral standardWeightLookup(experimentParams.weightingParameters, ds);
    standardWeightLookup.ExportValues("F:/Experiments/WeightExplorer/");
}