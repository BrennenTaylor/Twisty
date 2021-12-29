#include "FullExperimentRunnerOptimalPerturb.h"

#if defined(USE_CUDA)
#include "FullExperimentRunnerOptimalPerturbOptimized_GPU.h" 
#endif

#include <Bootstrapper.h>
#include <MathConsts.h>
#include <PathWeightUtils.h>

#include <FMath/Vector3.h>

#include <libconfig.h++>

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <filesystem>
#include <string>

twisty::ExperimentRunner::ExperimentParameters ParseExperimentParamsFromConfig(const libconfig::Config& config)
{
    twisty::ExperimentRunner::ExperimentParameters experimentParams;

    // Hardocded values
    experimentParams.maximumBootstrapCurveError = 0.5f;
    experimentParams.rotateInitialSeedCurveRadians = 0.0f;

    try
    {
        // Values loaded from the config file
        experimentParams.numPathsInExperiment = (long long)config.lookup("experiment.experimentParams.pathsToGenerate");
        experimentParams.numPathsToSkip = (int)config.lookup("experiment.experimentParams.pathsToSkip");
        experimentParams.experimentName = config.lookup("experiment.experimentParams.name").c_str();
        experimentParams.experimentDirPath = config.lookup("experiment.experimentParams.experimentDir").c_str();
        experimentParams.experimentDirPath += "/" + experimentParams.experimentName;

        experimentParams.maxPerturbThreads = (int)config.lookup("experiment.experimentParams.maxPerturbThreads");
        experimentParams.maxWeightThreads = (int)config.lookup("experiment.experimentParams.maxWeightThreads");

        experimentParams.outputBigFloatWeights = (bool)config.lookup("experiment.experimentParams.outputBigFloatWeights");
        experimentParams.outputPathBatches = (bool)config.lookup("experiment.experimentParams.outputPathBatches");
        experimentParams.useGpu = (bool)config.lookup("experiment.experimentParams.useGpu");

        experimentParams.numSegmentsPerCurve = (int)config.lookup("experiment.experimentParams.numSegments");
        experimentParams.arclength = config.lookup("experiment.experimentParams.arclength");

        experimentParams.bootstrapSeed = (int)config.lookup("experiment.experimentParams.random.bootstrapSeed");
        experimentParams.curvePurturbSeed = (int)config.lookup("experiment.experimentParams.random.perturbSeed");

        // Weighting parameter stuff
        int weightFunction = (int)config.lookup("experiment.experimentParams.weighting.weightFunction");
        switch (weightFunction)
        {
            // Radiative Transfer weight function
            case 0:
            {
                experimentParams.weightingParameters.weightingMethod = twisty::WeightingMethod::RadiativeTransfer;
            } break;

            // Simplified Model
            case 1:
            {
                experimentParams.weightingParameters.weightingMethod = twisty::WeightingMethod::SimplifiedModel;
            }
            break;

            // Default to the simplified model
            default:
            {
                experimentParams.weightingParameters.weightingMethod = twisty::WeightingMethod::RadiativeTransfer;
            }   break;
        }

        // Perturb method stuff
        int perturbMethod = (int)config.lookup("experiment.experimentParams.perturbMethod");
        switch (perturbMethod)
        {
            // Simplified Model
            case 1:
            {
                experimentParams.perturbMethod = twisty::ExperimentRunner::PerturbMethod::GeometricMinCurvature;
            }
            break;

            // Simplified Model
            case 2:
            {
                experimentParams.perturbMethod = twisty::ExperimentRunner::PerturbMethod::GeometricCombined;
            }
            break;

            // Default to the simplified model
            case 0:
            default:
            {
                experimentParams.perturbMethod = twisty::ExperimentRunner::PerturbMethod::GeometricRandom;
            }   break;
        }

        experimentParams.weightingParameters.mu = config.lookup("experiment.experimentParams.weighting.mu");
        experimentParams.weightingParameters.eps = config.lookup("experiment.experimentParams.weighting.eps");
        experimentParams.weightingParameters.numStepsInt = (int)config.lookup("experiment.experimentParams.weighting.numStepsInt");
        experimentParams.weightingParameters.numCurvatureSteps = (int)config.lookup("experiment.experimentParams.weighting.numCurvatureSteps");
        experimentParams.weightingParameters.absorbtion = config.lookup("experiment.experimentParams.weighting.absorbtion");
        
        std::vector<double> scatterValues;
        auto& scatterValuesLookup = config.lookup("experiment.experimentParams.weighting.scatterValues");
        for (int scatterIdx = 0; scatterIdx < scatterValuesLookup.getLength(); ++scatterIdx)
        {
            scatterValues.push_back(scatterValuesLookup[scatterIdx]);
        }
        experimentParams.weightingParameters.scatterValues = scatterValues;
    }
    catch (const libconfig::SettingNotFoundException &ex)
    {
        std::cerr << "Parse error at " << ex.getPath() << ":" << ex.what() << std::endl;
    }
    catch (const libconfig::SettingTypeException& ex)
    {
        std::cerr << "Setting Type Exception " << ex.getPath() << ":" << ex.what() << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    

    // TODO: Should these be configurable in the file?
    experimentParams.weightingParameters.minBound = 0.0;
    experimentParams.weightingParameters.maxBound = 10.0 / experimentParams.weightingParameters.eps;

    return experimentParams;
}

bool LoadConfigFile(const std::string& filename, libconfig::Config& experimentConfig)
{
    try
    {
        experimentConfig.readFile(filename);
    }
    catch (const libconfig::FileIOException &fioex)
    {
        std::cout << "I/O error while reading file." << std::endl;
        return false;
    }
    catch (const libconfig::ParseException &pex)
    {
        std::cout << "Parse error at " << pex.getFile() << ":" << pex.getLine()
                    << " - " << pex.getError() << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char *argv[])
{
    {
        if (argc < 2)
        {
            std::cout << "Call as: " << argv[0] << " configFilename" << std::endl;
            return 1;
        }
        std::string configFilename(argv[1]);
        libconfig::Config experimentConfig;
        if (!LoadConfigFile(configFilename, experimentConfig)) {
            std::cout << "Failed to load config file: " << configFilename << std::endl;
            return false;
        }

        twisty::ExperimentRunner::ExperimentParameters experimentParams = ParseExperimentParamsFromConfig(experimentConfig);

        if (!std::filesystem::exists(experimentParams.experimentDirPath))
        {
            std::filesystem::create_directories(experimentParams.experimentDirPath);
        }
        std::cout << "experimentDirPath: " << experimentParams.experimentDirPath << std::endl;
        const std::string experimentCfgCopyFilename = std::string(experimentParams.experimentDirPath) + "/" + experimentParams.experimentName + ".cfg";

        if (!std::filesystem::exists(experimentCfgCopyFilename)) {
            std::filesystem::copy_file(configFilename, experimentCfgCopyFilename, std::filesystem::copy_options::overwrite_existing);
        }

        twisty::PerturbUtils::BoundaryConditions experimentGeometry;

        try
        {
            {
                float x = (float)experimentConfig.lookup("experiment.geometry.startPos.x");
                float y = (float)experimentConfig.lookup("experiment.geometry.startPos.y");
                float z = (float)experimentConfig.lookup("experiment.geometry.startPos.z");
                experimentGeometry.m_startPos = Farlor::Vector3(x, y, z);
            }
            {
                float x = (float)experimentConfig.lookup("experiment.geometry.startDir.x");
                float y = (float)experimentConfig.lookup("experiment.geometry.startDir.y");
                float z = (float)experimentConfig.lookup("experiment.geometry.startDir.z");
                experimentGeometry.m_startDir = Farlor::Vector3(x, y, z).Normalized();
            }

            {
                float x = (float)experimentConfig.lookup("experiment.geometry.endPos.x");
                float y = (float)experimentConfig.lookup("experiment.geometry.endPos.y");
                float z = (float)experimentConfig.lookup("experiment.geometry.endPos.z");
                experimentGeometry.m_endPos = Farlor::Vector3(x, y, z);
            }
            {
                float x = (float)experimentConfig.lookup("experiment.geometry.endDir.x");
                float y = (float)experimentConfig.lookup("experiment.geometry.endDir.y");
                float z = (float)experimentConfig.lookup("experiment.geometry.endDir.z");
                experimentGeometry.m_endDir = Farlor::Vector3(x, y, z).Normalized();
            }
        }
        catch (const libconfig::SettingNotFoundException &ex)
        {
            std::cerr << "Parse error at " << ex.getPath() << ":" << ex.what() << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return 1;
        }

        experimentGeometry.arclength = experimentParams.arclength;

        // We run the experiment as well
        std::string outputDataFilename = std::string(experimentParams.experimentDirPath) + std::string("/Results.dat");
        std::ofstream ofs(outputDataFilename);

        experimentParams.perExperimentDirSubfolder = std::string("main");

        twisty::Bootstrapper bootstrapper(experimentGeometry);

        std::cout << "Experiment Path Count: " << experimentParams.numPathsInExperiment << std::endl;

        std::unique_ptr<twisty::ExperimentRunner> upExperimentRunner = nullptr;

        #if defined(USE_CUDA)
        if (experimentParams.useGpu)
        {
            upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturbOptimized_GPU>(experimentParams, bootstrapper);
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb_Gpu" << std::endl;
        }
        else
        {
            upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);
            std::cout << "Selected Runner Method: FullExperimentRunnerOptimalPerturb" << std::endl;
        }
        #else
        if (experimentParams.useGpu)
        {
            std::cout << "Error, gpu requested but not supported on this platform; defaulting to CPU" << std::endl;
        }
        upExperimentRunner = std::make_unique<twisty::FullExperimentRunnerOptimalPerturb>(experimentParams, bootstrapper);
        #endif

        auto start = std::chrono::high_resolution_clock::now();
        std::optional<twisty::ExperimentRunner::ExperimentResults> optionalResults = upExperimentRunner->RunExperiment();
        auto end = std::chrono::high_resolution_clock::now();
        auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        auto timeSec = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

        if (!optionalResults.has_value())
        {
            std::cout << "Experiment failed: no results returned." << std::endl;
            return 1;
        }
        const twisty::ExperimentRunner::ExperimentResults& results = optionalResults.value();

        std::cout << "Paths Generated: " << results.totalPathsGenerated << std::endl;
        
        auto experimentWeights = results.experimentWeights;
        
        for (int scatterIdx = 0; scatterIdx < results.experimentWeights.size(); scatterIdx++)
        {

            std::cout << "\tTotal experiment weight " << scatterIdx << ": " << results.experimentWeights[scatterIdx] << std::endl;
            std::cout << "\tAvg path weight " << scatterIdx << ": " << results.experimentWeights[scatterIdx] / results.totalPathsGenerated << std::endl;
            ofs << "Scatter value: " << experimentParams.weightingParameters.scatterValues[scatterIdx] << std::endl;
            ofs << "\tTotal experiment weight " << scatterIdx << ": " << results.experimentWeights[scatterIdx] << std::endl;
            ofs << "\tAvg path weight " << scatterIdx << ": " << results.experimentWeights[scatterIdx] / results.totalPathsGenerated << std::endl;
            ofs << "\tTotal experiment time (ms) " << results.totalExperimentMS << std::endl;
            ofs << "\tSetup time (ms) " << results.setupExperimentMS << std::endl;
            ofs << "\tPerturb time (ms) " << results.perturbExperimentMS << std::endl;
            ofs << "\tWeighting time (ms) " << results.weightingExperimentMS << std::endl;
        }
        ofs << std::endl;

        // Retrieve Data we want from experiment
        upExperimentRunner.reset(nullptr);        
        ofs.close();
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
    _CrtDumpMemoryLeaks();
    _CrtMemDumpAllObjectsSince(NULL);
#endif

    return 0;
}
