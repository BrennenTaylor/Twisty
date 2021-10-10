#pragma once

#include "Bootstrapper.h"

#include "PathWeightUtils.h"
#include <boost/multiprecision/cpp_dec_float.hpp>

#pragma warning(push, 0)
#include <rapidjson/document.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#pragma warning(pop)

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace twisty
{
    class ExperimentRunner
    {
    public:
        struct PathBatch
        {
            std::vector<float> m_curvatures;
            std::vector<Farlor::Vector3> m_positions;
            std::vector<Farlor::Vector3> m_tangents;
            std::vector<bool> perPathVailidity;
            uint64_t numberOfPathsInBatch;
            uint64_t index;
        };

        enum class PathNormalizerType
        {
            Default, // This is a default normalizer of 1
            PDF // This is the one derived for the phytra 5 work
        };

        struct ExperimentParameters
        {
            uint64_t numPathsInExperiment = 0;
            uint64_t numPathsPerBatch = 1000000;
            uint64_t numPathsToSkip = 0;
            uint32_t numSegmentsPerCurve = 3;
            float arclength = 0.0f;
            uint32_t exportPathBatchSize = 1000;
            bool exportGeneratedCurves = false;
            std::string experimentName = "Default_Experiment";
            std::string experimentDirPath = "./Default_Experiment/";
            std::string perExperimentDirSubfolder = "";
            std::string pathBatchPrepend = "";
            float maximumBootstrapCurveError = 0.1f;
            uint32_t bootstrapSeed = 0;
            uint32_t curvePurturbSeed = 0;
            twisty::WeightingParameters weightingParameters;
            float rotateInitialSeedCurveRadians = 0.0f;
            PathNormalizerType pathNormalizerType = PathNormalizerType::PDF;
        };

        struct ExperimentResults
        {
            // Double, float, whatever, weight value
            std::vector<boost::multiprecision::cpp_dec_float_100> experimentWeights;
            uint64_t totalPathsGenerated = 0;
            uint64_t numFailedPaths = 0;
        };

    public:
        ExperimentRunner(ExperimentParameters& experimentParams, Bootstrapper& bootstrapper);
        virtual ~ExperimentRunner();

        virtual bool Setup() = 0;
        virtual ExperimentResults RunExperiment() = 0;
        virtual void Shutdown() = 0;

        Curve* GetInitialCurvePtr() const
        {
            return m_upInitialCurve.get();
        }

    protected:
        bool BeginPathBatchOutput();
        void OutputPathBatch(PathBatch& pathBatch);
        void EndPathBatchOutput();

    protected:
        ExperimentParameters& m_experimentParams;
        Bootstrapper& m_bootstrapper;
        std::unique_ptr<Curve> m_upInitialCurve;
        std::string m_pathBatchOutputPath;

    private:
        rapidjson::Document m_pathBatchJsonIndex;
        rapidjson::Value m_pathBatchLinks;
    };
}