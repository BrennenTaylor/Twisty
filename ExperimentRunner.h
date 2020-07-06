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
        enum class CurvePerturbMethod
        {
            SimpleGeometry = 0,
            ComplexGeometry = 1,
            RootSolve = 2
        };

        //struct ExperimentSegmentTorsion
        //{
        //    float m_curvature;
        //    float m_x;
        //    float m_y;
        //    float m_z;
        //    float m_tx;
        //    float m_ty;
        //    float m_tz;
        //};

        //using KTSegments = std::vector<ExperimentSegmentTorsion>;
        
        struct PathBatch
        {
            std::vector<float> m_curvatures;
            std::vector<Farlor::Vector3> m_positions;
            std::vector<Farlor::Vector3> m_tangents;
            std::vector<bool> perPathVailidity;
            uint32_t numberOfPathsInBatch;
            uint32_t index;
        };

        struct WeightingParameters
        {
            double mu = 0.1;
            uint32_t numStepsInt = 2000;
            double minBound = 0.0;
            double maxBound = 100.0;
            double eps = 0.01;

            double scatter = 0.0;
            double absorbtion = 0.0;

            uint32_t numCurvatureSteps = 10000;
        };

        struct ExperimentParameters
        {
            uint32_t numPathsInExperiment = 0;
            uint32_t numPathsPerBatch = 1000000;
            uint32_t numPathsToSkip = 0;
            uint32_t numSegmentsPerCurve = 3;
            uint32_t exportPathBatchSize = 1000;
            bool exportGeneratedCurves = false;
            std::string experimentName = "Default_Experiment";
            std::string experimentDir = "Default_Experiment/";
            std::string pathBatchPrepend = "";
            float maximumBootstrapCurveError = 0.1f;
            uint32_t curvePurturbSeed = 0;
            CurvePerturbMethod curvePerturbMethod = CurvePerturbMethod::SimpleGeometry;
            WeightingParameters weightingParameters;
            float rotateInitialSeedCurveRadians = 0.0f;
        };

        struct ExperimentResults
        {
            // Double, float, whatever, weight value
            boost::multiprecision::cpp_dec_float_100 experimentWeight = 0.0;
            uint32_t totalPathsGenerated = 0;
            uint32_t numFailedPaths = 0;
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

    private:
        rapidjson::Document m_pathBatchJsonIndex;
        rapidjson::Value m_pathBatchLinks;
    };
}