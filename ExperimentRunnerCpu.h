/**
 * @file ExperimentRunnerCpu.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-03-18
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "ExperimentRunner.h"
#include "Range.h"
#include "PathWeightUtils.h"

#include <optional>
#include <random>

namespace twisty
{
    /**
     * @brief Implements the ExperimentRunner interface for running the experiment on the CPU
     * 
     */
    class ExperimentRunnerCpu : public ExperimentRunner
    {
    public:
        /**
         * @brief Construct a new Experiment Runner Cpu object
         * 
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         * @param kdsRange Range of allowed curvature * ds values
         * @param tdsRange Range of allowed torsion * ds values
         */
        ExperimentRunnerCpu(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper, Range kdsRange, Range tdsRange);
        virtual ~ExperimentRunnerCpu();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

    private:
        /*

        flag parameter is set to 0 for no errors.
        If the root solve fails, its set to 1
        If the root solve succeeds, but we dont accept the path due to calculated path error, we set to 2
        */
        std::unique_ptr<Curve> PurturbCurve(const Curve& curve, uint32_t& flag);

        std::unique_ptr<Curve> SimpleGeometryCurvePerturb(const Curve& curve, uint32_t& flag);
        std::unique_ptr<Curve> ComplexGeometryCurvePerturb(const Curve& curve, uint32_t& flag);
        std::unique_ptr<Curve> RootSolveCurvePerturb(const Curve& curve, uint32_t& flag);

    private:
        std::mt19937 m_rng;

        Range m_kdsRange;
        Range m_tdsRange;

        Range m_kRange;
        Range m_tRange;

        std::unique_ptr<PathSpaceUtils::RegularizedIntegral> m_upRegIntEvaluator;
    };
}