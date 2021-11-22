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
#include "PathWeightUtils.h"
#include "PerturbUtils.h"

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <optional>
#include <random>

namespace twisty
{
    /**
     * @brief Implements the ExperimentRunner interface for running the experiment on the CPU
     *
     */
    class FullExperimentRunnerOptimalPerturbOptimized : public ExperimentRunner
    {
    public:

        // Responsable for storing up to 10^6 big float values as double internally, while maintaining a significant amount of precision
        class CombinedWeightValues
        {
        public:
            inline static const double MaxDoubleLog10 = 300;
            inline static const double MaxNumberOfPathsLog10 = 6.0;
            inline static const uint32_t MaxNumberOfPaths = 1000000;

        public:
            void AddValue(double valueLog10)
            {
                // In the case we haven't added a value yet, we can early out
                if (m_numValues == 0)
                {
                    m_maxWeightLog10 = valueLog10;
                    m_maxPossibleFinalWeightLog10 = m_maxWeightLog10 + MaxNumberOfPathsLog10;
                    m_offset = MaxDoubleLog10 - m_maxPossibleFinalWeightLog10;
                    m_runningTotal += std::pow(10, (valueLog10 + m_offset));
                    m_numValues++;
                    return;
                }


                // If we already have a value and its not larger than the current max, then throw it in.
                if (m_maxWeightLog10 > valueLog10)
                {
                    m_runningTotal += std::pow(10, (valueLog10 + m_offset));
                    m_numValues++;
                    return;
                }

                // If it is larger, we need to rescale everything around that new value
                
                // New difference
                double newMaxPossibleFinalWeightLog10 = valueLog10 + MaxNumberOfPathsLog10;
                double newOffset = MaxDoubleLog10 - newMaxPossibleFinalWeightLog10;

                double offsetDelta = newOffset - m_offset;
                double log10RunningTotal = std::log10(m_runningTotal);
                double adjustedLog10RunningTotal = log10RunningTotal + offsetDelta;
                m_runningTotal = std::pow(10.0, adjustedLog10RunningTotal);

                // Update
                m_maxWeightLog10 = valueLog10;
                m_maxPossibleFinalWeightLog10 = newMaxPossibleFinalWeightLog10;
                m_offset = newOffset;

                m_runningTotal += std::pow(10, (valueLog10 + m_offset));
                m_numValues++;
            }

            boost::multiprecision::cpp_dec_float_100 ExtractFinalValue()
            {
                if (m_numValues == 0)
                {
                    return 0.0;
                }

                boost::multiprecision::cpp_dec_float_100 runningTotalLog10 = std::log10(m_runningTotal);
                runningTotalLog10 -= m_offset;
                return boost::multiprecision::pow(10.0, runningTotalLog10);
            }

        private:
            uint32_t m_numValues = 0;
            double m_runningTotal = 0.0;
            double m_offset = 0.0;
            double m_maxWeightLog10 = 0.0;
            double m_maxPossibleFinalWeightLog10 = 0.0;
        };


    public:

        /**
         * @brief Construct a new Experiment Runner Cpu object
         *
         * @param bootstrapper Bootstrapper object responsible for generating an initial curve given the experiment constraints
         */
        FullExperimentRunnerOptimalPerturbOptimized(ExperimentRunner::ExperimentParameters& experimentParams, Bootstrapper& bootstrapper, uint32_t numCombinedValuesForAvg, uint32_t numCombinedValuesForMax);
        virtual ~FullExperimentRunnerOptimalPerturbOptimized();

        virtual bool Setup() override;
        virtual ExperimentResults RunExperiment() override;
        virtual void Shutdown() override;

    private:
        /*
            flag parameter is set to 0 for no errors.
            If the root solve fails, its set to 1
            If the root solve succeeds, but we dont accept the path due to calculated path error, we set to 2
        */

        void GeometryPerturb(
            int64_t threadIdx,
            int64_t numCombinedWeightValuesTotal,
            int64_t numCombinedWeightValuesPerThread,
            int64_t numPathsToSkipPerThread,
            int64_t numSegmentsPerCurve,
            std::vector<std::mt19937_64>& rngGenerators,
            std::vector<Farlor::Vector3>& initialCurvePos,
            std::vector<Farlor::Vector3>& initialCurveTans,
            std::vector<float>& initialCurveCurvatures,
            std::vector<Farlor::Vector3>& globalPos,
            std::vector<Farlor::Vector3>& globalTans,
            std::vector<float>& globalCurvatures,
            std::vector<Farlor::Vector3>& scratchPositionSpaceLeft,
            std::vector<Farlor::Vector3>& scratchTangentSpaceLeft,
            std::vector<float>& scratchCurvatureSpaceLeft,
            std::vector<Farlor::Vector3>& scratchPositionSpaceRight,
            std::vector<Farlor::Vector3>& scratchTangentSpaceRight,
            std::vector<float>& scratchCurvatureSpaceRight,
            std::vector<CombinedWeightValues>& combinedWeightValues,
            std::vector<double>& cachedSegmentWeights,
            float segmentLength,
            const twisty::PathWeighting::WeightLookupTableIntegral& weightingIntegral,
            const twisty::PerturbUtils::BoundaryConditions& boundaryConditions,
            const PathWeighting::NormalizerStuff::BaseNormalizer& fn
        );

    private:
        uint32_t m_numCombinedValuesForAvg = 1;
        uint32_t m_numCombinedValuesForMax = 1;
    };
}