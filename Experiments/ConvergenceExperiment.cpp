#include <cstdint>

#include <SpecifiedCurveParamBootstrapper.h>
#include <ExperimentRunnerCpu.h>


int main()
{
    const float a = 0.0f;
    const float b = 0.0f;
    const float k_i = 0.1f;
    const float t_i = 0.1f;
    const uint32_t numSegments = 200;
    const float arclength = 16.0f;
    const float gaussianWidth = 0.1f;
    const float regConstant = 0.5f;

    twisty::Range arclengthRange;
    arclengthRange.m_min = arclength;
    arclengthRange.m_max = arclength;

    twisty::SpecifiedCurveParamBootstrapper specifiedBootstrapper(k_i, t_i, arclengthRange);

    // Test the CPU experiment
    twisty::Range kRange;
    kRange.m_min = -10.0f;
    kRange.m_max = -10.0f;

    twisty::Range tRange;
    tRange.m_min = -10.0f;
    tRange.m_max = -10.0f;

    twisty::ExperimentRunnerCpu cpuExperimentRunner(specifiedBootstrapper, kRange, tRange);
    cpuExperimentRunner.Setup(numSegments);

    twisty::ExperimentRunner::ExperimentParameters params;
    params.exportGeneratedCurves = true;
    params.numCurvesToDiscard = 0;
    //params.numTotalCurves = 10000000;
    params.numTotalCurves = 100;
    twisty::ExperimentRunner::ExperimentResults results = cpuExperimentRunner.RunExperiment(params);
}