#include "CurvePurturber.h"

#include "CurveUtils.h"

#include <assert.h>
#include <random>
#include <cmath>
#include <ctime>

namespace twisty
{
    CurvePuturber::CurvePuturber()
    {
    }

    std::unique_ptr<Curve> CurvePuturber::GetCurvePutrubation(const Curve& curve, const Range& kdsRange, const Range& tdsRange)
    {
        assert(false);
        return nullptr;
        ////TODO: Figure out how to pull all rng out into a single place which we can then use for repeating experiments
        //std::mt19937 rng(time(0));

        //// We want to copy over the curve
        //std::unique_ptr<Curve> upGeneratedCurve = std::make_unique<Curve>(curve.m_numSegments);
        //(*upGeneratedCurve) = curve;

        //// Three generators are used for the three index ranges
        //const uint32_t secondFromLeftIndex = 1;
        //std::uniform_int_distribution<int> midGen(secondFromLeftIndex, upGeneratedCurve->m_numSegments - 2); // uniform, unbiased
        //int32_t middleIndex = midGen(rng);
        //std::uniform_int_distribution<int> leftGen(0, middleIndex - 1);                           // uniform, unbiased
        //std::uniform_int_distribution<int> rightGen(middleIndex + 1, upGeneratedCurve->m_numSegments - 1); // uniform, unbiased
        //int32_t leftIndex = leftGen(rng);
        //int32_t rightIndex = rightGen(rng);

        //assert(leftIndex < middleIndex);
        //assert(middleIndex < rightIndex);

        //// Modify the curvature of the middle selection
        //std::uniform_real_distribution<float> curvatureGen(kdsRange.m_min, kdsRange.m_max);
        //std::uniform_real_distribution<float> torsionGen(tdsRange.m_min, tdsRange.m_max);

        //Farlor::Vector3 targetP = curve.m_targetPos;
        //Farlor::Vector3 targetN = curve.m_targetTangent;

        //CurveUtils::CurveState initialState = CurveUtils::RetrieveStateFromCurve(*upGeneratedCurve, leftIndex, middleIndex, rightIndex);

        //// Update with new middle curvature value
        //CurveUtils::CurveState currentState = initialState;
        //currentState.k2 = curvatureGen(rng);
        //currentState.k1 = curvatureGen(rng);
        //currentState.k3 = curvatureGen(rng);
        //currentState.t1 = torsionGen(rng);
        //currentState.t2 = torsionGen(rng);
        //currentState.t3 = torsionGen(rng);

        //CurveUtils::UpdateCurveWithState(*upGeneratedCurve, currentState, leftIndex, middleIndex, rightIndex);
        //return upGeneratedCurve;
    }

    std::unique_ptr<Curve> GetValidCurvePurtubation(Curve &curve, uint32_t numAttempts)
    {
        return nullptr;
        // //TODO: Figure out how to pull all rng out into a single place which we can then use for repeating experiments
        // std::mt19937 rng(time(0));

        // // We want to copy over the curve
        // std::unique_ptr<Curve> upGeneratedCurve = std::make_unique<Curve>(curve.m_numSegments);
        // (*upGeneratedCurve) = curve;

        // // Three generators are used for the three index ranges
        // const uint32_t secondFromLeftIndex = 1;
        // std::uniform_int_distribution<int> midGen(secondFromLeftIndex, upGeneratedCurve->m_numSegments - 2); // uniform, unbiased
        // int32_t middleIndex = midGen(rng);
        // std::uniform_int_distribution<int> leftGen(0, middleIndex - 1);                           // uniform, unbiased
        // std::uniform_int_distribution<int> rightGen(middleIndex + 1, newCurve.m_numSegments - 1); // uniform, unbiased
        // int32_t leftIndex = leftGen(rng);
        // int32_t rightIndex = rightGen(rng);

        // assert(leftIndex < middleIndex);
        // assert(middleIndex < rightIndex);

        // // Modify the curvature of the middle selection
        // std::uniform_real_distribution<float> curvatureGen(m_kRange.m_min, m_kRange.m_max);
        // std::uniform_real_distribution<float> torsionGen(m_tRange.m_min, m_tRange.m_max);

        // Farlor::Vector3 targetP = curve.m_targetPos;
        // Farlor::Vector3 targetN = curve.m_targetTangent;

        // CurveUtils::CurveState initialState = CurveUtils::RetrieveStateFromCurve(newCurve, leftIndex, middleIndex, rightIndex);

        // // Update with new middle curvature value
        // CurveUtils::CurveState currentState = initialState;
        // currentState.k2 = curvatureGen(rng);

        // CurveUtils::UpdateCurveWithState(newCurve, currentState, leftIndex, middleIndex, rightIndex);

        // // Calculate error of new curvature
        // float prevError = CurveUtils::CalculateCurveError(newCurve);

        // const uint32_t numIterations = 100;
        // // A little confused as to how we are going to discover this threshold
        // const float acceptance = 2.0f;
        // for (uint32_t i = 0; i < numIterations; ++i)
        // {
        //     CurveUtils::CurveState guessState = currentState;
        //     guessState.k1 = curvatureGen(rng);
        //     guessState.k3 = curvatureGen(rng);
        //     guessState.t1 = torsionGen(rng);
        //     guessState.t2 = torsionGen(rng);
        //     guessState.t3 = torsionGen(rng);

        //     CurveUtils::UpdateCurveWithState(newCurve, guessState, leftIndex, middleIndex, rightIndex);
        //     float newError = CurveUtils::CalculateCurveError(newCurve);

        //     if (newError < prevError)
        //     {
        //         currentState = guessState;
        //         prevError = newError;
        //     }

        //     // No need to continue, we have acceptance
        //     // Early out
        //     if (prevError <= acceptance)
        //     {
        //         break;
        //     }
        // }

        // // If we have too much error, we dont accept
        // if (prevError > acceptance)
        // {
        //     // We failed to achieve the acceptance criteria in the specified number of iterations
        //     std::cout << "\tFailed Puturb: " << prevError << std::endl;
        //     return {};
        // }

        // std::cout << "\tSuccessful path: " << prevError << std::endl;

        // return newCurve;
    }
}