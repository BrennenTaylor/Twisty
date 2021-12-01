#include "FullExperimentRunnerOptimalPerturb.h"

#include "CurvePerturbUtils.h"
#include "CurveUtils.h"
#include "MathConsts.h"


#include <assert.h>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <limits>

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdlib.h>
#include <memory>

//#define DetailedPurturb
//#define SingleThreadMode

const double AmountOfFullRotation = 1.0;

//#define SINGLE_THREAD_PERTURB_MODE
#define OutputBigFloatPathWeights

#define ExportPathBatches

#if defined(ExportPathBatches)

const static int64_t ExportPathBatchCacheSize = 100000;

static std::mutex ExportPathBatchesMutex;
static std::ofstream curvesBinaryFile;
static std::ofstream curvesMetadataFile;

#else

const int64_t ExportPathBatchCacheSize = 1000000;

#endif

namespace twisty
{
        // Assumes pVector3f is an array of 3 floats
    static void NormalizeVector3f(float* pVector3f)
    {
        float normalizer = pVector3f[0] * pVector3f[0] + pVector3f[1] * pVector3f[1] + pVector3f[2] * pVector3f[2];
        normalizer = 1.0 / sqrt(normalizer);
        pVector3f[0] *= normalizer;
        pVector3f[1] *= normalizer;
        pVector3f[2] *= normalizer;
    }

    // This has an out parameter
    static void RotationMatrixAroundAxis(float angle, float* pAxisVector3f, float* pMatrix3x3)
    {
        // Ensure its normalized
        NormalizeVector3f(pAxisVector3f);

        pMatrix3x3[0] = cos(angle) + pAxisVector3f[0] * pAxisVector3f[0] * (1.0f - cos(angle));
        pMatrix3x3[1] = pAxisVector3f[0] * pAxisVector3f[1] * (1.0f - cos(angle)) - pAxisVector3f[2] * sin(angle);
        pMatrix3x3[2] = pAxisVector3f[0] * pAxisVector3f[2] * (1.0f - cos(angle)) + pAxisVector3f[1] * sin(angle);

        pMatrix3x3[3] = pAxisVector3f[1] * pAxisVector3f[0] * (1.0f - cos(angle)) + pAxisVector3f[2] * sin(angle);
        pMatrix3x3[4] = cos(angle) + pAxisVector3f[1] * pAxisVector3f[1] * (1 - cos(angle));
        pMatrix3x3[5] = pAxisVector3f[1] * pAxisVector3f[2] * (1 - cos(angle)) - pAxisVector3f[0] * sin(angle);

        pMatrix3x3[6] = pAxisVector3f[2] * pAxisVector3f[0] * (1 - cos(angle)) - pAxisVector3f[1] * sin(angle);
        pMatrix3x3[7] = pAxisVector3f[2] * pAxisVector3f[1] * (1 - cos(angle)) + pAxisVector3f[0] * sin(angle);
        pMatrix3x3[8] = cos(angle) + pAxisVector3f[2] * pAxisVector3f[2] * (1 - cos(angle));
    }

    static float DotVector3fVector3f(float* lhs, float* rhs)
    {
        return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
    }

    static float MagVector3f(float* pVec)
    {
        return sqrt(pVec[0] * pVec[0] + pVec[1] * pVec[1] + pVec[2] * pVec[2]);
    }

    static void RotateVectorByMatrix(float* pRotationMatrix, float* pVector)
    {
        float val[3];
        val[0] = DotVector3fVector3f(pRotationMatrix, pVector);
        val[1] = DotVector3fVector3f(pRotationMatrix + 3, pVector);
        val[2] = DotVector3fVector3f(pRotationMatrix + 6, pVector);
        
        // Write it back to pVector
        pVector[0] = val[0];
        pVector[1] = val[1];
        pVector[2] = val[2];
    }

    FullExperimentRunnerOptimalPerturb::FullExperimentRunnerOptimalPerturb(ExperimentRunner::ExperimentParameters& experimentParams,
        Bootstrapper& bootstrapper)
        : ExperimentRunner(experimentParams, bootstrapper)
    {
    }

    FullExperimentRunnerOptimalPerturb::~FullExperimentRunnerOptimalPerturb()
    {
    }

    std::optional<ExperimentRunner::ExperimentResults> FullExperimentRunnerOptimalPerturb::RunExperiment()
    {
        auto runExperimentTimeStart = std::chrono::high_resolution_clock::now();

        /* --------------------- */
        auto setupTimeStart = std::chrono::high_resolution_clock::now();

        std::cout << "Random Seeds: " << std::endl;
        std::cout << "\tBootstrap seed: " << m_experimentParams.bootstrapSeed << std::endl;
        std::cout << "\tPerturb seed: " << m_experimentParams.curvePurturbSeed << std::endl;

        // Ask the bootstrapper to generate a discrete curve.
        // If we fail, we want to exit the experiment.
        bool successfulGen = false;
        while (!successfulGen)
        {
            m_upInitialCurve = m_bootstrapper.CreateCurveGeometricSafe(m_experimentParams.numSegmentsPerCurve, m_experimentParams.arclength);
            if (!m_upInitialCurve)
            {
                printf("Both bootstrap versions failed, now we have to error out.\n");
                return {};
            }

            // Lets also get the error of the initial curve, just to know
            float curveError = CurveUtils::CalculateCurveError(*m_upInitialCurve);
            std::cout << "Seed curve error: " << curveError << std::endl;

            if (curveError < m_experimentParams.maximumBootstrapCurveError)
            {
                successfulGen = true;
            }
        }

        const std::filesystem::path experimentDirPath = m_experimentParams.experimentDirPath;
        if (!std::filesystem::exists(experimentDirPath))
        {
            std::filesystem::create_directories(experimentDirPath);
        }

#if defined(ExportPathBatches)
        {
            BeginPathBatchOutput();

            std::filesystem::path generatedCurvesDirPath = m_experimentParams.experimentDirPath;
            generatedCurvesDirPath /= m_experimentParams.perExperimentDirSubfolder;
            generatedCurvesDirPath /= "GeneratedCurves";
            if (!std::filesystem::exists(generatedCurvesDirPath))
            {
                std::filesystem::create_directories(generatedCurvesDirPath);
            }

            std::stringstream pathBinaryFilenameSS;
            pathBinaryFilenameSS << m_experimentParams.pathBatchPrepend;
            pathBinaryFilenameSS << "Paths_Binary" << ".pbd";

            std::filesystem::path binaryFilePath = generatedCurvesDirPath;
            binaryFilePath.append(pathBinaryFilenameSS.str());
            curvesBinaryFile.open(binaryFilePath, std::ios::binary);

            std::stringstream pathMetadataFilenameSS;
            pathMetadataFilenameSS << m_experimentParams.pathBatchPrepend;
            pathMetadataFilenameSS << "Paths_Metadata" << ".pmd";

            std::filesystem::path metadataFilePath = generatedCurvesDirPath;
            metadataFilePath.append(pathMetadataFilenameSS.str());
            curvesMetadataFile.open(metadataFilePath);
        }
#endif

        // Say that we will start outputing the path batch output
        const double ds = m_upInitialCurve->m_arclength / m_experimentParams.numSegmentsPerCurve;

        std::vector<std::unique_ptr<twisty::PathWeighting::WeightLookupTableIntegral>> lookupEvaluators(m_experimentParams.weightingParameters.scatterValues.size());
        for (int scatterIdx = 0; scatterIdx < m_experimentParams.weightingParameters.scatterValues.size(); scatterIdx++)
        {
            twisty::WeightingParameters updatedWeightingParams = m_experimentParams.weightingParameters;
            updatedWeightingParams.scatter = m_experimentParams.weightingParameters.scatterValues[scatterIdx];
            lookupEvaluators[scatterIdx] = std::make_unique<twisty::PathWeighting::WeightLookupTableIntegral>(updatedWeightingParams, ds);

            std::filesystem::path exportTableDir = m_experimentParams.experimentDirPath;
            exportTableDir /= m_experimentParams.perExperimentDirSubfolder;
            exportTableDir /= std::to_string(scatterIdx);
            lookupEvaluators[scatterIdx]->ExportValues(exportTableDir.string());
        }


        //twisty::PathWeighting::WeightLookupTableIntegral lookupEvaluator(m_experimentParams.weightingParameters, ds);
        
        twisty::PerturbUtils::BoundaryConditions boundaryConditions;
        boundaryConditions.arclength = m_upInitialCurve->m_arclength;
        boundaryConditions.m_startPos = m_upInitialCurve->m_basePos;
        boundaryConditions.m_startDir = m_upInitialCurve->m_baseTangent;
        boundaryConditions.m_endPos = m_upInitialCurve->m_targetPos;
        boundaryConditions.m_endDir = m_upInitialCurve->m_targetTangent;
        
        // Constants
        double minCurvature = 0.0;
        double maxCurvature = 0.0;
        twisty::PathWeighting::CalcMinMaxCurvature(m_experimentParams.weightingParameters, minCurvature, maxCurvature, ds);
        const float curvatureStepSize = (maxCurvature - minCurvature) / m_experimentParams.weightingParameters.numCurvatureSteps;

        // This is a horible hack to make sure we always purturb a new curve
        Curve curveToBend = *m_upInitialCurve;

        auto experimentTimeStart = std::chrono::high_resolution_clock::now();

        // Create threads and dispatch them
// #ifdef SINGLE_THREAD_PERTURB_MODE
//         int64_t numPurturbThreads = 1;
// #else
        int64_t numPurturbThreads = std::thread::hardware_concurrency();
// #endif
        std::cout << "We have " << numPurturbThreads << " avalible for purturbation." << std::endl;

        // Setup rng stuff
        std::vector<std::mt19937_64> perThreadRngGenerators(numPurturbThreads);
        int64_t seed = m_experimentParams.curvePurturbSeed;
        if (seed == 0)
        {
            seed = time(0);
        }
        for (int64_t i = 0; i < numPurturbThreads; ++i)
        {
            perThreadRngGenerators[i] = std::mt19937_64(seed + i);
        }

        // Setup data structures
        std::vector<Farlor::Vector3> initialCurvePositions(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<Farlor::Vector3> initialCurveTangents(m_experimentParams.numSegmentsPerCurve + 1);
        std::vector<float> initialCurveCurvatures(m_experimentParams.numSegmentsPerCurve);

        // Positions
        // Hard code the first two positions
        initialCurvePositions[0] = m_upInitialCurve->m_basePos;
        initialCurvePositions[1] = m_upInitialCurve->m_basePos + m_upInitialCurve->m_baseTangent.Normalized() * m_upInitialCurve->m_segmentLength;
        for (int64_t segmentIdx = 2; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
        {
            initialCurvePositions[segmentIdx] = m_upInitialCurve->m_positions[segmentIdx];
        }
        // Hard code the final position
        initialCurvePositions[m_experimentParams.numSegmentsPerCurve] = m_upInitialCurve->m_targetPos;

#if defined(DetailedPurturb)
        {
            std::cout << "Positions" << std::endl;
            for (int64_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurvePositions[segmentIdx] << std::endl;
            }
        }
#endif
        twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(initialCurvePositions.data(), initialCurveTangents.data(),
            initialCurveCurvatures.data(), m_upInitialCurve->m_numSegments, boundaryConditions);

#if defined(DetailedPurturb)
        {
            std::cout << "Tangents" << std::endl;
            for (int64_t tanIdx = 0; tanIdx < m_experimentParams.numSegmentsPerCurve; ++tanIdx)
            {
                std::cout << "\t" << initialCurveTangents[tanIdx] << std::endl;
            }

            std::cout << "Curvatures" << std::endl;
            for (int64_t segmentIdx = 0; segmentIdx < m_experimentParams.numSegmentsPerCurve; ++segmentIdx)
            {
                std::cout << "\t" << initialCurveCurvatures[segmentIdx] << std::endl;
            }
        }
#endif

        const int64_t NumPosPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumTanPerCurve = (m_upInitialCurve->m_numSegments + 1);
        const int64_t NumCurvaturePerCurve = (m_upInitialCurve->m_numSegments);

        std::vector<Farlor::Vector3> perThreadCurvePositions(NumPosPerCurve * numPurturbThreads);
        std::vector<Farlor::Vector3> perThreadCurveTangents(NumTanPerCurve * numPurturbThreads);
        std::vector<float> perThreadCurveCurvatures(NumCurvaturePerCurve * numPurturbThreads);

        std::vector<Farlor::Vector3> perThreadPositionScratchLeft(NumPosPerCurve * numPurturbThreads);
        std::vector<Farlor::Vector3> perThreadTangentScratchLeft(NumTanPerCurve * numPurturbThreads);
        std::vector<float> perThreadCurvatureScratchLeft(NumCurvaturePerCurve * numPurturbThreads);

        std::vector<Farlor::Vector3> perThreadPositionScratchRight(NumPosPerCurve * numPurturbThreads);
        std::vector<Farlor::Vector3> perThreadTangentScratchRight(NumTanPerCurve * numPurturbThreads);
        std::vector<float> perThreadCurvatureScratchRight(NumCurvaturePerCurve * numPurturbThreads);

        for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
        {
            // Copy Pos
            for (int64_t posIdx = 0; posIdx < NumPosPerCurve; posIdx++)
            {
                perThreadCurvePositions[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
                perThreadPositionScratchLeft[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
                perThreadPositionScratchRight[NumPosPerCurve * threadIdx + posIdx] = initialCurvePositions[posIdx];
            }

            // Copy Tan
            for (int64_t tanIdx = 0; tanIdx < NumTanPerCurve; tanIdx++)
            {
                perThreadCurveTangents[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
                perThreadTangentScratchLeft[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
                perThreadTangentScratchRight[NumTanPerCurve * threadIdx + tanIdx] = initialCurveTangents[tanIdx];
            }

            // Copy Curvatures
            for (int64_t curvatureIdx = 0; curvatureIdx < NumCurvaturePerCurve; curvatureIdx++)
            {
                perThreadCurveCurvatures[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
                perThreadCurvatureScratchLeft[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
                perThreadCurvatureScratchRight[NumCurvaturePerCurve * threadIdx + curvatureIdx] = initialCurveCurvatures[curvatureIdx];
            }
        }

        std::vector<double> cachedSegmentWeights(m_experimentParams.numSegmentsPerCurve * numPurturbThreads);
        std::vector<double> compressedWeightBuffer(ExportPathBatchCacheSize * m_experimentParams.weightingParameters.scatterValues.size());

        std::stringstream fnFilenameSS;
        fnFilenameSS << "SavedFN";
        fnFilenameSS << m_experimentParams.numSegmentsPerCurve;
        fnFilenameSS << ".fnd";
        const std::filesystem::path fnFilePath = std::filesystem::current_path() / fnFilenameSS.str();
        std::unique_ptr<PathWeighting::NormalizerStuff::BaseNormalizer> upFN = nullptr;

        //if (m_experimentParams.pathNormalizerType == PathNormalizerType::Default)
        //{
        //    upFN = std::make_unique<PathWeighting::NormalizerStuff::BaseNormalizer>();
        //}
        //else
        // We dont need this actually, so we can just load the default one
        {
            // If we can load the fn data, load it
            if (std::filesystem::exists(fnFilePath))
            {
                std::cout << "Using cached fd file at: " << fnFilePath << std::endl;
                std::ifstream inFile(fnFilePath);
                upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(inFile);
                inFile.close();
            }
            // We need to generate it this time and save it off to use next time
            else
            {
                // This is the max M value
                const int maxorder = m_upInitialCurve->m_numSegments;

                // Generate the fn data table
                const int numZSamples = 5000;
                const int numIntegrationSamples = 5000;

                // Arbitrarily set min and max |r_vec| values.
                // Why this specific max bound, I do not know
                const double rMin = 0.0;
                const double rMax = 200.0;
                upFN = std::make_unique<PathWeighting::NormalizerStuff::FN>(numZSamples, numIntegrationSamples, maxorder, rMin, rMax);

                std::ofstream outFile(fnFilePath);
                dynamic_cast<PathWeighting::NormalizerStuff::FN*>(upFN.get())->WriteToFile(outFile);
                outFile.close();
            }
        }
        PathWeighting::NormalizerStuff::BaseNormalizer& fn = (*upFN);

        // Why the 1/(delta s) = (M+2)/s?
        Farlor::Vector3 Z = (boundaryConditions.m_endPos - boundaryConditions.m_startPos) * (m_upInitialCurve->m_numSegments + 2) / boundaryConditions.arclength
            - boundaryConditions.m_endDir - boundaryConditions.m_startDir;
        std::cout << "Z: " << Z << std::endl;
        std::cout << "|Z|: " << Z.Magnitude() << std::endl;

        PathWeighting::NormalizerStuff::NormalizerDoubleType pathNormalizer = PathWeighting::NormalizerStuff::Norm(fn, m_upInitialCurve->m_numSegments,
            Z.Magnitude(), boundaryConditions.arclength);

        const boost::multiprecision::cpp_dec_float_100 pathNormalizerLog10 = boost::multiprecision::log10(pathNormalizer);

        std::cout << "PathNormalizer: " << pathNormalizer << std::endl;
        std::cout << "PathNormalizerLog10: " << pathNormalizerLog10 << std::endl;
        //exit(1);

        auto setupTimeEnd = std::chrono::high_resolution_clock::now();
        /* --------------------- */


        /* --------------------- */
        const int64_t numDispatches = (m_experimentParams.numPathsInExperiment + ExportPathBatchCacheSize - 1) / ExportPathBatchCacheSize;
        std::cout << "numPathsInExperiment: " << m_experimentParams.numPathsInExperiment << std::endl;
        std::cout << "numPathsPerBatch: " << ExportPathBatchCacheSize << std::endl;
        std::cout << "Num dispatches required: " << numDispatches << std::endl;
        int64_t numPathsLeft = m_experimentParams.numPathsInExperiment;
        int64_t numPathsGenerated = 0;

        // We need to calculate the absorbtion/scattering piece
        std::vector<boost::multiprecision::cpp_dec_float_100> bigTotalExperimentWeights(lookupEvaluators.size());
        for (int idx = 0; idx < lookupEvaluators.size(); idx++)
        {
            bigTotalExperimentWeights[idx] = 0.0;
        }

        long long perturbTimeCount = 0;
        long long weightCalcTimeCount = 0;

        // We need a number of dispatches
        boost::multiprecision::cpp_dec_float_100 minimumPathWeight = 0.0;
        boost::multiprecision::cpp_dec_float_100 maximumPathWeight = 0.0;


        std::vector<twisty::PathWeighting::WeightLookupTableIntegral*> weightingIntegralsRawPointers(lookupEvaluators.size());
        for (int idx = 0; idx < lookupEvaluators.size(); idx++)
        {
            weightingIntegralsRawPointers[idx] = lookupEvaluators[idx].get();
        }

        for (int64_t dispatchIdx = 0; dispatchIdx < numDispatches; ++dispatchIdx)
        {
            auto dispatchTimeStart = std::chrono::high_resolution_clock::now();

            auto perturbTimeStart = std::chrono::high_resolution_clock::now();

            int64_t pathsInDispatch = std::min(ExportPathBatchCacheSize, numPathsLeft);
            std::cout << "Paths in dispatch " << dispatchIdx << ": " << pathsInDispatch << std::endl;
            {
                int64_t numPathsPerThread = (pathsInDispatch + numPurturbThreads - 1) / numPurturbThreads;
                std::vector<std::thread> threads(numPurturbThreads);
                for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
                {
                    std::thread newThread(&FullExperimentRunnerOptimalPerturb::GeometryPerturb, this,
                        threadIdx,
                        pathsInDispatch,
                        numPathsPerThread,
                        m_experimentParams.numPathsToSkip,
                        m_experimentParams.numSegmentsPerCurve,
                        std::ref(perThreadRngGenerators),
                        std::ref(perThreadCurvePositions),
                        std::ref(perThreadCurveTangents),
                        std::ref(perThreadCurveCurvatures),
                        std::ref(perThreadPositionScratchLeft),
                        std::ref(perThreadTangentScratchLeft),
                        std::ref(perThreadCurvatureScratchLeft),
                        std::ref(perThreadPositionScratchRight),
                        std::ref(perThreadTangentScratchRight),
                        std::ref(perThreadCurvatureScratchRight),
                        std::ref(compressedWeightBuffer),
                        std::ref(cachedSegmentWeights),
                        m_upInitialCurve->m_segmentLength,
                        weightingIntegralsRawPointers,
                        boundaryConditions,
                        fn
                    );

                    threads[threadIdx] = std::move(newThread);
                }

                for (int64_t threadIdx = 0; threadIdx < numPurturbThreads; ++threadIdx)
                {
                    if (threads[threadIdx].joinable())
                    {
                        threads[threadIdx].join();
                    }
                }
            }

            auto perturbTimeEnd = std::chrono::high_resolution_clock::now();
            perturbTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(perturbTimeEnd - perturbTimeStart).count();

            // -------------------
            auto weightingTimeStart = std::chrono::high_resolution_clock::now();

            std::vector<boost::multiprecision::cpp_dec_float_100> bigFloatWeightsLog10(pathsInDispatch);
            int64_t numWeightingThreads = std::thread::hardware_concurrency();

            std::vector<boost::multiprecision::cpp_dec_float_100> totalDispatchWeights(lookupEvaluators.size());
            for (int64_t idx = 0; idx < lookupEvaluators.size(); ++idx)
            {
                totalDispatchWeights[idx] = 0.0;
            }


            int64_t numWeightsPerThread = (pathsInDispatch + numWeightingThreads - 1) / numWeightingThreads;
            {
                std::vector<std::thread> threads(numWeightingThreads);
                std::vector<boost::multiprecision::cpp_dec_float_100> threadWeights(numWeightingThreads * lookupEvaluators.size());
                for (int64_t idx = 0; idx < numWeightingThreads * lookupEvaluators.size(); ++idx)
                {
                    threadWeights[idx] = 0.0;
                }
                
                for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    std::thread newThread(&FullExperimentRunnerOptimalPerturb::WeightCombineThreadKernel, this, threadIdx, pathsInDispatch, numWeightsPerThread, lookupEvaluators.size(), m_upInitialCurve->m_arclength,
                        m_upInitialCurve->m_numSegments, std::ref(compressedWeightBuffer), std::ref(bigFloatWeightsLog10), std::ref(threadWeights),
                        pathNormalizer);
                    threads[threadIdx] = std::move(newThread);
                }

                for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    if (threads[threadIdx].joinable())
                    {
                        threads[threadIdx].join();
                    }
                }

#if defined(DetailedPurturb)
                {
                    std::cout << "Thread weights: " << std::endl;
                    for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                    {
                        std::cout << threadWeights[threadIdx] << std::endl;
                    }
                }
#endif

                for (int64_t threadIdx = 0; threadIdx < numWeightingThreads; ++threadIdx)
                {
                    for (int64_t scatterIdx = 0; scatterIdx < lookupEvaluators.size(); ++scatterIdx)
                    {
                        totalDispatchWeights[scatterIdx] += threadWeights[threadIdx * lookupEvaluators.size() + scatterIdx];
                    }
                }
            }

            for (int64_t scatterIdx = 0; scatterIdx < lookupEvaluators.size(); ++scatterIdx)
            {
                std::cout << "Dispatch " << dispatchIdx << " Scatter Idx " << scatterIdx << " Weight: " << totalDispatchWeights[scatterIdx] << std::endl;
                bigTotalExperimentWeights[scatterIdx] += totalDispatchWeights[scatterIdx];
            }

            auto weightingTimeEnd = std::chrono::high_resolution_clock::now();
            weightCalcTimeCount += std::chrono::duration_cast<std::chrono::milliseconds>(weightingTimeEnd - weightingTimeStart).count();
            /* --------------------- */

#ifdef OutputBigFloatPathWeights
            std::filesystem::path weightDirectoryPath = m_pathBatchOutputPath;
            std::string bigfloatOutputFile = "BigFloatWeights.txt";
            weightDirectoryPath.append(bigfloatOutputFile);

            std::cout << "Output: " << dispatchIdx << " : " << bigFloatWeightsLog10.size() << std::endl;

            std::ofstream bigfloatOFS;
            if (dispatchIdx == 0)
            {
                bigfloatOFS.open(weightDirectoryPath.c_str());
                bigfloatOFS << m_experimentParams.numPathsInExperiment << std::endl;
            }
            else
            {
                bigfloatOFS.open(weightDirectoryPath.c_str(), std::ios::app);
            }
            
            for (int64_t i = 0; i < bigFloatWeightsLog10.size(); ++i)
            {
                // We have to add in the path normalizer here as it wasnt acounted for anywhere else before this for these specific saved off weights
                bigfloatOFS << (bigFloatWeightsLog10[i] + pathNormalizerLog10) << std::endl;
            }
#endif
            numPathsLeft -= pathsInDispatch;
            numPathsGenerated += pathsInDispatch;

            auto dispatchTimeEnd = std::chrono::high_resolution_clock::now();
            auto dispatchRunTime = std::chrono::duration_cast<std::chrono::milliseconds>(dispatchTimeEnd - dispatchTimeStart);
            std::cout << "\tDispatch " << dispatchIdx  << " Time: " << dispatchRunTime.count() << "ms" << std::endl;
        }

        std::cout << "Minimum Weight: " << minimumPathWeight << std::endl;
        std::cout << "Maximum Weight: " << maximumPathWeight << std::endl;
        //std::cout << "Big total weight before: " << bigTotalExperimentWeight << std::endl;

        auto runExperimentTimeEnd = std::chrono::high_resolution_clock::now();

        std::cout << "Experiment Time Reporting: " << std::endl;
        auto runExperimentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(runExperimentTimeEnd - runExperimentTimeStart);
        std::cout << "\tTotal Experiment Time: " << runExperimentTimeMs.count() << "ms" << std::endl;

        {
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupTimeEnd - setupTimeStart);
            std::cout << "\tsetup Time: " << timeMs.count() << "ms - " << ((float)timeMs.count() / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tperturb Time: " << perturbTimeCount << "ms - " << ((float)perturbTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

        {
            std::cout << "\tweighting Time: " << weightCalcTimeCount << "ms - " << ((float)weightCalcTimeCount / (float)runExperimentTimeMs.count()) * 100.0f << "%" << std::endl;
        }

#if defined(ExportPathBatches)
        {
        EndPathBatchOutput();

        curvesBinaryFile.close();
        curvesMetadataFile.close();
        }
#endif

        ExperimentResults results;
        results.experimentWeights = bigTotalExperimentWeights;
        results.totalPathsGenerated = numPathsGenerated;
        results.numFailedPaths = 0;
        return results;
    }

    void FullExperimentRunnerOptimalPerturb::WeightCombineThreadKernel(const int64_t threadIdx, int64_t numWeights, int64_t numWeightsPerThread,
        int numLookupEvaluators, float arclength, int64_t numSegmentsPerCurve,
        const std::vector<double>& compressedWeights, std::vector<boost::multiprecision::cpp_dec_float_100>& bigFloatWeightsLog10,
        std::vector<boost::multiprecision::cpp_dec_float_100>& threadScatterWeights, boost::multiprecision::cpp_dec_float_100 pathNormalizer)
    {
        for (int64_t i = 0; i < numWeightsPerThread; i++)
        {
            int64_t idx = threadIdx * numWeightsPerThread + i;
            if (idx >= numWeights)
            {
                break;
            }

            int64_t actualIdx = threadIdx * numWeightsPerThread * numLookupEvaluators + i * numLookupEvaluators;

            for (int scatterIdx = 0; scatterIdx < numLookupEvaluators; scatterIdx++)
            {
                const boost::multiprecision::cpp_dec_float_100 bigfloatCompressed = compressedWeights[actualIdx + scatterIdx];
                const boost::multiprecision::cpp_dec_float_100 decompressed = boost::multiprecision::pow(10.0, bigfloatCompressed);
                const boost::multiprecision::cpp_dec_float_100 pathWeight = decompressed * pathNormalizer;

                // Pulled from Jerry analysis
                bigFloatWeightsLog10[idx] = bigfloatCompressed;
                threadScatterWeights[threadIdx * numLookupEvaluators + scatterIdx] += pathWeight;
            }
        }
    }




    void FullExperimentRunnerOptimalPerturb::GeometryPerturb(
        int64_t threadIdx,
        int64_t numExperimentPaths,
        int64_t numPathsPerThread,
        int64_t numPathsToSkipPerThread,
        int64_t numSegmentsPerCurve,
        std::vector<std::mt19937_64>& rngGenerators,
        std::vector<Farlor::Vector3>& globalPos,
        std::vector<Farlor::Vector3>& globalTans,
        std::vector<float>& globalCurvatures,
        std::vector<Farlor::Vector3>& scratchPositionSpaceLeft,
        std::vector<Farlor::Vector3>& scratchTangentSpaceLeft,
        std::vector<float>& scratchCurvatureSpaceLeft,
        std::vector<Farlor::Vector3>& scratchPositionSpaceRight,
        std::vector<Farlor::Vector3>& scratchTangentSpaceRight,
        std::vector<float>& scratchCurvatureSpaceRight,
        std::vector<double>& globalPathWeights,
        std::vector<double>& cachedSegmentWeights,
        float segmentLength,
        std::vector<twisty::PathWeighting::WeightLookupTableIntegral*> weightingIntegrals,
        const twisty::PerturbUtils::BoundaryConditions& boundaryConditions,
        const PathWeighting::NormalizerStuff::BaseNormalizer& pathNormalizer
    )
    {
        uint32_t numPathsAccepted = 0;
        uint32_t numPathsUnaccepted = 0;
        uint32_t numPathsUnacceptedTangentPdf = 0;
        uint32_t numPathsUnacceptedCurvaturePdf = 0;

#if defined(ExportPathBatches)

        // This should be per thread
        int64_t numPosInCache = (numSegmentsPerCurve + 1) * ExportPathBatchCacheSize;
        std::vector<Farlor::Vector3> pathBatchCache(numPosInCache);
        {
            for (int64_t cacheEntryIdx = 0; cacheEntryIdx < ExportPathBatchCacheSize; ++cacheEntryIdx)
            {
                for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                {
                    int64_t pointEntryIdx = (numSegmentsPerCurve + 1) * cacheEntryIdx + pointIdx;
                    pathBatchCache[pointEntryIdx] = Farlor::Vector3(0.0f, 0.0f, 0.0f);
                }
            }
        }
#endif      
        const int64_t NumPosPerCurve = (numSegmentsPerCurve + 1);
        const int64_t NumTanPerCurve = (numSegmentsPerCurve + 1);
        const int64_t NumCurvaturesPerCurve = numSegmentsPerCurve;

        const int64_t CurrentThreadPosStartIdx = NumPosPerCurve * threadIdx;
        const int64_t CurrentThreadTanStartIdx = NumTanPerCurve * threadIdx;
        const int64_t CurrentThreadCurvatureStartIdx = NumCurvaturesPerCurve * threadIdx;

        // Now, we can begin the actual algorithm
        {

            // This is the perturbation piece.
            // Can we do this in place, most likely
            // This will modify pCurrentThreadCurve
            // Remember, the structure of this is:
            // Pos_0, .,,, Pos_M, Pos_[M+1], Tan_0, ..., Tan_M

            // Start at the thread's first path idx

            int64_t numCurvesInBatch = 0;
            int64_t outputIdx = 0;


            bool useOptimal = false;
            const uint32_t numRandom = 1000;
            const uint32_t numOptimal = 5000;
            uint32_t countCurrentMethod = 0;

            for (int64_t pathCount = 0; pathCount < (numPathsToSkipPerThread + numPathsPerThread); ++pathCount)
            {
                // Expect to go negative, thus int
                int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                // We can exit once this point is reached as we have generated all the paths necessary for this thread
                if (currentPathIdx >= numExperimentPaths)
                {
#if defined(ExportPathBatches)
                    if (numCurvesInBatch > 0)
                    {
                        ExportPathBatchesMutex.lock();

                        if (threadIdx == 11)
                        {
                            std::cout << "Should be exporting thread 12" << std::endl;
                        }


                        curvesMetadataFile << threadIdx << " ";
                        curvesMetadataFile << outputIdx << " ";
                        curvesMetadataFile << numCurvesInBatch << std::endl;

                        curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                        numCurvesInBatch = 0;
                        outputIdx++;

                        ExportPathBatchesMutex.unlock();
                    }
#endif
                    // We dont want to continue if we have already generated the correct number of paths.
                    break;
                }

                // Do the perturb now

                // Each time, we first copy the "old path" to the "scratch space"
                for (uint32_t segIdx = 0; segIdx <= numSegmentsPerCurve; ++segIdx)
                {
                    scratchPositionSpaceLeft[CurrentThreadPosStartIdx + segIdx] = globalPos[CurrentThreadPosStartIdx + segIdx];
                    scratchPositionSpaceRight[CurrentThreadPosStartIdx + segIdx] = globalPos[CurrentThreadPosStartIdx + segIdx];

                }
                twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
                    &scratchTangentSpaceLeft[CurrentThreadTanStartIdx], &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
                    numSegmentsPerCurve, boundaryConditions);

                twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&scratchPositionSpaceRight[CurrentThreadPosStartIdx],
                    &scratchTangentSpaceRight[CurrentThreadTanStartIdx], &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
                    numSegmentsPerCurve, boundaryConditions);

                std::uniform_int_distribution<int> diffDist(2, std::min((int)(numSegmentsPerCurve - 2), 25)); // uniform, unbiased
                int64_t diff = diffDist(rngGenerators[threadIdx]);

                std::uniform_int_distribution<int> leftPointIndexUniformDist(1, numSegmentsPerCurve - diff - 1); // uniform, unbiased
                int64_t leftPointIndex = leftPointIndexUniformDist(rngGenerators[threadIdx]);

                int64_t rightPointIndex = leftPointIndex + diff;

                assert((rightPointIndex - leftPointIndex) >= diff);
                assert(leftPointIndex < rightPointIndex);

#if defined(DetailedPurturb) && defined(SingleThreadMode)
                {
                    printf("Left point idx: %d\n", leftPointIndex);
                    printf("Right point idx: %d\n", rightPointIndex);
                }
#endif
                // We need two frames for each segment to get the new curvature and torsion.
                // we need the frame left of the segment, as well as the frame right of the segment.
                // The left point also will act as the origin for rotating the points between leftPoint and rightPoint
                const Farlor::Vector3 leftPoint = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
                const Farlor::Vector3 rightPoint = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];

                const Farlor::Vector3 N = (rightPoint - leftPoint).Normalized();

                double leftRotationAngle = 0.0;
                {
#if defined(DetailedPurturb) && defined(SingleThreadMode)
                    printf("Axis before (%.6f, %.6f, %.6f)\n",
                        N[0], N[1], N[2]
                    );
#endif

                    const Farlor::Vector3 Xss1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex - 1];
                    const Farlor::Vector3 Xs = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex];
                    const Farlor::Vector3 Xsp1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + leftPointIndex + 1];
                    const Farlor::Vector3 PL = Xs + ((Xsp1 - Xs).Dot(N)) * N;
                    const Farlor::Vector3 ZL = Xss1 - ((Xss1 - PL).Dot(N)) * N;

                    // Get side of plane Z is on
                    const Farlor::Vector3 NL = N.Cross((Xsp1 - PL).Normalized()).Normalized();
                    const double sideDistL = (ZL - PL).Dot(NL);

                    const Farlor::Vector3 ZPnorm = (ZL - PL).Normalized();
                    const Farlor::Vector3 Xsp1PLnorm = (Xsp1 - PL).Normalized();
                    double cosAngle = ZPnorm.Dot(Xsp1PLnorm);
                    cosAngle = std::max(-1.0, cosAngle);
                    cosAngle = std::min(1.0, cosAngle);
                    const double angle = acos(cosAngle);

                    const double threshold = 10e-10;

                    // In the case, we are aligned, we want a Pi rotation
                    if (abs(angle) <= threshold)
                    {
                        leftRotationAngle = TwistyPi;
                    }
                    // If we are Pi away, we want no rotation
                    else if (abs(abs(angle) - TwistyPi) < threshold)
                    {
                        leftRotationAngle = 0.0;
                    }
                    // On back side
                    else if (sideDistL < 0.0)
                    {
                        leftRotationAngle = TwistyPi - angle;
                    }
                    // On front side
                    else
                    {
                        leftRotationAngle = -1.0 * (TwistyPi - angle);
                    }
                }


                double rightRotationAngle = 0.0;
                {
                    //#if defined(DetailedPurturb) && defined(SingleThreadMode)
                    //                    printf("Axis before (%.6f, %.6f, %.6f)\n",
                    //                        N_R[0], N_R[1], N_R[2]
                    //                    );
                    //#endif

                    const Farlor::Vector3 Xes1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex - 1];
                    const Farlor::Vector3 Xe = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex];

                    // Handle case where we are rotation with end point as pivot point
                    Farlor::Vector3 Xep1(0.0f, 0.0f, 0.0f);
                    if (rightPointIndex == numSegmentsPerCurve) {
                        Xep1 = Xe + boundaryConditions.m_endPos + segmentLength * boundaryConditions.m_endDir.Normalized();
                    }
                    else {
                        Xep1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex + 1];
                    }


                    //const Farlor::Vector3 Xep1 = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + rightPointIndex + 1];
                    const Farlor::Vector3 PR = Xe + ((Xes1 - Xe).Dot(N)) * N;
                    const Farlor::Vector3 ZR = Xep1 - ((Xep1 - PR).Dot(N)) * N;

                    // Get side of plane Z is on
                    const Farlor::Vector3 NR = N.Cross((Xes1 - PR).Normalized()).Normalized();
                    const double sideDistR = (ZR - PR).Dot(NR);

                    const Farlor::Vector3 ZPnorm = (ZR - PR).Normalized();
                    const Farlor::Vector3 Xes1PLnorm = (Xes1 - PR).Normalized();
                    double cosAngle = ZPnorm.Dot(Xes1PLnorm);
                    cosAngle = std::max(-1.0, cosAngle);
                    cosAngle = std::min(1.0, cosAngle);
                    const double angle = acos(cosAngle);

                    const double threshold = 10e-10;

                    // In the case, we are aligned, we want a Pi rotation
                    if (abs(angle) <= threshold)
                    {
                        rightRotationAngle = TwistyPi;
                    }
                    // If we are Pi away, we want no rotation
                    else if (abs(abs(angle) - TwistyPi) < threshold)
                    {
                        rightRotationAngle = 0.0;
                    }
                    // On back side
                    else if (sideDistR < 0.0)
                    {
                        rightRotationAngle = TwistyPi - angle;
                    }
                    // On front side
                    else
                    {
                        rightRotationAngle = -1.0 * (TwistyPi - angle);
                    }
                }

                // Overwrite angle
                if (!useOptimal)
                {
                    countCurrentMethod++;
                    if (countCurrentMethod >= numRandom)
                    {
                        useOptimal = !useOptimal;
                    }
                    //double mult = (((pathCount / 2) % 2) == 0) ? 1 : -1;

                    std::uniform_real_distribution<float> zeroToTwoPiUniformDist(-TwistyPi * AmountOfFullRotation, TwistyPi * AmountOfFullRotation);
                    //rotationAngle = TwistyPi * 0.75 * mult;
                    double randRotationAngle = zeroToTwoPiUniformDist(rngGenerators[threadIdx]);
                    leftRotationAngle = randRotationAngle;
                    rightRotationAngle = randRotationAngle;
                    //std::cout << "Overwritten rotation angle: " << randRotationAngle << std::endl;
                }
                else
                {
                    countCurrentMethod++;
                    if (countCurrentMethod >= numOptimal)
                    {
                        useOptimal = !useOptimal;
                    }
                }

                // Left Rotation
                {
                    float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                    RotationMatrixAroundAxis(leftRotationAngle, (float*)(&N), rotationMatrix);

                    for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                    {
                        Farlor::Vector3 shiftedPoint = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                        // Rotate and stuff back in shifted point
                        RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));
                        // Update the point with the rotated version
                        scratchPositionSpaceLeft[CurrentThreadPosStartIdx + pointIdx] = shiftedPoint + leftPoint;
                    }

                    //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                    //We can do a different approach later.
                    // Here, we want to do a perturb update call
                    twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&scratchPositionSpaceLeft[CurrentThreadPosStartIdx],
                        &scratchTangentSpaceLeft[CurrentThreadTanStartIdx], &scratchCurvatureSpaceLeft[CurrentThreadCurvatureStartIdx],
                        numSegmentsPerCurve, boundaryConditions);
                }

                // Right Rotation
                {
                    float rotationMatrix[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
                    RotationMatrixAroundAxis(rightRotationAngle, (float*)(&N), rotationMatrix);

                    for (int64_t pointIdx = (leftPointIndex + 1); pointIdx < rightPointIndex; ++pointIdx)
                    {
                        Farlor::Vector3 shiftedPoint = scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx] - leftPoint;
                        // Rotate and stuff back in shifted point
                        RotateVectorByMatrix(rotationMatrix, (float*)(&shiftedPoint));
                        // Update the point with the rotated version
                        scratchPositionSpaceRight[CurrentThreadPosStartIdx + pointIdx] = shiftedPoint + leftPoint;
                    }

                    //Now, simply compute the difference in positions at the two edges of the rotated rigidbody.
                    //We can do a different approach later.
                    // Here, we want to do a perturb update call
                    twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&scratchPositionSpaceRight[CurrentThreadPosStartIdx],
                        &scratchTangentSpaceRight[CurrentThreadTanStartIdx], &scratchCurvatureSpaceRight[CurrentThreadCurvatureStartIdx],
                        numSegmentsPerCurve, boundaryConditions);
                }

                double leftPathWeightLog10 = twisty::PathWeighting::SimpleWeightCurveViaTangentDotProductLog10(&(scratchTangentSpaceLeft[CurrentThreadTanStartIdx]),
                    numSegmentsPerCurve, *weightingIntegrals[0]);

                double rightPathWeightLog10 = twisty::PathWeighting::SimpleWeightCurveViaTangentDotProductLog10(&(scratchTangentSpaceRight[CurrentThreadTanStartIdx]),
                    numSegmentsPerCurve, *weightingIntegrals[0]);

                bool useLeftRotation = (leftPathWeightLog10 >= rightPathWeightLog10);

                uint32_t numBetas = (rightPointIndex - 1) - leftPointIndex;
                std::vector<Farlor::Vector3> oldBetas(numBetas);
                std::vector<Farlor::Vector3> newBetas(numBetas);
                for (int64_t tanIdx = 0; tanIdx < numBetas; ++tanIdx)
                {
                    oldBetas[tanIdx] = globalTans[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];

                    if (useLeftRotation)
                    {
                        newBetas[tanIdx] = scratchTangentSpaceLeft[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];
                    }
                    else
                    {
                        newBetas[tanIdx] = scratchTangentSpaceRight[CurrentThreadTanStartIdx + leftPointIndex + tanIdx];
                    }
                }

                // Now we have a candidate path
                // We perform metropolis and see if we want to accept the path, i.e. copy the scratch space values to the actual curve values, or reroll a new curve
                std::uniform_real_distribution<double> uniformRandomZeroOne(0.0, 1.0);

                double acceptanceProb = uniformRandomZeroOne(rngGenerators[threadIdx]);

                // For now, we always accept a path
                // This is not using the metropolis sampling at all
                double oldPathWeightLog10 = twisty::PathWeighting::SimpleWeightCurveViaTangentDotProductLog10(&(globalTans[CurrentThreadTanStartIdx]),
                    numSegmentsPerCurve, *weightingIntegrals[0]);

                double newPathWeightLog10 = 0.0;

                if (useLeftRotation)
                {
                    newPathWeightLog10 = twisty::PathWeighting::SimpleWeightCurveViaTangentDotProductLog10(&(scratchTangentSpaceLeft[CurrentThreadTanStartIdx]),
                        numSegmentsPerCurve, *weightingIntegrals[0]);
                }
                else
                {
                    newPathWeightLog10 = twisty::PathWeighting::SimpleWeightCurveViaTangentDotProductLog10(&(scratchTangentSpaceRight[CurrentThreadTanStartIdx]),
                        numSegmentsPerCurve, *weightingIntegrals[0]);
                }

                double lambdaLog10 = newPathWeightLog10 - oldPathWeightLog10;

                double weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
                while (weightAcceptance == 0)
                {
                    weightAcceptance = uniformRandomZeroOne(rngGenerators[threadIdx]);
                }

                double weightAcceptanceLog10 = std::log10(weightAcceptance);

                for (uint32_t i = 0; i <= numSegmentsPerCurve; i++)
                {
                    if (useLeftRotation)
                    {
                        globalPos[CurrentThreadPosStartIdx + i] = scratchPositionSpaceLeft[CurrentThreadPosStartIdx + i];
                    }
                    else
                    {
                        globalPos[CurrentThreadPosStartIdx + i] = scratchPositionSpaceRight[CurrentThreadPosStartIdx + i];
                    }
                }

                twisty::PerturbUtils::RecalculateTangentsCurvaturesFromPos(&globalPos[CurrentThreadPosStartIdx],
                    &globalTans[CurrentThreadTanStartIdx], &globalCurvatures[CurrentThreadCurvatureStartIdx],
                    numSegmentsPerCurve, boundaryConditions);

                numPathsAccepted++;


                std::vector<double> differentScatteringPathWeights(weightingIntegrals.size());

                for (int scatteringIdx = 0; scatteringIdx < weightingIntegrals.size(); scatteringIdx++)
                {
                    differentScatteringPathWeights[scatteringIdx] = twisty::PathWeighting::SimpleWeightCurveViaTangentDotProductLog10(&(globalTans[CurrentThreadTanStartIdx]),
                        numSegmentsPerCurve, *weightingIntegrals[scatteringIdx]);
                }

                if (pathCount < numPathsToSkipPerThread)
                {
                    // Skip
                }
                else
                {
                    // Else, contribute to the paths
                    int64_t currentPathIdx = numPathsPerThread * threadIdx + pathCount - numPathsToSkipPerThread;
                    assert(currentPathIdx >= numPathsPerThread * threadIdx);

                    for (int scatteringIdx = 0; scatteringIdx < weightingIntegrals.size(); scatteringIdx++)
                    {
                        globalPathWeights[currentPathIdx * weightingIntegrals.size() + scatteringIdx] = differentScatteringPathWeights[scatteringIdx];
                    }

#if defined(ExportPathBatches)
                    // Add the path to the path batch
                    for (int64_t pointIdx = 0; pointIdx <= numSegmentsPerCurve; ++pointIdx)
                    {
                        Farlor::Vector3 currentPoint = globalPos[CurrentThreadPosStartIdx + pointIdx];
                        pathBatchCache[NumPosPerCurve * numCurvesInBatch + pointIdx] = currentPoint;
                    }
                    numCurvesInBatch++;

                    if (numCurvesInBatch == ExportPathBatchCacheSize)
                    {
                        ExportPathBatchesMutex.lock();

                        curvesMetadataFile << threadIdx << " ";
                        curvesMetadataFile << outputIdx << " ";
                        curvesMetadataFile << numCurvesInBatch << std::endl;

                        curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                        numCurvesInBatch = 0;
                        outputIdx++;

                        ExportPathBatchesMutex.unlock();
                    }
#endif
                }
            }

#if defined(ExportPathBatches)
            if (numCurvesInBatch > 0)
            {
                ExportPathBatchesMutex.lock();

                curvesMetadataFile << threadIdx << " ";
                curvesMetadataFile << outputIdx << " ";
                curvesMetadataFile << numCurvesInBatch << std::endl;

                curvesBinaryFile.write((char*)pathBatchCache.data(), sizeof(Farlor::Vector3) * NumPosPerCurve * numCurvesInBatch);
                numCurvesInBatch = 0;
                outputIdx++;

                ExportPathBatchesMutex.unlock();
            }
#endif
        }

        std::cout << "Num path accepted: " << numPathsAccepted << std::endl;
        std::cout << "Num path unaccepted: " << numPathsUnaccepted << std::endl;
        std::cout << "Num path unaccepted tangents: " << numPathsUnacceptedTangentPdf << std::endl;
        std::cout << "Num path unaccepted curvature: " << numPathsUnacceptedCurvaturePdf << std::endl;
    }
}