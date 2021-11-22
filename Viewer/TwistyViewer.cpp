#include "CurveViewer.h"

#include "CurveUtils.h"
#include "ExperimentRunnerCpu.h"
#include "Geometry.h"
#include "MathConsts.h"
#include "PathWeightUtils.h"
#include "Range.h"
#include "TwistyYamlUtils.h"

#include <QApplication>
#include <QSurfaceFormat>

#include <cstdint>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    QSurfaceFormat format;
    const uint32_t numSamples = 16;
    format.setSamples(numSamples);

    // Instantiate resources using app
    CurveViewer window;
    window.setFormat(format);
    window.resize(800, 600);
    window.show();

    auto CurveGenerator = [&]() -> bool
    {

        const uint32_t numSegments = 200;

        /*{
            twisty::Curve curve(numSegments);
            curve.m_arclength = 10.0f;
            curve.m_basePos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
            curve.m_baseTangent = Farlor::Vector3(1.0f, 0.0f, 0.0f);
            curve.m_baseNormal = Farlor::Vector3(0.0f, 1.0f, 0.0f);
            curve.m_baseBinormal = Farlor::Vector3(0.0f, 0.0f, 1.0f);

            curve.m_targetPos = Farlor::Vector3(10.0f, 0.0f, 0.0f);
            curve.m_targetTangent = Farlor::Vector3(1.0f, 0.0f, 0.0f);

            for (uint32_t i = 0; i < numSegments; ++i)
            {
                auto& segment = curve.m_segments[i];
                segment.m_length = curve.m_arclength / curve.m_numSegments;
                segment.m_curvature = 0.0f;
                segment.m_torsion = 0.0f;
                segment.UpdateRotation();
                window.AddCurve(curve);
            }
        }*/

        /*{
            twisty::Curve curve(numSegments);
            curve.m_arclength = 10.0f;
            curve.m_basePos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
            curve.m_baseTangent = Farlor::Vector3(1.0f, 0.0f, 0.0f);
            curve.m_baseNormal = Farlor::Vector3(0.0f, 1.0f, 0.0f);
            curve.m_baseBinormal = Farlor::Vector3(0.0f, 0.0f, 1.0f);

            curve.m_targetPos = Farlor::Vector3(0.0f, 0.0f, 0.0f);
            curve.m_targetTangent = Farlor::Vector3(1.0f, 0.0f, 0.0f);

            const float curvature = (twisty::TwistyPi * 2.0f) / numSegments / (curve.m_arclength / numSegments);

            for (uint32_t i = 0; i < numSegments; ++i)
            {
                auto& segment = curve.m_segments[i];
                segment.m_length = curve.m_arclength / curve.m_numSegments;
                segment.m_curvature = curvature;
                segment.m_torsion = 0.0f;
                segment.UpdateRotation();
            }

            float curveError = twisty::CurveUtils::CalculateCurveError(curve);
            std::cout << "Curve error: " << curveError << std::endl;
            float curveMeasure = twisty::CurveUtils::CalculateCurveMeasure(curve);
            std::cout << "Curve measure: " << curveMeasure << std::endl;
            window.AddCurve(curve);
        }*/


        // Use the actual experiment runner
        {
            // Bootstrap method
            const twisty::Range defaultBounds = { -1.0f, 1.0f };
            const Farlor::Vector3 emitterStart{ 0.0f, 0.0f, 0.0f };
            const Farlor::Vector3 emitterDir = Farlor::Vector3(1.0f, 0.0f, 0.0f).Normalized();
            twisty::RayGeometry rayEmitter(emitterStart, emitterDir);

            const Farlor::Vector3 recieverPos{ 10.0f, 0.0f, 0.0f };
            const float recieverRadius = 1.0f;
            const float recieverFov = twisty::TwistyPi;
            twisty::SphereGeometry sphereReciever(recieverPos, recieverRadius, recieverFov);

            const twisty::Range arclengthRange = { 10.0f, 30.0f };

            twisty::Bootstrapper bootstrapper(rayEmitter, sphereReciever);

            std::unique_ptr<twisty::Curve> upInitialCurve = bootstrapper.CreateCurve(numSegments);
            if (!upInitialCurve)
            {
                printf("Failed to create bootstrap curve.\n");
                return false;
            }
            upInitialCurve->ExportCurve("Exported_Curve");

            window.SetInitialCurve(*upInitialCurve);
        }

        // Linear bootstrapper
        //{
        //    // Bootstrap method
        //    twisty::testing::LinearBootstrapper bootstrapper;

        //    std::unique_ptr<twisty::Curve> upInitialCurve = bootstrapper.CreateCurve(numSegments);
        //    if (!upInitialCurve)
        //    {
        //        printf("Failed to create bootstrap curve.\n");
        //        return false;
        //    }

        //    window.AddCurve(*upInitialCurve);

        return true;
    };

    QObject::connect(&window, &CurveViewer::CurveReset, CurveGenerator);

    // Do an initial addition of a curve
    //CurveGenerator();

    return app.exec();
}