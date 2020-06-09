#include <Curve.h>

using namespace twisty;

int main(int argc, char *argv[])
{
    // Farlor::Matrix3x3 f0(Farlor::Vector3(1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 1.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));
    // Farlor::Matrix3x3 f1(Farlor::Vector3(0.0f, 1.0f, 0.0f), Farlor::Vector3(-1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));

    Farlor::Matrix3x3 f0(Farlor::Vector3(1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f), Farlor::Vector3(0.0f, -1.0f, 0.0f));
    Farlor::Matrix3x3 f1(Farlor::Vector3(0.0f, 0.0f, 1.0f), Farlor::Vector3(-1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, -1.0f, 0.0f));

    // Rotation around tangent
    // Farlor::Matrix3x3 f0(Farlor::Vector3(1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 1.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f));
    // Farlor::Matrix3x3 f1(Farlor::Vector3(1.0f, 0.0f, 0.0f), Farlor::Vector3(0.0f, 0.0f, 1.0f), Farlor::Vector3(0.0f, -1.0f, 0.0f));

    const float ds = 100.0f;

    Segment testSeg;
    testSeg.m_length = ds;
    {
        float curvature = ((f1.m_rows[0] - f0.m_rows[0]) * (1.0f / ds)).Magnitude();
        testSeg.m_curvature = curvature;
    }

    {
        auto torsionLeft = -1.0f * f0.m_rows[1];
        auto torsionRight = (f1.m_rows[2] - f0.m_rows[2]) * (1.0f / ds);
        float torsion = torsionLeft.Dot(torsionRight);
        testSeg.m_torsion = torsion;
    }
    testSeg.UpdateRotation();

    std::cout << "Curvature: " << testSeg.m_curvature << std::endl;
    std::cout << "Torsion: " << testSeg.m_torsion << std::endl;

    std::cout << "Rotation mat: " << testSeg.m_rotationMatrix << std::endl;

    {
        Farlor::Matrix3x3 rotatedFrame = testSeg.m_rotationMatrix * f0;
        std::cout << "Rotation * Frame: " << rotatedFrame << std::endl;
        std::cout << "Target after rot: " << f1 << std::endl;
    }
    {
        Farlor::Matrix3x3 rotatedFrame = f0 * testSeg.m_rotationMatrix;
        std::cout << "Frame * Rotation: " << rotatedFrame << std::endl;
        std::cout << "Target after rot: " << f1 << std::endl;
    }
}