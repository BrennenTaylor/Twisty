#include <FMath/Vector3.h>

#define _USE_MATH_DEFINES
#include <math.h>

const double scatteringCoefficient = 0.08;
const double absorbtionCoefficient = 0.1;

double GreensFunctionApprox(double s, const Farlor::Vector3& x2, const Farlor::Vector3& n2, const Farlor::Vector3& x1, const Farlor::Vector3& n1)
{
    

    const double p = scatteringCoefficient / 2.0;

    const Farlor::Vector3 r = p * (x2 - x1);
    const double A = pow((p*s - tanh(p*s)), -1);
    const double B = tanh(p*s) / 2.0;
    const double C = 9.0 / (2.0 * p * s);
    const double D = 3.0 * A * B * B - ((3.0) / (2.0 * sinh(2.0 * p * s)));
    const double E = 3.0 * A * B;
    const double F = (3.0 / 2.0) * A;



    double normalization = 1.0f;
    {
        normalization *= (p * p * p);
        double num = sqrt(F) * (E * E - 2.0 * D * F);
        double den = 4.0 * pow(M_PI, (5.0 / 2.0)) * (exp((E * E) / (F - D)) - exp(D));
        normalization *= (num / den);
        normalization *= exp(C);
    }

    double inside = (-C - D * n2.Dot(n1) + E * r.Dot(n2 + n1) - F * r.SqrMagnitude() - absorbtionCoefficient * s);
    double result = normalization * exp(inside);
    

    return result;
}

int main()
{
    Farlor::Vector3 startingPosition(0.0f, 0.0f, 0.0f);
    Farlor::Vector3 startingDir(1.0f, 0.0f, 0.0f);

    Farlor::Vector3 endingPosition(10.0f, 0.0f, 0.0f);
    Farlor::Vector3 endingDir(1.0f, 0.0f, 0.0f);

    const double minS = (endingPosition - startingPosition).Magnitude();
    const double maxS = minS * 5.0;
    const uint32_t numSteps = 100;
    const double stepSize = (maxS - minS) / numSteps;

    std::cout << "s, greens function" << std::endl;
    for (uint32_t i = 0; i < numSteps; ++i)
    {
        const double s = minS + stepSize * i;
        double green = GreensFunctionApprox(s, endingPosition, endingDir, startingPosition, startingDir);
        std::cout << s << ", " << green << std::endl;
    }

    return 0;
}