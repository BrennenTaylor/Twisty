#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <map>

#include <FMath/FMath.h>

const double alpha = 0.1;
const double ds = 10.0 / 5;
const double n0_x = 0.0;
const double n0_y = 0.0;
const double n0_z = 1.0;

long double HeavisideStepFunction(const long double x)
{
   if (x > 0)
   {
      return 1.0;
   }

   return 0.0;
}

double G_3(double q_x, double q_y, double q_z,
   double b_x, double b_y, double b_z)
{
   const double q = sqrt(q_x * q_x + q_y * q_y + q_z * q_z);

   // First terms
   const double firstTermEval = (2.0 * M_PI) / (ds * ds * ds) * HeavisideStepFunction(2.0 - q) / q * exp(-4.0 * alpha);

   // Second term
   const double dotProductEval = q_x * (n0_x + b_x) + q_y * (n0_y + b_y) + q_z * (n0_z + b_z);
   const double secondTermEval = exp((alpha / 2.0) * (q * q + dotProductEval));

   // Bessel evaluation piece
   const double magVectorPiece = sqrt(
      pow(b_x - q_x, 2.0)
      + pow(b_y - q_y, 2.0)
      + pow(b_z - q_z, 2.0)
   );
   double besselOrder = 0.0;
   const double besselEval = std::cyl_bessel_i(besselOrder,
      alpha * sqrt(1.0 - (q * q / 4.0)) * magVectorPiece
   );

   return firstTermEval * secondTermEval * besselEval;
}

int main()
{
   const uint32_t numThetaSteps = 1000;
   const uint32_t numPhiSteps = 1000;

   const long double arclength = 12.0;

   Farlor::Vector3 startPos(0.0f, 0.0f, 0.0f);
   Farlor::Vector3 startDir(0.0f, 0.0f, 1.0f);

   Farlor::Vector3 endPos(0.0f, 0.0f, 10.0f);
   Farlor::Vector3 endDir(0.0f, 0.0f, 1.0f);

   const Farlor::Vector3 q = (startPos - endPos) * (1.0 / ds) - (startDir + endDir);

   // We want to calculate G_4 Numerically

   return 0;
}
