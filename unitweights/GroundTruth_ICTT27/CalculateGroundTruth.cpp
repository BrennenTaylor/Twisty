#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <map>

#include <FMath/FMath.h>

long double HeavisideStepFunction(const long double x)
{
   if (x > 0)
   {
      return 1.0;
   }

   return 0.0;
}

double G_3(const Farlor::Vector3& qVec, const Farlor::Vector3& bVec, double ds, double alpha, const Farlor::Vector3& n0)
{
   const double q = qVec.Magnitude();

   // First terms
   const double firstTermEval = (2.0 * M_PI) / (ds * ds * ds) * HeavisideStepFunction(2.0 - q) / q * exp(-4.0 * alpha);

   // Second term
   const double dotProductEval = qVec.Dot(n0 + bVec);
   const double secondTermEval = exp((alpha / 2.0) * (q * q + dotProductEval));

   // Bessel evaluation piece
   const double magVectorPiece = (bVec - qVec).Magnitude();
   double besselOrder = 0.0;
   const double besselEval = std::cyl_bessel_i(besselOrder,
      alpha * sqrt(1.0 - (q * q / 4.0)) * magVectorPiece
   );

   //std::cout << "q: " << q << std::endl;
   //std::cout << "First term: " << firstTermEval << std::endl;
   //std::cout << "Second term: " << secondTermEval << std::endl;
   //std::cout << "Bessel term: " << besselEval << std::endl;

   double finalTerm = firstTermEval * secondTermEval * besselEval;

   if (isnan(finalTerm))
   {
       if (firstTermEval == 0.0 && isnan(besselEval))
       {
           finalTerm = 0.0;
       }
       else
       {
           std::cout << "Weird case" << std::endl;
           std::cout << "First term: " << firstTermEval << std::endl;
            std::cout << "Bessel term: " << besselEval << std::endl;
       }
   }

   //std::cout << "Final term: " << finalTerm << std::endl;
   return finalTerm;
}

// Spherical coordinates used for integeration
// https://tutorial.math.lamar.edu/classes/calcii/sphericalcoords.aspx
double G_4(const Farlor::Vector3& q, const Farlor::Vector3& beta, double ds, double alpha, const Farlor::Vector3& n0)
{
    const uint32_t numThetaSteps = 1000;
    const double thetaMin = 0.0;
    const double thetaMax = 2.0 * M_PI;
    const double thetaStepSize = (thetaMax - thetaMin) / (numThetaSteps - 1);

    const double dTheta = ((thetaMax - thetaMin) / numThetaSteps);

    //const uint32_t numPhiSteps = 5;
    //const double phiMin = 0.0;
    //const double phiMax = M_PI;
    //const double phiStepSize = (phiMax - phiMin) / (numPhiSteps - 1);

    const uint32_t numSSteps = 1000;
    const double sMin = -1.0;
    const double sMax= 1.0;
    const double sStepSize= (sMax - sMin) / (numSSteps - 1);

    const double dS = ((sMax - sMin) / numSSteps);

    double result = 0.0;

    for (uint32_t thetaIdx = 0; thetaIdx < numThetaSteps; thetaIdx++)
    {
        const double theta = thetaMin + thetaIdx * thetaStepSize;
        for (uint32_t sIdx = 0; sIdx < numSSteps; sIdx++)
        {
            const double s = sMin+ sIdx * sStepSize;
            const double ss = sqrt(1.0 - s * s);

            Farlor::Vector3 evalVector = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();

            //std::cout << "(theta, phi): (" << theta << ", " << phi << ")" << std::endl;
            //std::cout << "sin(phi): " << sin(phi) << std::endl;

            // Function eval 
            double functionEval = G_3(q - evalVector, evalVector, ds, alpha, n0);
            result += exp(-alpha) * functionEval * dTheta * dS * exp(alpha * beta.Dot(evalVector));
        }
    }
    return result;
}

int main()
{
   const long double arclength = 12.0;
   const uint32_t numSegments = 5;

   const double scatteringConstant = 0.9;
   const double mu = 0.5;

   const double ds = arclength/ numSegments;

   const double alpha = 1.0 / (scatteringConstant * mu * ds);


   Farlor::Vector3 startPos(0.0f, 0.0f, 0.0f);
   Farlor::Vector3 startDir(0.0f, 0.0f, 1.0f);

   Farlor::Vector3 endPos(0.0f, 0.0f, 10.0f);
   Farlor::Vector3 endDir(0.0f, 0.0f, 1.0f);

   const Farlor::Vector3 q = (endPos - startPos) * (1.0 / ds) - (startDir + endDir);

   // We want to calculate G_4 Numerically
   // What do we choose for beta?
   //Farlor::Vector3 beta(0.0f, 0.0f, 1.0f);


   // Here we do the loop
   const uint32_t numThetaSteps = 2;
   const double thetaMin = 0.0;
   const double thetaMax = 2.0 * M_PI;
   const double thetaStepSize = (thetaMax - thetaMin) / (numThetaSteps - 1);

   const uint32_t numSSteps = 10;
   const double sMin = -1.0;
   const double sMax = 1.0;
   const double sStepSize = (sMax - sMin) / (numSSteps - 1);

   double result = 0.0;

   for (uint32_t thetaIdx = 0; thetaIdx < numThetaSteps; thetaIdx++)
   {
       const double theta = thetaMin + thetaIdx * thetaStepSize;
       for (uint32_t sIdx = 0; sIdx < numSSteps; sIdx++)
       {
           const double s = sMin + sIdx * sStepSize;
           const double ss = sqrt(1.0 - s * s);

           Farlor::Vector3 evalVector = Farlor::Vector3(ss * cos(theta), ss * sin(theta), s).Normalized();

           double g4Result = G_4(q, evalVector, ds, alpha, startDir);

           std::cout << "Theta, s: (" << theta << ", " << s << ")" << std::endl;
           std::cout << "\tEnd Dir: " << evalVector << std::endl;
           std::cout << "\tEval: " << g4Result << std::endl;

       }
   }

   return 0;
}
