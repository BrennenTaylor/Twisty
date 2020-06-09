#define __CUDA_ARCH__
#define __cminpack_float__

#include "cminpack.h"

#include <math.h>

#include <iostream>

int fcn_nn(void *p, int n, const __cminpack_real__ *x, __cminpack_real__ *fvec, int iflag)
{
    return 1;
}

int main()
{
     auto testErrorFunc = [](void *p, int n, const __cminpack_real__ *x, __cminpack_real__ *fvec, int iflag) -> int
     {
         const int rootLevel = 10;

         for (int i = 0; i < n; ++i)
         {
             fvec[i] = std::pow(x[i] + 1.0f, 1.0f / rootLevel) - std::pow(i + 1.0f, 1.0f / rootLevel);
         }
         // Return negative value to exit
         return 0;
     };

     const int numParams = 5;
     const float tol = 0.0000001f;

     __cminpack_real__ initialGuess[numParams];
     initialGuess[0] = 1.0f;
     initialGuess[1] = 1.0f;
     initialGuess[2] = 1.0f;
     initialGuess[3] = 1.0f;
     initialGuess[4] = 1.0f;

     __cminpack_real__ outputEval[numParams];
     outputEval[0] = 0.0f;
     outputEval[1] = 0.0f;
     outputEval[2] = 0.0f;
     outputEval[3] = 0.0f;
     outputEval[4] = 0.0f;

    // Worrking area size needs to be no less than(n*(3 * n + 13)) / 2

     __cminpack_real__ workingArea[300];

    auto value = __cminpack_func__(hybrd1)(nullptr, numParams, initialGuess, outputEval, tol, workingArea, 300);
    printf("Value returned from func: %d\n", value);

    for (int i = 0; i < numParams; ++i)
    {
        printf("Pt %d: (%f, %f)\n", i, initialGuess[i], outputEval[i]);
    }
}