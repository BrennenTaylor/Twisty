#include "Twisty_Cuda_Helpers.h"

#include <assert.h>

namespace twisty
{
    // Safely check for a cuda error
    void CudaSafeErrorCheck(cudaError_t error, std::string message)
    {
        if (error != cudaSuccess)
        {
            std::string errorString(cudaGetErrorString(error));
            fprintf(stderr, "ERROR: %s : %s\n", message.c_str(), errorString.c_str());
            assert(false);
        }
    }

    __device__ void Multiply3x3(float pFirst[9], float pSecond[9])
    {
        float3 row0 = {0.0f, 0.0f, 0.0f};
        row0.x = pFirst[0] * pSecond[0] + pFirst[1] * pSecond[3] + pFirst[2] * pSecond[6];
        row0.y = pFirst[0] * pSecond[1] + pFirst[1] * pSecond[4] + pFirst[2] * pSecond[7];
        row0.z = pFirst[0] * pSecond[2] + pFirst[1] * pSecond[5] + pFirst[2] * pSecond[8];

        float3 row1 = {0.0f, 0.0f, 0.0f};
        row1.x = pFirst[3] * pSecond[0] + pFirst[4] * pSecond[3] + pFirst[5] * pSecond[6];
        row1.y = pFirst[3] * pSecond[1] + pFirst[4] * pSecond[4] + pFirst[5] * pSecond[7];
        row1.z = pFirst[3] * pSecond[2] + pFirst[4] * pSecond[5] + pFirst[5] * pSecond[8];

        float3 row2 = {0.0f, 0.0f, 0.0f};
        row2.x = pFirst[6] * pSecond[0] + pFirst[7] * pSecond[3] + pFirst[8] * pSecond[6];
        row2.y = pFirst[6] * pSecond[1] + pFirst[7] * pSecond[4] + pFirst[8] * pSecond[7];
        row2.z = pFirst[6] * pSecond[2] + pFirst[7] * pSecond[5] + pFirst[8] * pSecond[8];

        pFirst[0] = row0.x;
        pFirst[1] = row0.y;
        pFirst[2] = row0.z;

        pFirst[3] = row1.x;
        pFirst[4] = row1.y;
        pFirst[5] = row1.z;

        pFirst[6] = row2.x;
        pFirst[7] = row2.y;
        pFirst[8] = row2.z;
    }

    // Calculate the rotation matrix based on a curvature and torsion value
    // pRotation is assumed to be a valid 3z43 matrix
    __device__ void UpdateRotationU(float pRotation[9],
        float curvature, float torsion, float ds)
    {
        // Overwrite with no curvature and torsion
        float k = curvature;
        float t = torsion;

        float k2 = k * k;
        float t2 = t * t;
        float l2 = k2 + t2;
        float l = sqrt(l2);

        float c = cos(l * ds);
        float s = sin(l * ds);

        if (l == 0.0f)
        {
            pRotation[0] = 1.0f;
            pRotation[1] = 0.0f;
            pRotation[2] = 0.0f;

            pRotation[3] = 0.0f;
            pRotation[4] = 1.0f;
            pRotation[5] = 0.0f;

            pRotation[6] = 0.0f;
            pRotation[7] = 0.0f;
            pRotation[8] = 1.0f;
            return;
        }

        pRotation[0] = 1 - (k2 / l2) * (1 - c);
        pRotation[1] = (k / l) * s;
        pRotation[2] = (k * t) / l2 * (1 - c);

        pRotation[3] = -k / l * s;
        pRotation[4] = c;
        pRotation[5] = t / l * s;

        pRotation[6] = (k * t) / l2 * (1 - c);
        pRotation[7] = -(t / l) * s;
        pRotation[8] = 1 - (t2 / l2) * (1 - c);
    }
}