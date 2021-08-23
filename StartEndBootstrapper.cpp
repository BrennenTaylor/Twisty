#include "StartEndBootstrapper.h"

namespace twisty
{
    StartEndBootstrapper::StartEndBootstrapper(const Farlor::Vector3& startPos, const Farlor::Vector3& startDir,
            const Farlor::Vector3& endPos, const Farlor::Vector3& endDir, uint32_t randomSeed)
        : Bootstrapper(randomSeed)
    {
        m_startPos = startPos;
        m_startDir = startDir;
        m_endPos = endPos;
        m_endDir = endDir;
    }

    void StartEndBootstrapper::BeginReset()
    {
    }

    void StartEndBootstrapper::EndReset()
    {
    }
}