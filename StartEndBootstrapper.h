/**
 * @file StartEndBootstrapper.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-03-19
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "Bootstrapper.h"
#include "Range.h"

#include <FMath\Vector3.h>

#include <cstdint>

namespace twisty
{
    /**
     * @brief Bootstrapper which uses specified start and end directions and positions.
     *
     */
    class StartEndBootstrapper : public Bootstrapper
    {
    public:

        /**
         * @brief
         *
         */
        StartEndBootstrapper(const Farlor::Vector3& startPos, const Farlor::Vector3& startDir,
            const Farlor::Vector3& endPos, const Farlor::Vector3& endDir, Range arclengthRange, uint32_t randomSeed);

    protected:
        virtual void BeginReset() override;
        virtual void EndReset() override;
    };
}