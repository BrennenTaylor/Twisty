/**
 * @file Range.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-03-18
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#pragma once

namespace twisty
{
    /**
     * @brief Helper structure representing a start and ending range of float values.
     * 
     */
    struct Range
    {
        float m_min = 0.0f;
        float m_max = 0.0f;
    };
}