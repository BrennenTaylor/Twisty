/**
 * @file TwistyYamlUtils.h
 * @author Brennen Taylor (brtaylor1001@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-03-14
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#pragma once

#pragma warning(push, 0)
#include <yaml-cpp\yaml.h>
#pragma warning(pop)

#include <string>

namespace twisty
{
    namespace TwistyYamlLoader
    {
        /**
         * @brief Load a yaml file and return a node we can traverse.
         * The yaml parser is extended to merge yaml files together, allowing for an import of other yaml files into one.
         * 
         * @param configFilename Filename of yaml file to load
         * @param twistyConfigDir Path of directory to search for yaml file in
         * @return YAML::Node Traversable node of yaml data.
         */
        YAML::Node LoadYaml(std::string configFilename, std::string twistyConfigDir);
    }
}