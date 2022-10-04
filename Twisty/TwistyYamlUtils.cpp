#include "TwistyYamlUtils.h"

#include <experimental\filesystem>

#include <iostream>
#include <vector>

namespace twisty
{
    namespace TwistyYamlLoader
    {
        // Merges node b into a
        // "Borrowed" from: https://stackoverflow.com/questions/41326112/how-to-merge-node-in-yaml-cpp
        inline const YAML::Node & cnode(const YAML::Node &n) {
            return n;
        }

        YAML::Node MergeNodes(YAML::Node a, YAML::Node b)
        {
            if (!b.IsMap()) {
                // If b is not a map, merge result is b, unless b is null
                return b.IsNull() ? a : b;
            }
            if (!a.IsMap()) {
                // If a is not a map, merge result is b
                return b;
            }
            if (!b.size()) {
                // If a is a map, and b is an empty map, return a
                return a;
            }
            // Create a new map 'c' with the same mappings as a, merged with b
            auto c = YAML::Node(YAML::NodeType::Map);
            for (auto n : a) {
                if (n.first.IsScalar()) {
                    const std::string & key = n.first.Scalar();
                    auto t = YAML::Node(cnode(b)[key]);
                    if (t) {
                        c[n.first] = MergeNodes(n.second, t);
                        continue;
                    }
                }
                c[n.first] = n.second;
            }
            // Add the mappings from 'b' not already in 'c'
            for (auto n : b) {
                if (!n.first.IsScalar() || !cnode(c)[n.first.Scalar()]) {
                    c[n.first] = n.second;
                }
            }
            return c;
        }

        YAML::Node LoadYaml(std::string configFilename, std::string twistyConfigDir)
        {
            std::experimental::filesystem::path configPath(twistyConfigDir);

            YAML::Node rootNode = YAML::LoadFile(configPath.string() + configFilename);
            std::vector<YAML::Node> inheritNodes;
            if (!rootNode["inherits"])
            {
                return rootNode;
            }

            YAML::Node inheritsNode = rootNode["inherits"];
            for (int i = 0; i < inheritsNode.size(); i++)
            {
                YAML::Node node = LoadYaml(inheritsNode[i].as<std::string>(), twistyConfigDir);
                inheritNodes.push_back(node);
            }
            rootNode.remove(inheritNodes);

            // This is the combined node, made of:
            // First, the inherited roots
            // Then finally, the Root
            YAML::Node combinedNode;
            for (auto node : inheritNodes)
            {
                combinedNode = MergeNodes(combinedNode, node);
            }
            combinedNode = MergeNodes(combinedNode, rootNode);
            return combinedNode;
        }
    }
}