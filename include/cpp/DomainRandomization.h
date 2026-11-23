#pragma once
#include <string>
#include <yaml-cpp/yaml.h>

// Generates private scenario copies from immutable source XML; no live physics mutation.
// Each target contains additive min/max/delta values and an explicit part occurrence.
YAML::Node GenerateRandomizedScenario(const std::string& scene,
                                     const std::string& resources,
                                     const std::string& outputDirectory,
                                     const YAML::Node& targets,
                                     bool writeScenes = true);
