#pragma once
#include <yaml-cpp/yaml.h>
namespace sf { class SimulationManager; }
// Absolute, validated updates at an episode boundary. Physics must not be stepping.
YAML::Node ApplyRuntimeDomainRandomization(sf::SimulationManager& simulation,
                                         const YAML::Node& targets);
