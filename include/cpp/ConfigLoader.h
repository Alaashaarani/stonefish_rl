#ifndef CONFIGLOADER_H
#define CONFIGLOADER_H

#include "CommonTypes.h"
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

class ConfigLoader {
public:
    ConfigLoader() = default;

    StateConfig loadFromFile(const std::string& filepath);
    StateConfig loadFromString(const std::string& yaml_str);

    static StateConfig getDefaultConfig();

private:
    // Function name intentionally kept the same for now, per request.
    StateConfig parseYamlConfig(const YAML::Node& root);
    bool validateConfig(const StateConfig& config);
};

#endif // CONFIGLOADER_H
