#ifndef CONFIGLOADER_H
#define CONFIGLOADER_H

#include "CommonTypes.h"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

class ConfigLoader {
public:
    ConfigLoader() = default;
    
    StateConfig loadFromFile(const std::string& filepath);
    StateConfig loadFromString(const std::string& json_str);
    
    static StateConfig getDefaultConfig();

private:
    StateConfig parseJsonConfig(const nlohmann::json& j);  // Fixed signature
    bool validateConfig(const StateConfig& config);
};

#endif // CONFIGLOADER_H