#ifndef CONFIGLOADER_H
#define CONFIGLOADER_H

#include "CommonTypes.h"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

class ConfigLoader {
public:
    ConfigLoader() = default;
    
    ObservationConfig loadFromFile(const std::string& filepath);
    ObservationConfig loadFromString(const std::string& json_str);
    
    static ObservationConfig getDefaultConfig();

private:
    ObservationConfig parseJsonConfig(const nlohmann::json& j);  // Fixed signature
    bool validateConfig(const ObservationConfig& config);
};

#endif // CONFIGLOADER_H