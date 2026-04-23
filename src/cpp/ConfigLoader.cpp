#include "ConfigLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>

StateConfig ConfigLoader::loadFromFile(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[ConfigLoader] ERROR: Cannot open config file: " << filepath << std::endl;
            return getDefaultConfig();
        }
        
        nlohmann::json j;
        file >> j;
        return parseJsonConfig(j);
        
    } catch (const std::exception& e) {
        std::cerr << "[ConfigLoader] ERROR: Failed to parse config file '" << filepath 
                  << "': " << e.what() << std::endl;
        return getDefaultConfig();
    }
}

StateConfig ConfigLoader::loadFromString(const std::string& json_str) {
    try {
        nlohmann::json j = nlohmann::json::parse(json_str);
        StateConfig config = parseJsonConfig(j);
        
        if (!validateConfig(config)) {
            std::cerr << "[ConfigLoader] WARNING: Config validation failed, using default" << std::endl;
            return getDefaultConfig();
        }
        
        std::cout << "[ConfigLoader] Config loaded successfully: " 
                  << config.specs.size() << " state specs" << std::endl;
        
        return config;
        
    } catch (const std::exception& e) {
        std::cerr << "[ConfigLoader] ERROR: Failed to parse config string: " << e.what() << std::endl;
        return getDefaultConfig();
    }
}

StateConfig ConfigLoader::parseJsonConfig(const nlohmann::json& j) {
    StateConfig config;
    
    try {
        // Parse state_config section
        if (j.contains("state_config")) {
            const auto& obs_config = j["state_config"];
            
            // Parse specs array
            if (obs_config.contains("specs")) {
                for (const auto& spec_item : obs_config["specs"]) {
                    StateSpec spec;
                    spec.entity_name = spec_item.value("entity_name", "");
                    spec.field_type = spec_item.value("field_type", "");
                    spec.component = spec_item.value("component", "");
                    spec.output_name = spec_item.value("output_name", "");
                    spec.entity_1 = spec_item.value("entity_1","");
                    spec.entity_2 = spec_item.value("entity_2","");

                    if (!spec.entity_name.empty() && !spec.field_type.empty()) {
                        config.specs.push_back(spec);
                        std::cout << "[ConfigLoader] Added spec: " << spec.output_name 
                                  << " <- " << spec.entity_name << "." << spec.field_type 
                                  << "." << spec.component << std::endl;
                    }
                }
            }
        }
        
        // If no state_config found, try root level specs
        else if (j.contains("specs")) {
            for (const auto& spec_item : j["specs"]) {
                StateSpec spec;
                spec.entity_name = spec_item.value("entity_name", "");
                spec.field_type = spec_item.value("field_type", "");
                spec.component = spec_item.value("component", "");
                spec.output_name = spec_item.value("output_name", "");
                
                if (!spec.entity_name.empty() && !spec.field_type.empty()) {
                    config.specs.push_back(spec);
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[ConfigLoader] ERROR parsing JSON: " << e.what() << std::endl;
    }
    
    return config;
}

bool ConfigLoader::validateConfig(const StateConfig& config) {
    // Basic validation
    if (config.specs.empty()) {
        std::cerr << "[ConfigLoader] WARNING: No state specs configured" << std::endl;
        return false;
    }
    
    // Validate that all specs have required fields
    for (const auto& spec : config.specs) {
        if (spec.entity_name.empty()) {
            std::cerr << "[ConfigLoader] ERROR: Spec missing entity_name" << std::endl;
            return false;
        }
        if (spec.field_type.empty()) {
            std::cerr << "[ConfigLoader] ERROR: Spec missing field_type" << std::endl;
            return false;
        }
    }
    
    return true;
}

StateConfig ConfigLoader::getDefaultConfig() {
    StateConfig config;
    
    // Default: observe robot position and yaw
    config.specs.push_back({"girona", "position", "x", "robot_x"});
    config.specs.push_back({"girona", "position", "y", "robot_y"});
    config.specs.push_back({"girona", "position", "z", "robot_z"});
    config.specs.push_back({"girona", "rotation", "yaw", "robot_yaw"});
    config.specs.push_back({"girona", "collision", "binary", "collision_flag"});
    
    std::cout << "[ConfigLoader] Using default configuration with " 
              << config.specs.size() << " specs" << std::endl;
    
    return config;
}