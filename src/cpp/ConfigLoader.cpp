#include "ConfigLoader.h"
#include <iostream>

namespace {
std::string getStringOrDefault(const YAML::Node& node,
                               const std::string& key,
                               const std::string& default_value = "") {
    if (!node || !node[key]) {
        return default_value;
    }

    try {
        return node[key].as<std::string>();
    } catch (const YAML::Exception& e) {
        std::cerr << "[ConfigLoader] WARNING: Invalid string value for key '"
                  << key << "': " << e.what() << std::endl;
        return default_value;
    }
}
}

StateConfig ConfigLoader::loadFromFile(const std::string& filepath) {
    try {
        YAML::Node root = YAML::LoadFile(filepath);
        StateConfig config = parseYamlConfig(root);

        if (!validateConfig(config)) {
            std::cerr << "[ConfigLoader] WARNING: Config validation failed, using default" << std::endl;
            return getDefaultConfig();
        }

        std::cout << "[ConfigLoader] YAML config loaded successfully from '"
                  << filepath << "': " << config.specs.size()
                  << " state specs" << std::endl;

        return config;

    } catch (const YAML::Exception& e) {
        std::cerr << "[ConfigLoader] ERROR: Failed to parse YAML config file '"
                  << filepath << "': " << e.what() << std::endl;
        return getDefaultConfig();
    } catch (const std::exception& e) {
        std::cerr << "[ConfigLoader] ERROR: Failed to load config file '"
                  << filepath << "': " << e.what() << std::endl;
        return getDefaultConfig();
    }
}

StateConfig ConfigLoader::loadFromString(const std::string& yaml_str) {
    try {
        YAML::Node root = YAML::Load(yaml_str);
        StateConfig config = parseYamlConfig(root);

        if (!validateConfig(config)) {
            std::cerr << "[ConfigLoader] WARNING: Config validation failed, using default" << std::endl;
            return getDefaultConfig();
        }

        std::cout << "[ConfigLoader] YAML config loaded successfully: "
                  << config.specs.size() << " state specs" << std::endl;

        return config;

    } catch (const YAML::Exception& e) {
        std::cerr << "[ConfigLoader] ERROR: Failed to parse YAML config string: "
                  << e.what() << std::endl;
        return getDefaultConfig();
    } catch (const std::exception& e) {
        std::cerr << "[ConfigLoader] ERROR: Failed to load config string: "
                  << e.what() << std::endl;
        return getDefaultConfig();
    }
}

StateConfig ConfigLoader::parseYamlConfig(const YAML::Node& root) {
    StateConfig config;

    try {
        YAML::Node specs_node;

        if (root && root["state_config"] && root["state_config"]["specs"]) {
            specs_node = root["state_config"]["specs"];
        }
        else if (root && root["specs"]) {
            specs_node = root["specs"];
        }
        else {
            std::cerr << "[ConfigLoader] WARNING: YAML config has no 'state_config.specs' or root 'specs' section" << std::endl;
            return config;
        }

        if (!specs_node.IsSequence()) {
            std::cerr << "[ConfigLoader] ERROR: 'specs' must be a YAML sequence/list" << std::endl;
            return config;
        }

        for (const auto& spec_item : specs_node) {
            if (!spec_item || !spec_item.IsMap()) {
                std::cerr << "[ConfigLoader] WARNING: Skipping invalid state spec entry" << std::endl;
                continue;
            }

            StateSpec spec;
            spec.entity_name = getStringOrDefault(spec_item, "entity_name");
            spec.field_type  = getStringOrDefault(spec_item, "field_type");
            spec.component   = getStringOrDefault(spec_item, "component");
            spec.output_name = getStringOrDefault(spec_item, "output_name");
            spec.entity_1    = getStringOrDefault(spec_item, "entity_1");
            spec.entity_2    = getStringOrDefault(spec_item, "entity_2");

            if (!spec.entity_name.empty() && !spec.field_type.empty()) {
                config.specs.push_back(spec);
                std::cout << "[ConfigLoader] Added spec: " << spec.output_name
                          << " <- " << spec.entity_name << "." << spec.field_type
                          << "." << spec.component << std::endl;
            } else {
                std::cerr << "[ConfigLoader] WARNING: Skipping state spec with missing entity_name or field_type" << std::endl;
            }
        }

    } catch (const YAML::Exception& e) {
        std::cerr << "[ConfigLoader] ERROR parsing YAML: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ConfigLoader] ERROR parsing config: " << e.what() << std::endl;
    }

    return config;
}

bool ConfigLoader::validateConfig(const StateConfig& config) {
    if (config.specs.empty()) {
        std::cerr << "[ConfigLoader] WARNING: No state specs configured" << std::endl;
        return false;
    }

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

    config.specs.push_back({"girona", "position", "x", "robot_x"});
    config.specs.push_back({"girona", "position", "y", "robot_y"});
    config.specs.push_back({"girona", "position", "z", "robot_z"});
    config.specs.push_back({"girona", "rotation", "yaw", "robot_yaw"});
    config.specs.push_back({"girona", "collision", "binary", "collision_flag"});

    std::cout << "[ConfigLoader] Using default configuration with "
              << config.specs.size() << " specs" << std::endl;

    return config;
}
