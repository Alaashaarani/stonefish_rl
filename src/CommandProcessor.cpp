#include "CommandProcessor.h"
#include <iostream>
#include <sstream>
#include <algorithm>

std::vector<RobotResetInfo> CommandProcessor::parseResetCommand(const std::string& command) {
    std::vector<RobotResetInfo> result;
    size_t pos = 0;

    // Find all objects in the format { ... }
    while ((pos = command.find("{", pos)) != std::string::npos) {
        size_t end = command.find("}", pos);
        if (end == std::string::npos) break;
        
        std::string object_str = command.substr(pos, end - pos + 1);
        RobotResetInfo obj = parseObjectFromJson(object_str);
        result.push_back(obj);
        
        pos = end + 1;
    }
    // debug output
    // std::cout << "[CommandProcessor] Parsed " << result.size() << " reset objects" << std::endl;
    return result;
}

void CommandProcessor::parseActionCommands(const std::string& command) {
    clear(); // Clear previous commands
    
    size_t obs_pos = command.find("OBS:");
    if (obs_pos == std::string::npos) {
        std::cerr << "[CommandProcessor] Missing 'OBS:' in command string" << std::endl;
        return;
    }

    std::string cmd_str = command.substr(0, obs_pos);
    std::string obs_str = command.substr(obs_pos + 4);

    // Parse action commands
    std::stringstream ss(cmd_str);
    std::string token;
    
    while (std::getline(ss, token, ';')) {
        if (!token.empty()) {
            parseCommandToken(token);
        }
    }

    // Parse observation filters
    if (!obs_str.empty()) {
        parseObservationFilter(obs_str);
    }
    /* Debug output 
    std::cout << "[CommandProcessor] Parsed " << commands_.size() << " actuators, " 
    << relevant_obs_names_.size() << " observation filters" << std::endl;
    */
}

RobotResetInfo CommandProcessor::parseObjectFromJson(const std::string& object_str) {
    RobotResetInfo obj;
    
    // Parse name
    size_t name_pos = object_str.find("\"name\"");
    if (name_pos != std::string::npos) {
        size_t start_quote = object_str.find("\"", name_pos + 6);
        size_t end_quote = object_str.find("\"", start_quote + 1);
        if (start_quote != std::string::npos && end_quote != std::string::npos) {
            obj.name = object_str.substr(start_quote + 1, end_quote - start_quote - 1);
        }
    }

    // Parse position
    size_t pos_pos = object_str.find("\"position\"");
    if (pos_pos != std::string::npos) {
        size_t bracket_start = object_str.find("[", pos_pos);
        size_t bracket_end = object_str.find("]", bracket_start);
        if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
            std::string list = object_str.substr(bracket_start + 1, bracket_end - bracket_start - 1);
            std::stringstream ss(list);
            std::string val;
            while (std::getline(ss, val, ',')) {
                obj.position.push_back(std::stof(val));
            }
        }
    }

    // Parse rotation
    size_t rot_pos = object_str.find("\"rotation\"");
    if (rot_pos != std::string::npos) {
        size_t bracket_start = object_str.find("[", rot_pos);
        size_t bracket_end = object_str.find("]", bracket_start);
        if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
            std::string list = object_str.substr(bracket_start + 1, bracket_end - bracket_start - 1);
            std::stringstream ss(list);
            std::string val;
            while (std::getline(ss, val, ',')) {
                obj.rotation.push_back(std::stof(val));
            }
        }
    }

    return obj;
}

void CommandProcessor::parseCommandToken(const std::string& token) {
    std::istringstream tokenStream(token);
    std::string actuator_name, action, action_value;
    
    if (std::getline(tokenStream, actuator_name, ':') &&
        std::getline(tokenStream, action, ':') &&
        std::getline(tokenStream, action_value)) {
        
        try {
            float value = std::stof(action_value);
            commands_[actuator_name][action] = value;
            // debug print
            // std::cout << "[CommandProcessor] Command: " << actuator_name << ":" 
                    //   << action << " = " << value << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[CommandProcessor] Invalid value for " << actuator_name 
                      << ":" << action << " -> '" << token << "': " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[CommandProcessor] Invalid command format: '" << token 
                  << "'. Expected: 'actuator:action:value'" << std::endl;
    }
}

void CommandProcessor::parseObservationFilter(const std::string& obs_str) {
    std::istringstream obsStream(obs_str);
    std::string obj_name;

    while (std::getline(obsStream, obj_name, ';')) {
        if (!obj_name.empty()) {
            relevant_obs_names_.insert(obj_name);
            std::cout << "[CommandProcessor] Observation filter: " << obj_name << std::endl;
        }
    }
}

void CommandProcessor::clear() {
    commands_.clear();
    relevant_obs_names_.clear();
}

bool CommandProcessor::isObjectRelevant(const std::string& objectName) const {
    return relevant_obs_names_.empty() || relevant_obs_names_.count(objectName) > 0;
}