#include "CommandProcessor.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <yaml-cpp/yaml.h>

namespace {
std::string trimCopy(const std::string& input) {
    const size_t start = input.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return "";
    }

    const size_t end = input.find_last_not_of(" \t\n\r\f\v");
    return input.substr(start, end - start + 1);
}

std::string stripOptionalTrailingSemicolon(const std::string& input) {
    std::string output = trimCopy(input);
    if (!output.empty() && output.back() == ';') {
        output.pop_back();
        output = trimCopy(output);
    }
    return output;
}

std::string getStringOrDefault(const YAML::Node& node,
                               const std::string& key,
                               const std::string& default_value = "") {
    if (!node || !node[key]) {
        return default_value;
    }

    try {
        return node[key].as<std::string>();
    } catch (const YAML::Exception& e) {
        std::cerr << "[CommandProcessor] WARNING: Invalid string for key '"
                  << key << "': " << e.what() << std::endl;
        return default_value;
    }
}

std::vector<float> getFloatArrayOrDefault(const YAML::Node& node,
                                          const std::string& key,
                                          const std::vector<float>& default_value) {
    if (!node || !node[key] || !node[key].IsSequence()) {
        return default_value;
    }

    std::vector<float> result;
    result.reserve(node[key].size());

    for (const auto& elem : node[key]) {
        try {
            result.push_back(elem.as<float>());
        } catch (const YAML::Exception&) {
            try {
                result.push_back(elem.as<bool>() ? 1.0f : 0.0f);
            } catch (const YAML::Exception&) {
                result.push_back(0.0f);
            }
        }
    }

    return result;
}
}

float CommandProcessor::safe_stof(const std::string& str, const std::string& context) {
    if (str.empty()) {
        std::cerr << "[ERROR] safe_stof: Empty string" << context << std::endl;
        throw std::invalid_argument("Empty string");
    }
    
    for (size_t i = 0; i < str.length(); ++i) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        if (!(std::isprint(c) && c != '\r' && c != '\n' && c != '\t')) {
            std::cerr << "\\x" << std::hex << std::setw(2) << std::setfill('0') 
                     << static_cast<int>(c) << std::dec;
        }
    }
    
    try {
        return std::stof(str);
    } catch (const std::invalid_argument& e) {
        std::cerr << "[ERROR] safe_stof: Invalid argument" << context 
                  << ": '" << str << "' - " << e.what() << std::endl;
        throw;
    } catch (const std::out_of_range& e) {
        std::cerr << "[ERROR] safe_stof: Out of range" << context 
                  << ": '" << str << "' - " << e.what() << std::endl;
        throw;
    }
}

bool CommandProcessor::isValidFloatString(const std::string& s) {
    if (s.empty()) {
        std::cerr << "[DEBUG] isValidFloatString: Empty string" << std::endl;
        return false;
    }
    
    std::string trimmed = trimCopy(s);
    
    if (trimmed.empty()) {
        std::cerr << "[DEBUG] isValidFloatString: Empty after trim" << std::endl;
        return false;
    }
    
    char first_char = trimmed[0];
    if (!(std::isdigit(static_cast<unsigned char>(first_char)) || 
          first_char == '-' || first_char == '+' || first_char == '.')) {
        std::cerr << "[DEBUG] isValidFloatString: Invalid first character '" << first_char << "'" << std::endl;
        return false;
    }
    
    int dot_count = 0;
    int digit_count = 0;
    
    for (size_t i = 0; i < trimmed.length(); ++i) {
        char c = trimmed[i];
        
        if (c == '.') {
            dot_count++;
            if (dot_count > 1) {
                std::cerr << "[DEBUG] isValidFloatString: Multiple dots" << std::endl;
                return false;
            }
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            digit_count++;
        } else if (c == '-' || c == '+') {
            if (i != 0) {
                std::cerr << "[DEBUG] isValidFloatString: Sign not at start" << std::endl;
                return false;
            }
        } else if (c == 'e' || c == 'E') {
            if (i + 1 >= trimmed.length()) {
                std::cerr << "[DEBUG] isValidFloatString: Incomplete scientific notation" << std::endl;
                return false;
            }
            char next = trimmed[i + 1];
            if (!std::isdigit(static_cast<unsigned char>(next)) && next != '-' && next != '+') {
                std::cerr << "[DEBUG] isValidFloatString: Invalid char after 'e'" << std::endl;
                return false;
            }
        } else {
            std::cerr << "[DEBUG] isValidFloatString: Invalid character '" << c << "'" << std::endl;
            return false;
        }
    }
    
    if (digit_count == 0) {
        std::cerr << "[DEBUG] isValidFloatString: No digits found" << std::endl;
        return false;
    }
    
    return true;
}




std::vector<ResetInfo> CommandProcessor::parseResetCommand(const std::string& command) {
    std::vector<ResetInfo> result;

    const std::string yaml_payload = stripOptionalTrailingSemicolon(command);
    if (yaml_payload.empty()) {
        std::cerr << "[CommandProcessor] Empty RESET payload" << std::endl;
        return result;
    }

    try {
        YAML::Node root = YAML::Load(yaml_payload);

        if (root.IsSequence()) {
            for (const auto& item : root) {
                if (item && item.IsMap()) {
                    // result.push_back(parseResetObject(item.as<YAML::Node>()));      
                    result.push_back(parseResetObject(item));               
         
                 } else {
                    std::cerr << "[CommandProcessor] WARNING: Skipping invalid RESET item" << std::endl;
                }
            }
        } else if (root.IsMap()) {
            result.push_back(parseResetObject(root));
        } else {
            std::cerr << "[CommandProcessor] RESET payload must be a YAML map or sequence" << std::endl;
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "[CommandProcessor] Failed to parse RESET YAML payload: " << e.what() << std::endl;
    }

    return result;
}

void CommandProcessor::parseActionCommands(const std::string& command) {
    clear();
    
    size_t obs_pos = command.find("OBS:");
    if (obs_pos == std::string::npos) {
        std::cerr << "[CommandProcessor] Missing 'OBS:' in command string" << std::endl;
        return;
    }

    std::string cmd_str = command.substr(0, obs_pos);
    std::string obs_str = command.substr(obs_pos + 4);

    std::stringstream ss(cmd_str);
    std::string token;
    
    while (std::getline(ss, token, ';')) {
        if (!token.empty()) {
            parseCommandToken(token);
        }
    }

    if (!obs_str.empty()) {
        parseStateFilter(obs_str);
    }
}


ResetInfo CommandProcessor::parseResetObject(const YAML::Node& node) {
    ResetInfo obj;

    if (!node || !node.IsMap()) {
        obj.name = "unknown";
        obj.position = {0.0f, 0.0f, 0.0f};
        obj.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        obj.current = {0.0f, 0.0f, 0.0f};
        return obj;
    }

    obj.name = getStringOrDefault(node, "name", "unknown");
    obj.position = getFloatArrayOrDefault(node, "position", {0.0f, 0.0f, 0.0f});
    obj.rotation = getFloatArrayOrDefault(node, "rotation", {0.0f, 0.0f, 0.0f, 1.0f});
    obj.current = getFloatArrayOrDefault(node, "current", {0.0f, 0.0f, 0.0f});

    return obj;
}

void CommandProcessor::parseCommandToken(const std::string& token) {
    std::istringstream tokenStream(token);
    std::string actuator_name, action, action_value;
    
    if (std::getline(tokenStream, actuator_name, ':') &&
        std::getline(tokenStream, action, ':') &&
        std::getline(tokenStream, action_value)) {
        
        action_value = trimCopy(action_value);
        
        try {
            float value = safe_stof(action_value, 
                                   " for command: " + actuator_name + ":" + action);
            commands_[actuator_name][action] = value;
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] CommandProcessor: Invalid value for " << actuator_name 
                      << ":" << action << " -> '" << token << "': " << e.what() << std::endl;
            commands_[actuator_name][action] = 0.0f;
        }
    } else {
        std::cerr << "[ERROR] CommandProcessor: Invalid command format: '" << token 
                  << "'. Expected: 'actuator:action:value'" << std::endl;
    }
}

void CommandProcessor::parseStateFilter(const std::string& obs_str) {
    std::istringstream obsStream(obs_str);
    std::string obj_name;

    while (std::getline(obsStream, obj_name, ';')) {
        if (!obj_name.empty()) {
            relevant_obs_names_.insert(obj_name);
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
