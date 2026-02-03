#include "CommandProcessor.h"
#include <iostream>
#include <sstream>
#include <algorithm>

// Helper function to safely convert string to float with detailed error reporting

#include <cctype>  // ADD THIS INCLUDE
#include <iomanip> // ADD THIS FOR hex output

float CommandProcessor::safe_stof(const std::string& str, const std::string& context) {
    // Check if string is empty
    if (str.empty()) {
        std::cerr << "[ERROR] safe_stof: Empty string" << context << std::endl;
        throw std::invalid_argument("Empty string");
    }
    
    // DEBUG: Print the actual string with non-printable characters
    // std::cerr << "[DEBUG] safe_stof" << context << ": length=" << str.length() << ", content='";
    for (size_t i = 0; i < str.length(); ++i) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        if (std::isprint(c) && c != '\r' && c != '\n' && c != '\t') {
            // std::cerr << c;
        } else {
            // Show non-printable as hex
            std::cerr << "\\x" << std::hex << std::setw(2) << std::setfill('0') 
                     << static_cast<int>(c) << std::dec;
        }
    }
    // std::cerr << "'" << std::endl;
    
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
    
    // First, check for completely empty after trimming
    std::string trimmed = s;
    
    // Trim leading whitespace
    size_t start = trimmed.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        std::cerr << "[DEBUG] isValidFloatString: All whitespace" << std::endl;
        return false;  // String is all whitespace
    }
    
    // Trim trailing whitespace
    size_t end = trimmed.find_last_not_of(" \t\n\r\f\v");
    trimmed = trimmed.substr(start, end - start + 1);
    
    if (trimmed.empty()) {
        std::cerr << "[DEBUG] isValidFloatString: Empty after trim" << std::endl;
        return false;
    }
    
    // Simple validation: must start with digit, minus, plus, or dot
    char first_char = trimmed[0];
    if (!(std::isdigit(static_cast<unsigned char>(first_char)) || 
          first_char == '-' || first_char == '+' || first_char == '.')) {
        std::cerr << "[DEBUG] isValidFloatString: Invalid first character '" << first_char << "'" << std::endl;
        return false;
    }
    
    // Count dots
    int dot_count = 0;
    int digit_count = 0;
    
    for (size_t i = 0; i < trimmed.length(); ++i) {
        char c = trimmed[i];
        
        if (c == '.') {
            dot_count++;
            if (dot_count > 1) {
                std::cerr << "[DEBUG] isValidFloatString: Multiple dots" << std::endl;
                return false;  // Multiple dots not allowed
            }
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            digit_count++;
        } else if (c == '-' || c == '+') {
            // Sign must be at position 0
            if (i != 0) {
                std::cerr << "[DEBUG] isValidFloatString: Sign not at start" << std::endl;
                return false;
            }
        } else if (c == 'e' || c == 'E') {
            // Scientific notation - check next character
            if (i + 1 >= trimmed.length()) {
                std::cerr << "[DEBUG] isValidFloatString: Incomplete scientific notation" << std::endl;
                return false;
            }
            // Next char must be digit or sign
            char next = trimmed[i + 1];
            if (!std::isdigit(static_cast<unsigned char>(next)) && next != '-' && next != '+') {
                std::cerr << "[DEBUG] isValidFloatString: Invalid char after 'e'" << std::endl;
                return false;
            }
        } else {
            std::cerr << "[DEBUG] isValidFloatString: Invalid character '" << c << "'" << std::endl;
            return false;  // Invalid character
        }
    }
    
    if (digit_count == 0) {
        std::cerr << "[DEBUG] isValidFloatString: No digits found" << std::endl;
        return false;
    }
    
    return true;
}
// end of helper function 

std::vector<ResetInfo> CommandProcessor::parseResetCommand(const std::string& command) {
    std::vector<ResetInfo> result;
    size_t pos = 0;

    // Find all objects in the format { ... }
    while ((pos = command.find("{", pos)) != std::string::npos) {
        size_t end = command.find("}", pos);
        if (end == std::string::npos) break;
        
        std::string object_str = command.substr(pos, end - pos + 1);
        ResetInfo obj = parseObjectFromJson(object_str);
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
    //  Debug output 
    // std::cout << "[CommandProcessor] Parsed " << commands_.size() << " actuators, " 
    // << relevant_obs_names_.size() << " observation filters" << std::endl;
    
}


ResetInfo CommandProcessor::parseObjectFromJson(const std::string& object_str) {
    ResetInfo obj;
    
    try {
        json j = json::parse(object_str);
    
        

        // Use helper function to get array with type conversion
        auto getFloatArray = [](const json& j, const std::string& key, 
                                const std::vector<float>& default_value) -> std::vector<float> {
            if (!j.contains(key) || !j[key].is_array()) {
                // debug output
                // std::cerr << "[WARNING] Missing or invalid '" << key << "' field" << std::endl;
                return default_value;
            }
            
            std::vector<float> result;
            for (const auto& elem : j[key]) {
                if (elem.is_number()) {
                    result.push_back(elem.get<float>());
                } else if (elem.is_boolean()) {
                    // Question: What should we do with booleans in numeric arrays?
                    // Option 1: Convert to 0.0/1.0
                    result.push_back(elem.get<bool>() ? 1.0f : 0.0f);
                    
                    // Option 2: Throw error for invalid data
                    // throw json::type_error::create(302, 
                    //     "Boolean in numeric array: " + elem.dump());
                } else {
                    // For other types, try conversion or use 0.0
                    result.push_back(0.0f);
                }
            }
            return result;
        };

        // Get values
        obj.name = j.value("name", "unknown");
        obj.position = getFloatArray(j, "position", {0.0f, 0.0f, 0.0f});
        obj.rotation = getFloatArray(j, "rotation", {0.0f, 0.0f, 0.0f, 1.0f});
        obj.current = getFloatArray(j, "current", {0.0f, 0.0f, 0.0f}); // current need to be added to one robot only

    } catch (const json::exception& e) {
        std::cerr << "[ERROR] JSON parsing failed: " << e.what() << std::endl;
        throw;  // Or return default object
    }
    
    return obj;
}

void CommandProcessor::parseCommandToken(const std::string& token) {
    std::istringstream tokenStream(token);
    std::string actuator_name, action, action_value;
    
    if (std::getline(tokenStream, actuator_name, ':') &&
        std::getline(tokenStream, action, ':') &&
        std::getline(tokenStream, action_value)) {
        
        // Trim whitespace from action_value
        action_value.erase(0, action_value.find_first_not_of(" \t\n\r\f\v"));
        action_value.erase(action_value.find_last_not_of(" \t\n\r\f\v") + 1);
        
        try {
            // Use our safe_stof function instead of std::stof
            float value = safe_stof(action_value, 
                                   " for command: " + actuator_name + ":" + action);
            commands_[actuator_name][action] = value;
            
            // Optional: uncomment for debug
            // std::cout << "[CommandProcessor] Command: " << actuator_name << ":" 
            //           << action << " = " << value << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[ERROR] CommandProcessor: Invalid value for " << actuator_name 
                      << ":" << action << " -> '" << token << "': " << e.what() << std::endl;
            
            // Option 1: Set default value (0.0) to continue execution
            commands_[actuator_name][action] = 0.0f;
            
            // Option 2: Re-throw to stop execution
            // throw;
        }
    } else {
        std::cerr << "[ERROR] CommandProcessor: Invalid command format: '" << token 
                  << "'. Expected: 'actuator:action:value'" << std::endl;
    }
}

void CommandProcessor::parseObservationFilter(const std::string& obs_str) {
    std::istringstream obsStream(obs_str);
    std::string obj_name;

    while (std::getline(obsStream, obj_name, ';')) {
        if (!obj_name.empty()) {
            relevant_obs_names_.insert(obj_name);
            // std::cout << "[CommandProcessor] Observation filter: " << obj_name << std::endl;
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