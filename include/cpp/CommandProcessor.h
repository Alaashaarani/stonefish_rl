#ifndef COMMANDPROCESSOR_H
#define COMMANDPROCESSOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "CommonTypes.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class CommandProcessor {
public:
    CommandProcessor() = default;
    
    // Parse reset command and return robot reset information
    std::vector<ResetInfo> parseResetCommand(const std::string& command);

    
    // Parse action commands and state filters
    void parseActionCommands(const std::string& command);
    
    // Getters
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& getCommands() const { 
        return commands_; 
    }
    
    const std::unordered_set<std::string>& getRelevantStates() const { 
        return relevant_obs_names_; 
    }
    
    // Clear all stored commands and filters
    void clear();
    
    // Check if an object should be included in states
    bool isObjectRelevant(const std::string& objectName) const;

private:
    // Utility to safely convert string to float
    float safe_stof(const std::string& str, const std::string& context = "");
    bool isValidFloatString(const std::string& s);
    
    std::unordered_map<std::string, std::unordered_map<std::string, float>> commands_;
    std::unordered_set<std::string> relevant_obs_names_;
    
    // Helper methods
    ResetInfo parseObjectFromJson(const std::string& object_str);
    void parseCommandToken(const std::string& token);
    void parseStateFilter(const std::string& obs_str);
};

#endif // COMMANDPROCESSOR_H