#ifndef COMMANDPROCESSOR_H
#define COMMANDPROCESSOR_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "CommonTypes.h"


class CommandProcessor {
public:
    CommandProcessor() = default;
    
    // Parse reset command and return robot reset information
    std::vector<RobotResetInfo> parseResetCommand(const std::string& command);

    
    // Parse action commands and observation filters
    void parseActionCommands(const std::string& command);
    
    // Getters
    const std::unordered_map<std::string, std::unordered_map<std::string, float>>& getCommands() const { 
        return commands_; 
    }
    
    const std::unordered_set<std::string>& getRelevantObservations() const { 
        return relevant_obs_names_; 
    }
    
    // Clear all stored commands and filters
    void clear();
    
    // Check if an object should be included in observations
    bool isObjectRelevant(const std::string& objectName) const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, float>> commands_;
    std::unordered_set<std::string> relevant_obs_names_;
    
    // Helper methods
    RobotResetInfo parseObjectFromJson(const std::string& object_str);
    void parseCommandToken(const std::string& token);
    void parseObservationFilter(const std::string& obs_str);
};

#endif // COMMANDPROCESSOR_H