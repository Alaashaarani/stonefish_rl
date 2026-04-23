#ifndef STONEFISH_RL_H
#define STONEFISH_RL_H

#include "ZMQCommunicator.h"
#include "StateManager.h"
#include "CommandProcessor.h"
#include "ActuatorController.h"
#include "ConfigLoader.h"

#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/core/SimulationApp.h>
#include <Stonefish/core/ScenarioParser.h>
#include <Stonefish/entities/forcefields/Uniform.h>
#include <vector>
#include <string>

class StonefishRL : public sf::SimulationManager {
public:
    /**
     * @brief Constructor for StonefishRL
     * @param path Path to the .scn file
     * @param state_conf_path Path to the state JSON config
     * @param action_conf_path Path to the action JSON config
     * @param frequency Simulation frequency (Hz)
     * @param port Unique ZMQ port for this instance (default 5555)
     */
    StonefishRL(const std::string &path, 
                const std::string &state_conf_path, 
                const std::string &action_conf_path, 
                double frequency,
                int port = 5555);

    // Core RL Communication loop
    std::string RecieveInstructions(sf::SimulationApp& simApp);
    void SendStates();
    void ApplyCommands(const std::string& str_cmds);
    
    // Stonefish lifecycle overrides
    virtual void BuildScenario() override;
    void ExitRequest();

private:
    // Communication & Configuration
    std::string scenePath;
    ZMQCommunicator* communicator;
    CommandProcessor command_processor_;
    StateManager state_manager_;
    ActuatorController actuator_controller_;
    
    // Scene Entity Tracking
    std::vector<std::string> robotNames;
    std::vector<std::string> sensorNames;
    std::vector<std::string> actuatorNames;

    // Internal Utility Methods
    std::vector<std::string> RobotCollisionDetector(std::string& collision_robot);
    bool CheckNameForCollision(std::string name, std::string name2, std::string& collision_robot);
    void PrintAll();
};

#endif // STONEFISH_RL_H