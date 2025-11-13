#ifndef STONEFISH_RL_H
#define STONEFISH_RL_H

#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/sensors/Sample.h>
#include <Stonefish/core/SimulationApp.h>
#include <Stonefish/core/ScenarioParser.h>
#include <Stonefish/entities/FeatherstoneEntity.h>
#include <Stonefish/entities/SolidEntity.h>
#include <Stonefish/core/Robot.h>
#include <Stonefish/sensors/Sensor.h>
#include <Stonefish/actuators/Actuator.h>
#include "ZMQCommunicator.h"
#include "CommandProcessor.h"
#include "StateManager.h"
#include "ConfigLoader.h" 
#include "ActuatorController.h"
#include "CommonTypes.h"
#include <vector>
#include <string>

class StonefishRL : public sf::SimulationManager {
public:
    StonefishRL(const std::string &path, const std::string &observation_conf_path, const std::string &action_conf_path, double frequency);
    
    std::string RecieveInstructions(sf::SimulationApp& simApp);
    void SendObservations();
    void ApplyCommands(const std::string& str_cmds);
    void BuildScenario();
    void ExitRequest();

private:
    std::string scenePath;
    ZMQCommunicator* communicator;
    CommandProcessor command_processor_;
    StateManager state_manager_;
    ActuatorController actuator_controller_;
    
    std::vector<std::string> robotNames;
    std::vector<std::string> sensorNames;
    std::vector<std::string> actuatorNames;

    // StonefishRL specific methods that remain
    std::vector<std::string> RobotCollisionDetector(std::string& collision_robot);
    bool CheckNameForCollision(std::string name, std::string name2, std::string& collision_robot);
    void PrintAll();
};

#endif // STONEFISH_RL_H