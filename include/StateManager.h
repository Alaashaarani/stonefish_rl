#ifndef STATEMANAGER_H
#define STATEMANAGER_H

#include "CommonTypes.h"
#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/core/Robot.h>
#include <Stonefish/sensors/Sample.h>
#include <Stonefish/sensors/Sensor.h>
#include <Stonefish/sensors/ScalarSensor.h>
#include <Stonefish/actuators/Actuator.h>
#include <Stonefish/actuators/Servo.h>
#include <Stonefish/actuators/Thruster.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <functional>
class StateManager {
public:
    StateManager();
    
    // Configuration
    void setObservationConfig(const ObservationConfig& config);
    
    // Observation methods
    std::vector<float> getObservationVector(sf::SimulationManager* sim);
    std::vector<std::string> getObservationNames() const;
    
    // Robot management
    void updateRobotPosition(const std::vector<RobotResetInfo>& robot_info, sf::SimulationManager* sim);
    
    // Utility
    size_t getObservationSize() const { return observation_specs_.size(); }
    void printObservationSpecs() const;

private:
    ObservationConfig observation_config_;
    std::vector<ObservationSpec> observation_specs_;
    
    // Field extraction system
    std::unordered_map<std::string, std::function<float(sf::SimulationManager*, const std::string&)>> robot_extractors_;
    std::unordered_map<std::string, std::function<float(sf::SimulationManager*, const std::string&)>> sensor_extractors_;
    std::unordered_map<std::string, std::function<float(sf::SimulationManager*, const std::string&)>> actuator_extractors_;
    
    // Initialization
    void initializeExtractors();
    
    // Extraction methods
    float extractRobotField(sf::SimulationManager* sim, const ObservationSpec& spec);
    float extractSensorField(sf::SimulationManager* sim, const ObservationSpec& spec);
    float extractActuatorField(sf::SimulationManager* sim, const ObservationSpec& spec);
    
    // Entity finding
    sf::Robot* findRobot(sf::SimulationManager* sim, const std::string& name);
    sf::Sensor* findSensor(sf::SimulationManager* sim, const std::string& name);
    sf::Actuator* findActuator(sf::SimulationManager* sim, const std::string& name);
    
    // Collision detection
    float getCollisionFlag(sf::SimulationManager* sim, const std::string& robot_name);

    // Robot positioning helper
    void positionSingleRobot(const RobotResetInfo& info, sf::SimulationManager* sim);
};

#endif // STATEMANAGER_H