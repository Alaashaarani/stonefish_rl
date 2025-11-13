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
    
    // Initialization
    void initializeExtractors();
    
    // Field extraction system Used during initialization to map field types from the Json to extractor functions
    /* 
    This maps the extractor functions for robot fields. creating a reference to retrive the value based on field type.
    Example: "position.x" -> function to extract x position from robot
    */
    std::unordered_map<std::string, std::function<float(sf::SimulationManager*, const std::string&)>> robot_extractors_;
    // Sensors
    std::unordered_map<std::string, std::function<float(sf::SimulationManager*, const std::string&)>> sensor_extractors_;
    // Actuators
    std::unordered_map<std::string, std::function<float(sf::SimulationManager*, const std::string&)>> actuator_extractors_;
    
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
    // used to extract information from sensors
    /* 
    \param sim is the stonefish class
    \param name is the sensors name ex(ROBOT/Sensor, girona500/gps.......)
    \param expected_type 
    \param channel_index the position of this value in the msg(ex, 0,1,2,3,4,5 .... )
    */
    static float extractFromSensorType(sf::SimulationManager* sim, const std::string& name, 
                           sf::ScalarSensorType expected_type, int channel_index);

    
};

#endif // STATEMANAGER_H