#ifndef ACTUATORCONTROLLER_H
#define ACTUATORCONTROLLER_H

#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/actuators/Actuator.h>
#include <Stonefish/actuators/Servo.h>
#include <Stonefish/actuators/Thruster.h>
#include <unordered_map>
#include <iostream>

class ActuatorController {
public:
    ActuatorController() = default;
    
    void applyCommands(const std::unordered_map<std::string, 
                      std::unordered_map<std::string, float>>& commands,
                      sf::SimulationManager* sim);
    
    void printActuatorInfo(sf::SimulationManager* sim);

private:
    void controlServo(sf::Servo* servo, const std::unordered_map<std::string, float>& actions);
    void controlThruster(sf::Thruster* thruster, const std::unordered_map<std::string, float>& actions);
};

#endif // ACTUATORCONTROLLER_H
