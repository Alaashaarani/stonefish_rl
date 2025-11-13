#include "ActuatorController.h"

void ActuatorController::applyCommands(const std::unordered_map<std::string, 
                                      std::unordered_map<std::string, float>>& commands,
                                      sf::SimulationManager* sim) {
    unsigned int id = 0;
    sf::Actuator *actuator_ptr;

    while ((actuator_ptr = sim->getActuator(id++)) != nullptr) {
        std::string actuator_name = actuator_ptr->getName();
        auto actuator_commands = commands.find(actuator_name);
        if (actuator_commands == commands.end()) {
            continue;
        }

        switch (actuator_ptr->getType()) {
        case sf::ActuatorType::SERVO: {
            sf::Servo *servo = dynamic_cast<sf::Servo *>(actuator_ptr);
            if (servo) {
                controlServo(servo, actuator_commands->second);
            } else {
                std::cout << "[ActuatorController] Failed to convert " << actuator_name << " to SERVO" << std::endl;
            }
            break;
        }

        case sf::ActuatorType::THRUSTER: {
            sf::Thruster *thruster = dynamic_cast<sf::Thruster *>(actuator_ptr);
            if (thruster) {
                controlThruster(thruster, actuator_commands->second);
            } else {
                std::cout << "[ActuatorController] Failed to convert " << actuator_name << " to THRUSTER" << std::endl;
            }
            break;
        }

        default:
            std::cout << "[ActuatorController] Actuator type not supported: " << actuator_name << std::endl;
            break;
        }
    }
}

void ActuatorController::controlServo(sf::Servo* servo, const std::unordered_map<std::string, float>& actions) {
    for (const auto &[action, action_value] : actions) {
        if (action == "VELOCITY" || action == "TORQUE") {
            servo->setControlMode(sf::ServoControlMode::VELOCITY);
            servo->setDesiredVelocity(action_value);
            std::cout << "[ActuatorController] Set servo velocity: " << action_value << std::endl;
        }
        else if (action == "POSITION") {
            servo->setControlMode(sf::ServoControlMode::POSITION);
            servo->setDesiredPosition(action_value);
            std::cout << "[ActuatorController] Set servo position: " << action_value << std::endl;
        }
        else {
            std::cout << "[ActuatorController] Unknown command '" << action << "' for servo" << std::endl;
        }
    }
}

void ActuatorController::controlThruster(sf::Thruster* thruster, const std::unordered_map<std::string, float>& actions) {
    for (const auto &[action, action_value] : actions) {
        if (action == "VELOCITY" || action == "TORQUE") {
            thruster->setSetpoint(action_value);
            // debug print
            // std::cout << "[ActuatorController] Set thruster setpoint: " << action_value << std::endl;
        }
        else {
            std::cout << "[ActuatorController] Unknown command '" << action << "' for thruster" << std::endl;
        }
    }
}

void ActuatorController::printActuatorInfo(sf::SimulationManager* sim) {
    unsigned int id = 0;
    sf::Actuator *actuator_ptr;

    std::cout << "\n=== Actuator Information ===" << std::endl;
    while ((actuator_ptr = sim->getActuator(id++)) != nullptr) {
        std::cout << "Actuator: " << actuator_ptr->getName() << std::endl;
        
        if (actuator_ptr->getType() == sf::ActuatorType::SERVO) {
            sf::Servo *servo = dynamic_cast<sf::Servo *>(actuator_ptr);
            if (servo) {
                std::cout << "  Type: SERVO" << std::endl;
                std::cout << "  Position: " << servo->getPosition() << std::endl;
                std::cout << "  Velocity: " << servo->getVelocity() << std::endl;
            }
        }
        else if (actuator_ptr->getType() == sf::ActuatorType::THRUSTER) {
            sf::Thruster *thruster = dynamic_cast<sf::Thruster *>(actuator_ptr);
            if (thruster) {
                std::cout << "  Type: THRUSTER" << std::endl;
                std::cout << "  Setpoint: " << thruster->getSetpoint() << std::endl;
                std::cout << "  Thrust: " << thruster->getThrust() << std::endl;
            }
        }
        std::cout << std::endl;
    }
}