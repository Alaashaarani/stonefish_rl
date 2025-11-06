#include "StateManager.h"
#include <iostream>
#include <cmath>

StateManager::StateManager() {
    initializeExtractors();
    std::cout << "[StateManager] Dynamic StateManager initialized" << std::endl;
}

void StateManager::setObservationConfig(const ObservationConfig& config) {
    observation_config_ = config;
    observation_specs_ = config.specs;
    std::cout << "[StateManager] Observation config set with " << observation_specs_.size() << " specs" << std::endl;
    printObservationSpecs();
}

void StateManager::initializeExtractors() {
    // Robot field extractors - explicitly cast double to float
    robot_extractors_ = {
        {"position.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto robot = sim->getRobot(name);
            return robot ? static_cast<float>(robot->getTransform().getOrigin().x()) : NAN;
        }},
        {"position.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto robot = sim->getRobot(name);
            return robot ? static_cast<float>(robot->getTransform().getOrigin().y()) : NAN;
        }},
        {"position.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto robot = sim->getRobot(name);
            return robot ? static_cast<float>(robot->getTransform().getOrigin().z()) : NAN;
        }},
        {"rotation.roll", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto robot = sim->getRobot(name);
            if (!robot) return NAN;
            sf::Scalar yaw, pitch, roll;
            robot->getTransform().getRotation().getEulerZYX(yaw, pitch, roll);
            return static_cast<float>(roll);
        }},
        {"rotation.pitch", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto robot = sim->getRobot(name);
            if (!robot) return NAN;
            sf::Scalar yaw, pitch, roll;
            robot->getTransform().getRotation().getEulerZYX(yaw, pitch, roll);
            return static_cast<float>(pitch);
        }},
        {"rotation.yaw", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto robot = sim->getRobot(name);
            if (!robot) return NAN;
            sf::Scalar yaw, pitch, roll;
            robot->getTransform().getRotation().getEulerZYX(yaw, pitch, roll);
            return static_cast<float>(yaw);
        }},
        {"collision.binary", [this](sf::SimulationManager* sim, const std::string& name) -> float {
            return this->getCollisionFlag(sim, name);
        }}
    };
    
    // Add sensor and actuator extractors as needed...
    // For now, using placeholder extractors
    sensor_extractors_ = {
        {"value", [](sf::SimulationManager* sim, const std::string& name) -> float {
            // Generic sensor value extractor
            return 0.0f;
        }}
    };
    
    actuator_extractors_ = {
        {"setpoint", [](sf::SimulationManager* sim, const std::string& name) -> float {
            // Generic actuator value extractor
            return 0.0f;
        }}
    };
}

std::vector<float> StateManager::getObservationVector(sf::SimulationManager* sim) {
    std::vector<float> observations;
    observations.reserve(observation_specs_.size());
    
    for (const auto& spec : observation_specs_) {
        float value = NAN;
        
        // Determine entity type and extract accordingly
        if (spec.field_type == "collision") {
            value = getCollisionFlag(sim, spec.entity_name);
        }
        else if (findRobot(sim, spec.entity_name)) {
            value = extractRobotField(sim, spec);
        }
        else if (findSensor(sim, spec.entity_name)) {
            value = extractSensorField(sim, spec);
        }
        else if (findActuator(sim, spec.entity_name)) {
            value = extractActuatorField(sim, spec);
        }
        else {
            std::cerr << "[StateManager] WARNING: Entity not found: " << spec.entity_name << std::endl;
            value = 0.0f; // Default to 0 instead of NaN for robustness
        }
        
        observations.push_back(value);
    }
    
    return observations;
}

float StateManager::extractRobotField(sf::SimulationManager* sim, const ObservationSpec& spec) {
    std::string field_key = spec.field_type + "." + spec.component;
    auto extractor = robot_extractors_.find(field_key);
    
    if (extractor != robot_extractors_.end()) {
        try {
            return extractor->second(sim, spec.entity_name);
        } catch (const std::exception& e) {
            std::cerr << "[StateManager] ERROR extracting " << field_key << " from " 
                      << spec.entity_name << ": " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[StateManager] WARNING: No extractor for field: " << field_key << std::endl;
    }
    
    return 0.0f;
}

float StateManager::extractSensorField(sf::SimulationManager* sim, const ObservationSpec& spec) {
    // Implement sensor field extraction
    std::cerr << "[StateManager] WARNING: Sensor extraction not yet implemented for " 
              << spec.entity_name << std::endl;
    return 0.0f;
}

float StateManager::extractActuatorField(sf::SimulationManager* sim, const ObservationSpec& spec) {
    // Implement actuator field extraction  
    std::cerr << "[StateManager] WARNING: Actuator extraction not yet implemented for " 
              << spec.entity_name << std::endl;
    return 0.0f;
}

// Entity finding methods
sf::Robot* StateManager::findRobot(sf::SimulationManager* sim, const std::string& name) {
    unsigned int id = 0;
    sf::Robot* robot;
    while ((robot = sim->getRobot(id++)) != nullptr) {
        if (robot->getName() == name) return robot;
    }
    return nullptr;
}

sf::Sensor* StateManager::findSensor(sf::SimulationManager* sim, const std::string& name) {
    unsigned int id = 0;
    sf::Sensor* sensor;
    while ((sensor = sim->getSensor(id++)) != nullptr) {
        if (sensor->getName() == name) return sensor;
    }
    return nullptr;
}

sf::Actuator* StateManager::findActuator(sf::SimulationManager* sim, const std::string& name) {
    unsigned int id = 0;
    sf::Actuator* actuator;
    while ((actuator = sim->getActuator(id++)) != nullptr) {
        if (actuator->getName() == name) return actuator;
    }
    return nullptr;
}

void StateManager::updateRobotPosition(const std::vector<RobotResetInfo>& robot_info, sf::SimulationManager* sim) {
    for (const auto& info : robot_info) {
        positionSingleRobot(info, sim);
    }
}

void StateManager::positionSingleRobot(const RobotResetInfo& info, sf::SimulationManager* sim) {
    sf::Robot* robot = findRobot(sim, info.name);
    if (!robot) {
        std::cerr << "[StateManager] ERROR: Robot not found for reset: " << info.name << std::endl;
        return;
    }
    
    if (info.position.size() < 3) {
        std::cerr << "[StateManager] ERROR: Invalid position for robot " << info.name 
                  << ", expected 3 values, got " << info.position.size() << std::endl;
        return;
    }
    
    // Create transform
    sf::Transform tf;
    sf::Vector3 new_position(info.position[0], info.position[1], info.position[2]);
    tf.setOrigin(new_position);
    
    // Handle rotation (Euler angles or quaternion)
    if (info.rotation.size() >= 3) {
        // Assume Euler angles [roll, pitch, yaw]
        sf::Quaternion rotation(info.rotation[0], info.rotation[1], info.rotation[2]);
        tf.setRotation(rotation);
    } else if (info.rotation.size() == 4) {
        // Quaternion [x, y, z, w]
        sf::Quaternion rotation(info.rotation[0], info.rotation[1], info.rotation[2], info.rotation[3]);
        tf.setRotation(rotation);
    }
    // If no rotation provided, keep current rotation
    
    robot->Respawn(sim, tf);
    std::cout << "[StateManager] Repositioned robot: " << info.name << std::endl;
}

float StateManager::getCollisionFlag(sf::SimulationManager* sim, const std::string& robot_name) {
    // Simplified collision detection - implement your logic here
    // Return 1.0 for collision, 0.0 for no collision
    // This is a placeholder - you'll need to implement proper collision detection
    return 0.0f;
}

std::vector<std::string> StateManager::getObservationNames() const {
    std::vector<std::string> names;
    for (const auto& spec : observation_specs_) {
        names.push_back(spec.output_name);
    }
    return names;
}

void StateManager::printObservationSpecs() const {
    std::cout << "[StateManager] Observation Specifications:" << std::endl;
    for (const auto& spec : observation_specs_) {
        std::cout << "  " << spec.output_name << " <- " << spec.entity_name 
                  << "." << spec.field_type << "." << spec.component << std::endl;
    }
}