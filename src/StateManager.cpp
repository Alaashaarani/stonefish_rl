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
        // Generic sensor value extractors (fallback for any scalar sensor)
        {"sensor.value", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto sensor_ptr = sim->getSensor(name);
            sf::ScalarSensor *sensor = dynamic_cast<sf::ScalarSensor *>(sensor_ptr);
            return sensor && sensor->getNumOfChannels() > 0 ? 
                static_cast<float>(sensor->getLastSample().getValue(0)) : NAN;
        }},
        {"sensor.channel0", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto sensor_ptr = sim->getSensor(name);
            sf::ScalarSensor *sensor = dynamic_cast<sf::ScalarSensor *>(sensor_ptr);
            return sensor && sensor->getNumOfChannels() > 0 ? 
                static_cast<float>(sensor->getLastSample().getValue(0)) : NAN;
        }},
        {"sensor.channel1", [](sf::SimulationManager* sim, const std::string& name) -> float {
            auto sensor_ptr = sim->getSensor(name);
            sf::ScalarSensor *sensor = dynamic_cast<sf::ScalarSensor *>(sensor_ptr);
            return sensor && sensor->getNumOfChannels() > 1 ? 
                static_cast<float>(sensor->getLastSample().getValue(1)) : NAN;
        }},

        // ENCODER SENSOR
        {"encoder.angle", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ENCODER, 0);
        }},
        {"encoder.angular_velocity", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ENCODER, 1);
        }},

        // ODOMETRY SENSOR - Complete mapping
        {"odom.position.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 0);
        }},
        {"odom.position.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 1);
        }},
        {"odom.position.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 2);
        }},
        {"odom.linear_velocity.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 3);
        }},
        {"odom.linear_velocity.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 4);
        }},
        {"odom.linear_velocity.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 5);
        }},
        {"odom.rotation.roll", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 6);
        }},
        {"odom.rotation.pitch", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 7);
        }},
        {"odom.rotation.yaw", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 8);
        }},
        {"odom.angular_velocity.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 10);
        }},
        {"odom.angular_velocity.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 11);
        }},
        {"odom.angular_velocity.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::ODOM, 12);
        }},

        // PRESSURE SENSOR
        {"pressure.value", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::PRESSURE, 0);
        }},
        {"pressure.depth", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::PRESSURE, 0);
        }},

        // FORCE_TORQUE SENSOR
        {"ft.force.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::FT, 0);
        }},
        {"ft.force.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::FT, 1);
        }},
        {"ft.force.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::FT, 2);
        }},
        {"ft.torque.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::FT, 3);
        }},
        {"ft.torque.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::FT, 4);
        }},
        {"ft.torque.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::FT, 5);
        }},

        // GPS SENSOR
        {"gps.latitude", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::GPS, 0);
        }},
        {"gps.longitude", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::GPS, 1);
        }},
        {"gps.north", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::GPS, 2);
        }},
        {"gps.east", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::GPS, 3);
        }},

        // IMU SENSOR - Complete mapping
        {"imu.rotation.roll", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 0);
        }},
        {"imu.rotation.pitch", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 1);
        }},
        {"imu.rotation.yaw", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 2);
        }},
        {"imu.angular_velocity.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 3);
        }},
        {"imu.angular_velocity.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 4);
        }},
        {"imu.angular_velocity.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 5);
        }},
        {"imu.linear_acceleration.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 6);
        }},
        {"imu.linear_acceleration.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 7);
        }},
        {"imu.linear_acceleration.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::IMU, 8);
        }},

        // DVL SENSOR (Doppler Velocity Log) - if available in your Stonefish version
        {"dvl.velocity.x", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::DVL, 0);
        }},
        {"dvl.velocity.y", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::DVL, 1);
        }},
        {"dvl.velocity.z", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::DVL, 2);
        }},
        {"dvl.altitude", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::DVL, 3);
        }},

        // PROFILER SENSOR - if available
        {"profiler.range", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::PROFILER, 0);
        }},

        // MULTIBEAM SENSOR - if available  
        {"multibeam.range", [](sf::SimulationManager* sim, const std::string& name) -> float {
            return extractFromSensorType(sim, name, sf::ScalarSensorType::MULTIBEAM, 0);
        }}
    };
    
    actuator_extractors_ = {
        {"setpoint", [](sf::SimulationManager* sim, const std::string& name) -> float {
            // Generic actuator value extractor
            return 0.0f;
        }}
    };
}

// Helper function
float StateManager::extractFromSensorType(sf::SimulationManager* sim, const std::string& name, 
                           sf::ScalarSensorType expected_type, int channel_index) {
    auto sensor_ptr = sim->getSensor(name);
    sf::ScalarSensor* sensor = dynamic_cast<sf::ScalarSensor*>(sensor_ptr);
    if (!sensor || sensor->getScalarSensorType() != expected_type) 
        return NAN;
    if (channel_index >= sensor->getNumOfChannels()) 
        return NAN;
    return static_cast<float>(sensor->getLastSample().getValue(channel_index));
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
    std::string field_key = spec.field_type + "." + spec.component;
    auto extractor = sensor_extractors_.find(field_key);
    if (extractor != sensor_extractors_.end()) {
        try {
            return extractor->second(sim, spec.entity_name);
        } catch (const std::exception& e) {
            std::cerr << "[StateManager] ERROR extracting " << field_key << " from " 
                      << spec.entity_name << ": " << e.what() << std::endl;
        }
    } else {
        std::cerr << "[StateManager] WARNING: No extractor for field: " << field_key << std::endl;
    }
    // // Implement sensor field extraction
    // std::cerr << "[StateManager] WARNING: Sensor extraction not yet implemented for " 
    //           << spec.entity_name << std::endl;
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
        // debug
        // std::cout << "[StateManager] Checking sensor: " << sensor->getName() << " at value" << id << std::endl;
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