#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <string>
#include <vector>
#include <unordered_map>

// Simple observation specification
struct ObservationSpec {
    std::string entity_name;    // "girona", "imu_sensor", etc.
    std::string field_type;     // "position", "rotation", "velocity", "collision"
    std::string component;      // "x", "y", "z", "yaw", "binary"
    std::string output_name;    // "girona_position_x", "collision_flag"
};

// Action specification
struct ActionSpec {
    std::string actuator_name;  // "thruster_1", "servo_1"
    std::string action_type;    // "setpoint", "position", "velocity"
    std::string output_name;    // "thruster_1_setpoint"
};

// Robot reset information 
struct RobotResetInfo {
    std::string name;
    std::vector<float> position;  // [x, y, z]
    std::vector<float> rotation;  // [roll, pitch, yaw] or [x, y, z, w] for quaternion
};

// Configuration structures
struct ObservationConfig {
    std::vector<ObservationSpec> specs;
};

struct ActionConfig {
    std::vector<ActionSpec> specs;
};

struct SimulationConfig {
    ObservationConfig observation_config;
    ActionConfig action_config;
};

#endif // COMMON_TYPES_H