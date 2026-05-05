#ifndef COMMON_TYPES_H
#define COMMON_TYPES_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>

// Simple state specification
struct StateSpec {
    std::string entity_name;    // "girona", "imu_sensor", etc.
    std::string field_type;     // "position", "rotation", "velocity", "collision"
    std::string component;      // "x", "y", "z", "yaw", "binary"
    std::string entity_1;
    std::string entity_2;
    std::string output_name;    // "girona_position_x", "collision_flag"

};

// Action specification
struct ActionSpec {
    std::string actuator_name;  // "thruster_1", "servo_1"
    std::string action_type;    // "setpoint", "position", "velocity"
    std::string output_name;    // "thruster_1_setpoint"
};

// Robot reset information 
struct ResetInfo {
    std::string name;
    std::vector<float> position;  // [x, y, z]
    std::vector<float> rotation;  // [roll, pitch, yaw] or [x, y, z, w] for quaternion
    std::vector<float> velocity;  // [vx, vy, vz]
    std::vector<float> angular_velocity; // [wx, wy, wz]
    std::vector<float> current; // in case of using currents 


    void print() const{
        std::cout << "Name: " << name << "\n";
        std::cout << "position: ";
        for (auto x : position)
          std::cout << x << " ";
        std::cout << "\n";

                std::cout << "rotation: ";
        for (auto x : rotation)
          std::cout << x << " ";
        std::cout << "\n";

        //         std::cout << "velocity: ";
        // for (auto x : velocity)
        //   std::cout << x << " ";
        // std::cout << "\n";

        //         std::cout << "angular_velocity: ";
        // for (auto x : angular_velocity)
        //   std::cout << x << " ";
        // std::cout << "\n";

                std::cout << "current: ";
        for (auto x : current)
          std::cout << x << " ";
        std::cout << "\n";
    }
};


// Configuration structures
struct StateConfig {
    std::vector<StateSpec> specs;
};

struct ActionConfig {
    std::vector<ActionSpec> specs;
};

struct SimulationConfig {
    StateConfig state_config;
    ActionConfig action_config;
};

#endif // COMMON_TYPES_H