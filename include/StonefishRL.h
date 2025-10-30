#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/core/SimulationApp.h>

#include <Stonefish/actuators/Actuator.h>
#include <Stonefish/actuators/Thruster.h>

#include <Stonefish/sensors/Sensor.h> 

#include <zmq.hpp> // Per la comunicació amb Python

class StonefishRL : public sf::SimulationManager {
public:

// IMPORTANT: Whenever a new struct and/or parameter is added to send to Python
// (e.g. a new sensor whose readings you want to include), you must also update
// the functions 'InfoObjectToJson', 'FillWithNanInfoObject', and 'GetStateScene',
// because those are responsible for sending data to Python.
    
    // Contains all information for one scene object
    struct InfoObject 
    {
        std::string name;
    
        float angle;
        float pressure;
    
        std::vector<float> position;
        std::vector<float> rotation;
        
        std::vector<float> linear_velocity;
        std::vector<float> linear_acceleration;
        std::vector<float> angular_velocity;
    
        std::vector<float> force; 
        std::vector<float> torque; 

        std::vector<float> gps; // Indices: [0] latitude, [1] longitude, [2] North, [3] East
        std::vector<std::string> collisions; //includes names of objects colliding with this one
    };

    // Holds all objects in the scene
    struct StateScene
    {
        std::vector<InfoObject> observations;
    };

    struct RobotInfo 
    {
    std::string name;
    unsigned int id;
    sf::Robot* pointer; // Optional: store pointer if you need direct access
    };

    


    // Constructor
    StonefishRL(const std::string& path, double frequency);

    // Methods for the RL interface
    std::string RecieveInstructions(sf::SimulationApp& simApp);
    
    void SendObservations();

    // detect collisions in the simulation
    std::vector<std::string> RobotCollisionDetector(std::string& collision_robot); // checks collisions with a specific robot and returns list of objects colliding with it. 

    bool CheckNameForCollision(std::string name, std::string name2, std::string& collision_robot);

    void ApplyCommands(const std::string& str_cmds);
    
    StateScene GetStateScene();

    std::vector<InfoObject> ParseResetCommand(const std::string& str);

    void ApplyReset(const std::vector<InfoObject>& objects);

    void ParseCommandsAndObservations(const std::string& str);

    void SetRobotPosition(const std::vector<InfoObject>& obj);
    
    void ExitRequest();

    bool ObjImportantForObs(const std::string& objName) const;

    void FillWithNanInfoObject(InfoObject& pose);

    void PrintAll();

    // saving robot and other items names
    const std::vector<std::string>& getRobotNames() const { return robotNames; }
    const std::vector<std::string>& getSensorNames() const { return sensorNames; }
    const std::vector<std::string>& getActuatorNames() const { return actuatorNames; }

    // Convert to JSON to send to Python
    std::string InfoObjectToJson(const InfoObject& pose); 
    std::string EscapeJson(const std::string& s);
    std::string SerializeScene(const std::vector<InfoObject>& poses);
    std::string SafeFloat(float val);

protected:
    // Override of the method 'sf::SimulationManager::BuildScenario'
    void BuildScenario() override;

private:
    
    void InitializeZMQ();

    std::string scenePath;

    std::unordered_set<std::string> relevant_obs_names_;

    std::map<std::string, std::map<std::string, float>> commands_; // Values to apply to actuators
    StateScene current_state_; // Represents the latest observed state of the environment
    
    //storage members
    std::vector<std::string> robotNames;
    std::vector<std::string> sensorNames;
    std::vector<std::string> actuatorNames;

    zmq::context_t context; // ZeroMQ context
    zmq::socket_t socket;   // ZeroMQ socket
};
