#include "StonefishRL.h"
#include <iostream>
#include <sstream>

// Constructor
StonefishRL::StonefishRL(const std::string &path, const std::string &observation_conf_path,const std::string &action_conf_path, double frequency)
    : sf::SimulationManager(frequency),
      scenePath(path),
      communicator(nullptr)
{
    communicator = new ZMQCommunicator("tcp://*:5555");
    std::cout << "[StonefishRL] Initialized with scene: " << scenePath << std::endl;

     // Load observation configuration
    ConfigLoader loader;
    ObservationConfig config = loader.loadFromFile(observation_conf_path);
    state_manager_.setObservationConfig(config);
    
    std::cout << "[StonefishRL] Initialized with scene: " << scenePath << std::endl;

}

std::string StonefishRL::RecieveInstructions(sf::SimulationApp& simApp) {
    zmq::message_t request;
    auto result = communicator->receive(request, zmq::recv_flags::none);

    std::string cmd = request.to_string();
    // debug output
    // std::cout << "[StonefishRL] Received: (" << result << " bytes) in command: " << cmd << std::endl;

    int pos = cmd.find(":");
    std::string prefix = cmd.substr(0, pos);
    cmd = cmd.substr(pos + 1);

    if (prefix == "RESET") {
        // Updated: Use RobotResetInfo instead of InfoObject
        std::vector<RobotResetInfo> command_data = command_processor_.parseResetCommand(cmd);
        state_manager_.updateRobotPosition(command_data, this);
        
        // Send observations using new vector approach
        SendObservations();
        // std::cout << "[StonefishRL] Received RESET command\n";
        return "RESET";
    }
    else if (prefix == "EXIT") {
        std::cout << "[StonefishRL] Received EXIT command\n";
        communicator->sendJson("EXIT OK");
        return "EXIT";
    }
    else if (prefix == "CMD") {
        command_processor_.parseActionCommands(cmd);
        ApplyCommands(cmd);
        return "CMD";
    }
    else {
        std::cout << "[StonefishRL] Unknown command prefix: " << prefix << std::endl;
        return "INVALID";
    }
}


void StonefishRL::SendObservations() {
    // Get observation vector from new StateManager
    std::vector<float> observations = state_manager_.getObservationVector(this);
    
    // Convert to JSON array for sending
    std::string obs_json = "[";
    for (size_t i = 0; i < observations.size(); ++i) {
        obs_json += std::to_string(observations[i]);
        if (i < observations.size() - 1) obs_json += ",";
    }
    obs_json += "]";
    
    communicator->sendJson(obs_json);
    // Debug output
    // std::cout << "[StonefishRL] Sent observation vector: " << observations.size() << " elements" << std::endl;
}

void StonefishRL::ApplyCommands(const std::string& str_cmds) {
    const auto& commands = command_processor_.getCommands();
    actuator_controller_.applyCommands(commands, this);
    // debug output
    // std::cout << "[StonefishRL] Applied commands to " << commands.size() << " actuators" << std::endl;
}

void StonefishRL::BuildScenario() {
    std::cout << "[StonefishRL] Building scenario from: " << scenePath << std::endl;
    sf::ScenarioParser parser(this);

    if (!parser.Parse(scenePath)) {
        std::cerr << "[StonefishRL] Error loading scenario: " << scenePath << "\n";
        for (const auto &msg : parser.getLog()) {
            std::cerr << "[ScenarioParser] " << msg.text << "\n";
        }
        return;
    }

    // Clear previous data
    robotNames.clear();
    sensorNames.clear();
    actuatorNames.clear();

    // Store entity names for reference
    unsigned int id = 0;
    sf::Sensor* sensor_ptr;
    while ((sensor_ptr = getSensor(id++)) != nullptr) {
        sensorNames.push_back(sensor_ptr->getName());
    }

    id = 0;
    sf::Actuator* actuator_ptr;
    while ((actuator_ptr = getActuator(id++)) != nullptr) {
        actuatorNames.push_back(actuator_ptr->getName());
    }

    id = 0;
    sf::Robot* robot_ptr;
    while ((robot_ptr = getRobot(id++)) != nullptr) {
        robotNames.push_back(robot_ptr->getName());
    }

    // Force StateManager to rebuild cache

    std::cout << "[StonefishRL] Scenario loaded successfully. Found: " 
              << robotNames.size() << " robots, "
              << sensorNames.size() << " sensors, "
              << actuatorNames.size() << " actuators\n";
}


std::vector<std::string> StonefishRL::RobotCollisionDetector(std::string& collision_robot)
{
    sf::Entity* entA;
    sf::Entity* entB; 
    std::vector<std::string> collision_list;
    int i=0;
    if  (collision_robot.find("girona") == std::string::npos) {
        return collision_list; // only check collisions for the girona robot
    }
    // std::cout << "Checking for collisions..." << std::endl;

    btDispatcher* dispatcher = getDynamicsWorld()->getDispatcher();
    int numManifolds = dispatcher->getNumManifolds();

    for(int i=0; i<numManifolds; ++i)
        {
            btPersistentManifold* contactManifold = dispatcher->getManifoldByIndexInternal(i);
            if(contactManifold->getNumContacts() == 0)
                continue;

            btCollisionObject* coA = (btCollisionObject*)contactManifold->getBody0();
            btCollisionObject* coB = (btCollisionObject*)contactManifold->getBody1();
            entA = (sf::Entity*)coA->getUserPointer();
            entB = (sf::Entity*)coB->getUserPointer();
        
            if (CheckNameForCollision(entA->getName(), entB->getName(), collision_robot))
            {   
                std::string collision_info = entA->getName() +"AND"+ entB->getName();
                collision_list.push_back(collision_info);
            }
        };

        // std::cout << "Collisions detected with: ";
        // for (const auto& name : collision_list)
        // {
        //     std::cout << name << " ";
        // }
        // std::cout << collision_list.size() << std::endl;

    return collision_list;
}


bool StonefishRL::CheckNameForCollision(std::string name, std::string name2, std::string& collision_robot)
{
    std::string obj1, obj2;
    std::vector<std::string> collision_target = {"Tank"}; // we will add the parts of the docking station later     
    // Extract robot names from both input strings
    
    if (name.find(collision_robot) != std::string::npos) {
        obj1 = name;
        // std::cout << "Collision with target detected: " << name << std::endl;
    }

    for (const auto& part : collision_target)
    {
        
        if (name2.find(part) != std::string::npos) {
                obj2 = name2;
                // std::cout << "Collision with target detected: " << name2 << std::endl;

            }
    }
    // check if the robot is coliding with the tank
    // Check if we found two DIFFERENT robots
    return (!obj1.empty() && !obj2.empty() && obj1 != obj2);
}

void StonefishRL::ExitRequest() {
    // socket.close();
    // context.close();
    delete communicator;

    std::cout << "[INFO] Simulation finished." << std::endl;
    std::exit(0);
}

// No functional purpose, just prints values to console
void StonefishRL::PrintAll()
{
    sf::Sensor *sensor_ptr;
    sf::Robot *robot_ptr;
    sf::Actuator *actuator_ptr;

    unsigned int id = 0;
    std::cout << "\n ROBOT INFO: \n";
    while ((robot_ptr = getRobot(id++)) != nullptr)
    {
        std::cout << robot_ptr->getName() << std::endl;
        sf::Vector3 origin = robot_ptr->getTransform().getOrigin();
        std::cout << "[ROBOT INFORMATION CHANNEL] POSICIO X: " << origin.getX() << std::endl;
        std::cout << "[ROBOT INFORMATION CHANNEL] POSICIO Y: " << origin.getY() << std::endl;
        std::cout << "[ROBOT INFORMATION CHANNEL] POSICIO Z: " << origin.getZ() << std::endl;
    }

    id = 0;
    std::cout << "\n ACTUATOR INFO: \n";
    while ((actuator_ptr = getActuator(id++)) != nullptr)
    {
        std::cout << actuator_ptr->getName() << std::endl;
        if (actuator_ptr->getType() == sf::ActuatorType::SERVO)
        {   
            std::cout << "SERVO\n";
            sf::Servo *servo_ptr = dynamic_cast<sf::Servo *>(actuator_ptr);
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Angle: " << servo_ptr->getPosition() << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Angular velocity: " << servo_ptr->getVelocity() << std::endl;;
        }
        else if (actuator_ptr->getType() == sf::ActuatorType::THRUSTER)
        {   
            std::cout << "THRUSTER\n";
            sf::Thruster *thruster_ptr = dynamic_cast<sf::Thruster *>(actuator_ptr);
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Is propeller right-handed? " << thruster_ptr->isPropellerRight() << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Propeller diameter: " << thruster_ptr->getPropellerDiameter() << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Angular Velocity: " << thruster_ptr->getOmega() << " [rad/s]" << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Angle: " << thruster_ptr->getAngle() << " [rad]" << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Torque: " << thruster_ptr->getTorque() << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Thrust: " << thruster_ptr->getThrust() << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Set Point: " << thruster_ptr->getSetpoint() << std::endl;
            std::cout << "[ACTUATOR INFORMATION CHANNEL] Set Point Limit: " << thruster_ptr->getSetpointLimit() << std::endl;
        }
    }

    id = 0;
    std::cout << "\n SENSOR INFO: \n";
    while ((sensor_ptr = getSensor(id++)) != nullptr)
    {
        if (sensor_ptr->getType() == (sf::SensorType::JOINT) || sensor_ptr->getType() == sf::SensorType::LINK) {
            std::cout << sensor_ptr->getName() << std::endl;
            sf::ScalarSensor *scalar_sensor = dynamic_cast<sf::ScalarSensor *>(sensor_ptr);
            for (unsigned int i = 0; i < scalar_sensor->getNumOfChannels(); i++)
            {
                std::string channel_name = scalar_sensor->getSensorChannelDescription(i).name;
                float value = scalar_sensor->getLastSample().getValue(i);
                std::cout << "[SENSOR INFORMATION CHANNEL] " << channel_name << ": " << value << std::endl;    
            }
        }
    }
}
