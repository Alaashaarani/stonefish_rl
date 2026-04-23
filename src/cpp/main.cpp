#include "StonefishRL.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <Stonefish/core/GraphicalSimulationApp.h>
#include <Stonefish/core/ConsoleSimulationApp.h>
#include <Stonefish/core/SimulationApp.h>
#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/sensors/scalar/RotaryEncoder.h>
#include <Stonefish/sensors/scalar/IMU.h>
#include <Stonefish/sensors/ScalarSensor.h>
#include <Stonefish/sensors/Sample.h>
#include <Stonefish/actuators/Motor.h>
#include <Stonefish/actuators/Servo.h>
#include <Stonefish/actuators/Thruster.h>

#include <Stonefish/StonefishCommon.h>


struct LearningThreadData
{
    sf::SimulationApp& sim;
};

/*
These values should match with the values in the enviroment you are using
*/
double physics_frequency = 300; // frequencey used to compute physics (Number of times physics is computed per sec)  
double sf_dt = 0.1; // do 10 steps per sec
double rl_observation_freq = 10; // default value, can be specified through the config file yaml


int learning(void* data) {
    sf::SimulationApp& simApp = static_cast<LearningThreadData*>(data)->sim;
    sf::SimulationManager* simManager = simApp.getSimulationManager();
    StonefishRL* myManager = static_cast<StonefishRL*>(simManager);

    // Extract steps from command (default to 1 if not specified)
    while (simApp.getState() == sf::SimulationState::NOT_READY)
    {
        SDL_Delay(10);
    }

    // Start the simulation (includes building the scenario)
    simApp.StartSimulation();
    std::string nextStepSim;

    // checking simulation speed and executing the observation frequency accordingly 
    double time0 = myManager->getSimulationTime();
    for (int i = 0; i < 10; i++) {
        simApp.StepSimulation();
    }   
    double sim_speed = (myManager->getSimulationTime() - time0)/10;
    int sim_steps = (1/sim_speed)/rl_observation_freq;  // the simulation steps this amount before calling the RL agent 
    
    // wait for 2 sec


    while(nextStepSim != "EXIT")
    {   
        nextStepSim = myManager->RecieveInstructions(simApp);
        
        if(nextStepSim == "CMD"){
            
            
            myManager->SendStates(); // Send states after all steps
            for (int i = 0;i < sim_steps;i++) {
                simApp.StepSimulation();
            }
                
        }      
        else if (nextStepSim == "RESET"){
            simApp.StepSimulation();
        }                                               
    }

    std::cout << "[INFO] Learning thread finished." << std::endl;
    myManager->ExitRequest();
    return 0;
}


int main(int argc, char **argv) {

    if (argc < 8) { // Changed from 4 to 5
        std::cerr << "[ERROR] Usage (ALL STR): SCENE_PATH RESOURCES_PATH OBS_CONFIG_PATH ACTION_CONFIG_PATH PORT RESOLUTION " << std::endl;
        return 1;
    }
    std::cout << "[Main] Number of arguments: " << argc << std::endl; 

    std::string scene_path = argv[1]; 
    std::string resources_path = argv[2]; 
    std::string state_conf_path = argv[3]; 
    std::string action_conf_path = argv[4]; 
    int port = std::stoi(argv[5]); // Parse the port
    int resolution = std::stoi(argv[6]); // Parse the resolution
    std::string graphical_arg = argv[7];
    std::cout << "[Main] argument Check: " << argv[8] << std::endl; 
    rl_observation_freq = std::stod(argv[8]);
    std::cout << "[Main] argument Check: " << argv[9] << std::endl; 
    sf_dt = std::stod(argv[9]); // Parse the dt if provided


    // std::cout << "[MAIN] Scene Path: " << scene_path << std::endl;
    // std::cout << "[MAIN] Resources Path: " << resources_path << std::endl;
    std::cout << "[MAIN] Using dt: " << sf_dt << " seconds." << std::endl;
    // std::cout << "[MAIN] Using arg8: " << argv[8] << " seconds." << std::endl;
    // std::cout << "[MAIN] Using graphical interface: " << graphical_arg << std::endl;
    // std::cout << "[MAIN] Using resolution: " << resolution << std::endl;
    

    bool graphical = graphical_arg == "True";
    sf::HelperSettings h;
    sf::RenderSettings r;
    r.windowW = resolution;
    r.windowH = resolution;
    
    
    sf::SimulationApp* app = nullptr;

    StonefishRL* simManager = new StonefishRL(scene_path, state_conf_path, action_conf_path, physics_frequency, port); 
    
    if (graphical) {
        // Create the graphical simulation app
        app = new sf::GraphicalSimulationApp("STONEFISH RL"+std::to_string(port), resources_path, r, h, simManager);
    }else {
        // headless version
        app = new sf::ConsoleSimulationApp("DEMO", resources_path, simManager);
    }
    
    LearningThreadData data {*app}; 
    SDL_Thread* learningThread = SDL_CreateThread(learning, "learningThread", &data);



    app->Run(true, false, sf::Scalar(sf_dt));
    SDL_WaitThread(learningThread, nullptr);
    delete app;
    return 0;
}