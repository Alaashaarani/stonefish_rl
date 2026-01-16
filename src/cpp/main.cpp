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
double physics_frequency = 200; // frequencey used to compute physics (Number of times physics is computed per sec)  
double stonefish_frequency = 50; // stonefish steps Simulation frequency in Hz (should match your env params)
double rl_frequency = 10 ;  // rl frequency should match your enviroment 

int learning(void* data) {
    sf::SimulationApp& simApp = static_cast<LearningThreadData*>(data)->sim;
    sf::SimulationManager* simManager = simApp.getSimulationManager();
    StonefishRL* myManager = static_cast<StonefishRL*>(simManager);

    // Extract steps from command (default to 1 if not specified)
    int stonefish_steps = stonefish_frequency/rl_frequency;

    while (simApp.getState() == sf::SimulationState::NOT_READY)
    {
        SDL_Delay(10);
    }

    // Start the simulation (includes building the scenario)
    simApp.StartSimulation();
    std::string nextStepSim;

    while(nextStepSim != "EXIT")
    {   
        nextStepSim = myManager->RecieveInstructions(simApp);
        
        float time0 = simManager->getSimulationTime();


        if(nextStepSim == "CMD"){
            
            // Execute multiple simulation steps
            for(int i = 0; i < stonefish_steps; i++) {
                simApp.StepSimulation();
            }
        
            myManager->SendObservations(); // Send observations after all steps
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
    auto sf_dt = 1/stonefish_frequency; 

    if (argc < 7) { // Changed from 4 to 5
        std::cerr << "[ERROR] Usage (ALL STR): SCENE_PATH RESOURCES_PATH OBS_CONFIG_PATH ACTION_CONFIG_PATH PORT RESOLUTION " << std::endl;
        return 1;
    }

    std::string scene_path = argv[1]; 
    std::string resources_path = argv[2]; 
    std::string obser_conf_path = argv[3]; 
    std::string action_conf_path = argv[4]; 
    int port = std::stoi(argv[5]); // Parse the port
    int resolution = std::stoi(argv[6]); // Parse the resolution
    std::string graphical_arg = argv[7];
    bool graphical = graphical_arg == "True";
    sf::HelperSettings h;
    sf::RenderSettings r;
    r.windowW = resolution;
    r.windowH = resolution;
    
    
    sf::SimulationApp* app = nullptr;

    StonefishRL* simManager = new StonefishRL(scene_path, obser_conf_path, action_conf_path, physics_frequency, port); 
    
    if (graphical) {
        // Create the graphical simulation app
        app = new sf::GraphicalSimulationApp("STONEFISH RL"+std::to_string(port), resources_path, r, h, simManager);
    }else {
        // headless version
        app = new sf::ConsoleSimulationApp("DEMO", resources_path, simManager);
    }
    
    LearningThreadData data {*app}; 
    SDL_Thread* learningThread = SDL_CreateThread(learning, "learningThread", &data);

    std::cout << "[main] Port: " << port << " | Physics freq: " << stonefish_frequency << std::endl; 

    app->Run(true, false, sf::Scalar(sf_dt));
    SDL_WaitThread(learningThread, nullptr);
    delete app;
    return 0;
}