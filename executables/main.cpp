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
double stonefish_frequency = 50; // stonefish steps Simulation frequency in Hz (should match youe env params)
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

    auto sf_dt = 1/stonefish_frequency; // sf_step time in simulation

    if (argc < 4) {
        std::cerr << "[ERROR] Arg input should be, SCENE_PATH, OBS_CONFIG_PATH, ACTION_CONFIG_PATH" << std::endl;
        return 1;
    }

    std::string scene_path = argv[1]; 
    std::string resources_path = argv[2]; 
    std::string obser_conf_path = argv[3]; 
    std::string action_conf_path = argv[4]; 

    sf::HelperSettings h;
    sf::RenderSettings r;
    r.windowW = 900;
    r.windowH = 600;
    
    StonefishRL* simManager = new StonefishRL(scene_path, obser_conf_path, action_conf_path, physics_frequency); // Create the StonefishRL simulation manager

    sf::GraphicalSimulationApp app("DEMO STONEFISH RL", resources_path, r, h, simManager);
    //sf::ConsoleSimulationApp app("DEMO STONEFISH RL", scene_path, simManager);

    LearningThreadData data {app}; // is a struct that holds a reference to the sim app
    SDL_Thread* learningThread = SDL_CreateThread(learning, "learningThread", &data);
    
    std::cout << "[main] Physics frequencey is: " << stonefish_frequency << " and sf_dt: " << sf_dt << std::endl; 

    app.Run(false, false, sf::Scalar(sf_dt));
    // app.Run(false, false, sf::Scalar(0));
    SDL_WaitThread(learningThread, nullptr);
    return 0;
}