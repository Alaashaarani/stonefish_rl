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


double physics_frequency = 200; // frequencey used to compute physics (Number of times physics is computed per sec)  
double sf_dt; 
double rl_freq; 

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
    std::string nextStepSim;

    int sim_step = static_cast<int> (1/(sf_dt*rl_freq));

    double time0= 0.0;
    while(nextStepSim != "EXIT")
    {   
        // std::cout << " SimulationTime " << myManager->getSimulationTime()-time0 << std::endl;
        nextStepSim = myManager->RecieveInstructions(simApp);
        if(nextStepSim == "CMD"){
            myManager->SendStates(); // Send states after all steps
            simApp.StepSimulation();

            // for(int i=0;i < sim_step;i++ ){
            //     simApp.StepSimulation();
            // }
        }      
        else{ // THis includes the RESET or others
            // time0 = myManager->getSimulationTime();
            simApp.StepSimulation();
        }                               
    }

    myManager->ExitRequest();
    return 0;
}


int main(int argc, char **argv) {

    if (argc < 8) { // Changed from 4 to 5
        std::cerr << "[ERROR] Usage (ALL STR): SCENE_PATH RESOURCES_PATH STATE_CONFIG_PATH ACTION_CONFIG_PATH PORT RESOLUTION GRAPHICAL_ENABLE STEP_TIME" << std::endl;
        return 1;
    }

    std::string scene_path = argv[1]; 
    std::string resources_path = argv[2]; 
    std::string state_conf_path = argv[3]; 
    std::string action_conf_path = argv[4]; 
    int port = std::stoi(argv[5]); // Parse the port
    int resolution = std::stoi(argv[6]); // Parse the resolution
    std::string graphical_arg = argv[7];
    rl_freq = std::stod(argv[8]); // Parse the dt if provided
    // if (rl_freq < 10){sf_dt = 1/(rl_freq*2); }else{ sf_dt = 1/rl_freq; }; 
    sf_dt = 1/rl_freq; 

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