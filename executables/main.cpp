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


int learning(void* data) {
    sf::SimulationApp& simApp = static_cast<LearningThreadData*>(data)->sim;
    sf::SimulationManager* simManager = simApp.getSimulationManager();
    StonefishRL* myManager = static_cast<StonefishRL*>(simManager);

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

        // TODO: l'StepSimulation va a 1000 Hz mentre que els agents van a 5-20Hz
        // si cada vegada que enviem un StepSimulation fem un SendObservations
        // es mooolt lent!
        if(nextStepSim == "CMD")
        {
            simApp.StepSimulation();
            myManager->SendObservations();
        }
        else if (nextStepSim == "RESET"){
            simApp.StepSimulation();
        }
       
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));   // Si el treiem (comentem aquesta linea) va el màxim de ràpid.
                                                                     // Si el posem a 1, va approx. a real-time                                                                     
    }

    std::cout << "[INFO] Learning thread finished." << std::endl;
    myManager->ExitRequest();
    return 0;
}


int main(int argc, char **argv) {

    double frequency = 200; // Simulation frequency in Hz
    
    if (argc < 4) {
        std::cerr << "[ERROR] Arg input should be, SCENE_PATH, OBS_CONFIG_PATH, ACTION_CONFIG_PATH" << std::endl;
        return 1;
    }

    std::string scene_path = argv[1]; 
    std::string obser_conf_path = argv[2]; 
    std::string action_conf_path = argv[3]; 

    sf::HelperSettings h;
    sf::RenderSettings r;
    r.windowW = 900;
    r.windowH = 600;
    
    StonefishRL* simManager = new StonefishRL(scene_path, obser_conf_path, action_conf_path, frequency); // Create the StonefishRL simulation manager

    sf::GraphicalSimulationApp app("DEMO STONEFISH RL", scene_path, r, h, simManager);
    //sf::ConsoleSimulationApp app("DEMO STONEFISH RL", scene_path, simManager);

    LearningThreadData data {app}; // is a struct that holds a reference to the sim app
    SDL_Thread* learningThread = SDL_CreateThread(learning, "learningThread", &data);

    app.Run(false, false, sf::Scalar(1/frequency));

    SDL_WaitThread(learningThread, nullptr);
    return 0;
}