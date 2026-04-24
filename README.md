# Reinforcement Learning Environments with Stonefish

This repository contains several simulation environments based on **Stonefish** and controlled via **Python**.

##  Building
    1. Clone repository.
    2. cd stonefish_rl
    3. mkdir build
    4. cd build
    5. cmake ..
    6. make [-jX]
    8. sudo make install

## Available Environments
- [AcrobotEnv](./docs/README_acrobot.md) – Control a two-link pendulum to reach a target height.
- [G500Env](./docs/README_girona1000.md) – girona1000 robot with a gripper to approach a target object (Ball)-
- [G500TestEnv](./docs/README_tests.md) – Testing environment for sensors and actuators.


## Developer Guide:  
- [Installation](./docs/README_installation.md) – Setup Stonefish, build the C++ server, create the Python environment and run scenes.
- [Manual](./docs/README_manual.md) – Commands (CMD/RESET/EXIT), adding sensors/actuators, creating robots, create an env structure.


## Technologies
- **Simulator**: [Stonefish](https://github.com/patrykcieslak/stonefish)
- **Reinforcement Learning**: Gymnasium + stable-baselines3
- **Communication**: ZeroMQ (pyzmq + cppzmq)  


## Stonefish documentation
This project is built on the Stonefish simulator. For full details about the simulator itself, installation, scene format, robots, sensors/actuators, rendering, and physics options, see the [official repository](https://github.com/patrykcieslak/stonefish).

> The READMEs in this repository focus on the Python and C++ integration and the RL environments.
> For simulator-specific topics (scene syntax, available components, configuration, ...), refer to the Stonefish docs.


## Project Structure   
```
stonefish_rl/
├── src/
|   ├── cpp  # 
|   │   ├── ActuatorController.cpp
|   │   ├── CommandProcessor.cpp
|   │   ├── ConfigLoader.cpp
|   │   ├── main.cpp
|   │   ├── StateManager.cpp
|   │   ├── StonefishRL.cpp
|   │   └── ZQMCommunicator.cpp
|   └── python
|       ├── docking_env.py
|       ├── EnvStonefishRL.py
|       ├── evaluate_docking.py
|       ├── test_docking_env.py
|       ├── train_docking.py
|       └── utils
|           ├── .... 
├── include/
|   ├── cpp
|   │   ├── ActuatorController.h
|   │   ├── CommandProcessor.h
|   │   ├── CommonTypes.h
|   │   ├── ConfigLoader.h
|   │   ├── StateManager.h
|   │   ├── StonefishRL.h
|   │   └── ZMQCommunicator.h
|   ├── observations
|   │   ├── ds_action_config.json
|   │   ├── ds_state_v1_config.json
|   │   └── ds_state_v2_config.json
|   └── parameters
|       ├── evaluation_param.yaml
|       ├── test_param.yaml
|       └── train_param.yaml
├── Resources/
    ├── data
    │   ├── ...
    ├── scenarios
    │   ├── ...
    └── texture
        └── ...
    
