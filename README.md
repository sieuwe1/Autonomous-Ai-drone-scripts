# Fully autonomous AI powered drone
<p align="center">
<img src="https://github.com/sieuwe1/Autonomous-AI-drone-scripts/raw/main/logo.png" alt="drawing" width="600" />
</p align="center">
  
This repository pushes to create an state of the art fully autonomous navigation and obstacle avoidance system for multi rotor vehicles. Our approach is based on the novel idea of an fully END-2-END AI model which takes the sensor inputs and directly output the desired control commands for the drone. Currently we are working on creating the necessary code for training and running this approach. 

This project also contains an fully autonomous system for tracking a moving target using an Camera and an LiDAR. This sytem uses an AI based object detection model for detecting the target. This system has already been fully tested in real flights and is fully functional!

<p align="center">
<img src="https://github.com/sieuwe1/Autonomous-AI-drone-scripts/raw/main/demo_media/flight.gif" alt="drawing" width="800"/>
</p align="center">
  
## Hardware 

Our quadcopter is fitted with a [Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/) running advanced AI models and algorithms real time on the edge on the quadcopter itself. This makes our quadcopter able to work fully autonomsly without the need of a data connection to offload heavy workloads. Because of optimizations our algorithms can work in the computational limited environment of the [Jetson Nano](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/).

![Drone image](https://github.com/sieuwe1/Autonomous-AI-drone-scripts/raw/main/demo_media/216C5829-F7F0-4B80-9CDA-27B1BF304F7F.jpeg)

![Drone image](https://github.com/sieuwe1/Autonomous-AI-drone-scripts/raw/main/demo_media/F819741B-72A1-48B6-A64A-C11C24E5973E.jpeg)

## Setup

To setup and build the project on a [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano-developer-kit), please follow the instruction located at [BUILD.md](BUILD.md). These instructions should guide you towards a fully operational Jetson Nano, including all the required software to start flying autonomously.

## Scripts

### Follow person

The follow person script can autonomously follow a person using a live RGB camera feed and a MobileNet AI model. This model can detect a person and use the centerpoint to calculate the `yaw` commands for the drone. `roll` commands are calculated by using a TF Luna solid state LiDAR. This LiDAR can measure the distance between the person and the drone. Both axis commands are generated using PID controllers.

#### Usage

Ensure the ArduCopter is in guided mode, then execute the following command:

```sh
sudo python3 follow_person_main.py --mode=flight --debug_path=debug/flight1
```

For ArduPilot SITL simulator testing, you can have to set the mode to `test`, this can be done by changing the `--mode` parameter from `flight` to `test`.

Press Q to exit the script, the drone will automatically land. Note: this is currently not working in real flight! Switch ArduPilot to `Loiter` mode and then manually land the drone.

### Data plotter

A script was also made to plot the debug data from the PID controllers.

#### Usage

Execute the following command to debug the data from the PID controllers:

```sh
python3 data_plotter.py
```

Ensure the file name in the script is set to the correct debug file. This can be done by editing the `data_plotter.py` script.

## Note

This project is still under heavy development. All code is experimental so please use the code with caution! We are not responsible for damage to people or property from the usage of these scripts.
