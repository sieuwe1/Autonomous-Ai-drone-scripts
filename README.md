# Fully autonomous Ai powered drone's!

## This repo contains ready to use and extensively real life tested code for flying advanced quad copters autonomosly. 

![Alt Text](https://github.com/sieuwe1/Autonomous-Ai-drone-scripts/blob/main/demo_media/flight.gif)

#### Our quad copter is fitted with a Jetson Nano running advanced Ai models and algorithms real time on the edge on the quad copter itself. This makes our quad copter able to work fully autonomsly without the need of a data connection to offload heavy workloads. Because of optimizations our algorithms can work in the compuational limited envoirement of the Jetson Nano!

### status
#### Work is still being done and more will come! All code is experimental soo please use any code with caution!

### scripts
We have a person follow script which can autonomsly follow a person using a live RGB camera feed and a mobilenet Ai model. This model can detect a person and use the centerpoint to calculate the YAW commands for the drone. ROLL commands are calculated by using a tf-luna solid state lidar. This lidar can measure the distance between the person and the drone. Both axis commands are generated using PID controllers

- Run "sudo python3 follow_person_main.py --mode=flight --debug_path=debug/flight1". Make sure ArduCopter is in guided mode!
- For ardupilot SITL simulator testing use "sudo python3 follow_person_main.py --mode=test --debug_path=debug/flight1".

Press Q to quit the person follower. The drone will automaticly land. Currently not working in real flight! Switch ardupilot to loiter mode and then manually land the drone.

We also have a data_plotter to plot debug data from the PID controllers. 
- Run "python3 data_plotter.py". Make sure to change the file name in the script to the correct debug file. 

### packages
#### The following software packages are needed
- Jetson Interference from https://github.com/dusty-nv/jetson-inference
- Opencv
- Numpy
- Dronekit
- pyserial
- keyboard

### Hardware 
![Alt Text](https://github.com/sieuwe1/Autonomous-Ai-drone-scripts/blob/main/demo_media/216C5829-F7F0-4B80-9CDA-27B1BF304F7F.jpeg)

![Alt Text](https://github.com/sieuwe1/Autonomous-Ai-drone-scripts/blob/main/demo_media/F819741B-72A1-48B6-A64A-C11C24E5973E.jpeg)
