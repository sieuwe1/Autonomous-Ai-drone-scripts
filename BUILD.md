# NVIDIA Jetson Nano setup manual

## Manual

### Install the operating system

Follow the instructions at https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write to install the operating system to the Jetson Nano. Make sure to finish the installation process, e.g.  setting the timezone and creating a user account.

#### Upgrade and cleanup

Upgrade the system and automatically remove  unnecessary packages

```bash
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
```

#### Install prerequisites

Before starting the installation process, please ensure that `git` and `python3-pip` are installed.

```bash
sudo apt install nano git python3-pip
```

#### Optional: Strip the operating system

Because the standard Jetson Nano operating system comes with a lot of tools which won't be used, we can save valuable space and resources by removing them.

```bash
sudo apt purge snapd lx* nautilus* openbox* printer* rhythmbox* gnome* lightdm* xubuntu* xscreensaver* xfce* lxde* x2go* word* thunderbird libreoffice* chromium-* vim* docker* libvisionworks* ubuntu-wallpapers-bionic light-themes adwaita-icon-theme -y && sudo apt autoremove -y
```

> This command will strip the operating system of *most* unnecessary packages. Please read the command before executing to avoid removing necessary packages.

### Installation of the project

Clone the project from the [GitHub repository](https://github.com/sieuwe1/Autonomous-Ai-drone-scripts).

```bash
git clone https://github.com/sieuwe1/Autonomous-Ai-drone-scripts.git
```

#### Project packages

##### Jetson Inference

Start by installing the required packages for the installation process.

```bash
sudo apt install cmake libpython3-dev python3-numpy
```

Execute the following commands in sequence to install Jetson Inference.

> During the installation, make sure to deselect all models and finally select the `SSD MobileNet V2` model. After the download has completed, please also install PyTorch.

```bash
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig
```

*Source: https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md*

##### Remaining packages

```bash
python3 -m pip install --upgrade pip
python3 -m pip install scikit-build cmake opencv-python numpy dronekit pyserial keyboard simple-pid
```

#### Utilities

##### X11 forwarding

For viewing sensors, media and other things with a graphical user interface, we can use X11 forwarding to forward the interfaces over SSH. 

Enable X11 forwarding in the `sshd` configuration file.

```bash
sudo nano /etc/ssh/sshd_config
```

Look for a parameter called `X11Forwarding`, ensure that the value is set to `yes`. It should look like this.

```
X11Forwarding yes
```

To connect to the Jetson Nano with X11 support, use the following command:

```bash
ssh -X -Y USERNAME@IP
```

###### Viewing videos

A common use case for X11 within the drone project is to watch back debugging videos. To review debugging videos from the drone via X11, you can use the `mplayer` package, which is a lightweight media player.

Install `mplayer`

```bash
sudo apt install mplayer
```

Example of `mplayer` usage

```bash
mplayer video.mp4
```

### Simulator environment installation

Clone the simulator from ArduPilot from the [GitHub repository](https://github.com/ArduPilot/ardupilot/).

```bash
git clone https://github.com/ArduPilot/ardupilot --recursive
cd ardupilot
```

*Source: https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux*

#### Environment prerequisites

Before executing the prerequisite installation script from ArduPilot, please install some additional packages.

```bash
sudo apt install libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev python3-setuptools python3-dev python3 libportmidi-dev
```

After that, install all prerequisites using the tool from ArduPilot by executing the following command:

```bash
Tools/environment_install/install-prereqs-ubuntu.sh -y
```

Reload the path (log-out and log-in to make permanent).

```bash
. ~/.profile
```

#### Build

Before building, ensure that the project is configured.

```bash
./waf configure
```

After the project is configured, build the project.

```bash
./waf clean
./waf build
```

#### Startup

Ensure that the current directory is the scripts directory.

```bash
cd Tools/scripts
```

Start the simulator for a `ArduCopter`

```bash
sim_vehicle.py -v ArduCopter --map --console
```

### Cleanup

Finish by removing all unused packages.

```bash
sudo apt autoremove -y
```
