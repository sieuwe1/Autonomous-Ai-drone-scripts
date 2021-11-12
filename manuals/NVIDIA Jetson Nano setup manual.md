# NVIDIA Jetson Nano setup manual

Author: Jelle Maas
Date of writing: September 28th, 2021

## Table of contents
* * *
[[toc]]

## Context

During the autonomous drone project, we had to reinstall the Jetson Nano operating system. This took a long time because there was no clear manual available. This document contains a manual to more easily reproduce the installation process of the operating system.

In addition to that it also gives advice over potential improvements of the foundation of the operating system.

## Manual

### Install the operating system

Follow the instructions at <https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write> to install the operating system on the Jetson Nano. Make sure to finish the installation process, e.g.  setting the timezone and creating a user account.

#### Upgrade and cleanup

Upgrade the system and automatically remove  unnecessary packages

```bash
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
```

#### Install prerequisites

This manual uses the `nano` text editor, which is not installed by default on the Jetson Nano. You can use `vim` instead or install `nano` by executing the following command.

```bash
sudo apt install nano
```

#### Optional: Strip the operating system

Because the standard Jetson Nano operating system comes with a lot of tools which won't be used in the context of the drone project. My advice is to remove these tools so we can save valuable space and computer resources (e.g. memory, space and processing power).

The following command create a lite version of the operating system, by stripping it of _most_ unnecessary packages. Please read the command before executing to avoid removing necessary packages.

```bash
sudo apt purge snapd lx* nautilus* gedit openbox* printer* rhythmbox* gnome* lightdm* xscreensaver* thunderbird libreoffice* chromium-* docker* ubuntu-wallpapers* light-themes && sudo apt autoremove
```

##### Removal of Network Manager and WPA supplicant

Additional packages to remove are `network-manager` and `wpasupplicant`. They are both packages that helps with setting up network connections. If the drone doesn't use a complicated network connection (e.g. Eduroam), we can use the default Linux `systemd-networkd` service for this.

Before removing `network-manager` and `wpasupplicant`, please ensure that DHCP is enabled for the Ethernet interface. This is to prevent losing SSH access over the Ethernet port. You can do this by setting up a network file, if one does not exist already, please create one.

```bash
sudo nano /etc/systemd/network/eth0.network
```

The name of the file does not matter, it can be anything that describes it's purpose. The matching interface is set within the network file under the `[Match]` attribute.

The final contents of the network file should look something like the following.

    [Match]
    Name=eth0

    [Network]
    DHCP=yes

Last but not least, restart and enable the `systemd-networkd` service.

    sudo systemctl restart systemd-networkd
    sudo systemctl enable systemd-networkd

After everything is setup,  `network-manager` and `wpasupplicant` can removed safely.

    sudo apt purge network-manager wpasupplicant

### Autonomous drone

#### Installation

##### Prerequisites

Before starting the installation process, please ensure that `git` and `python3-pip` are installed.

```bash
sudo apt install git python3-pip
```

##### Clone the project

Clone the project from the [GitHub repository](https://github.com/sieuwe1/Autonomous-Ai-drone-scripts).

```bash
git clone https://github.com/sieuwe1/Autonomous-Ai-drone-scripts.git
```

##### Dependencies

###### Jetson Inference

Start by installing the required dependencies for the installation process.

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

_Source: <https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md>_

###### Remaining packages

```bash
python3 -m pip install --upgrade pip
python3 -m pip install scikit-build cmake opencv-python numpy dronekit pyserial keyboard simple-pid geographiclib
```

#### Usage

Ensure the ArduCopter is in guided mode, then execute the following command:

```bash
sudo python3 follow_person_main.py --mode=flight --debug_path=debug/flight1
```

For ArduPilot SITL simulator testing, you can have to set the mode to `test`, this can be done by changing the `--mode` parameter from `flight` to `test`.

### Simulator environment installation

Clone the simulator from ArduPilot from the [GitHub repository](https://github.com/ArduPilot/ardupilot/).

```bash
git clone https://github.com/ArduPilot/ardupilot --recursive
cd ardupilot
```

_Source: <https://ardupilot.org/dev/docs/building-setup-linux.html#building-setup-linux>_

#### Environment prerequisites

Before executing the prerequisite installation script from ArduPilot, please install some additional packages.

```bash
sudo apt-get install python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev
```

Also ensure that you have the latest version of `setuptools` and `wheel`.

```bash
pip2 install -U setuptools wheel
```

After that, install all prerequisites using environment installation script from ArduPilot by executing the following command:

```bash
Tools/environment_install/install-prereqs-ubuntu.sh -y
```

Reload the path to add the necessary tools to bash, log-out and log-in to make it permanent.

```bash
. ~/.profile
```

##### Pygame not installing

If the `pygame` package is refusing to install, you can force the version of `pygame` to a different version than the latest version. During our testing, version 2.0.0 was working. The following command will modify the script so it targets `pygame` a working version.

```bash
sed -i 's/\$PYTHON_PKGS pygame /$PYTHON_PKGS pygame==2.0.0 /' Tools/environment_install/install-prereqs-ubuntu.sh
```

Then retry executing the environment installation script from ArduPilot by following the instructions at [[#Environment prerequisites|Environment prerequisites]].

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

#### Usage

If everything is done right, the simulator should be appended to the bash source at the bottom of `~/.bashrc`. Start the simulator for a `ArduCopter`.

```bash
sim_vehicle.py -v ArduCopter
```

Alternatively, you can execute the script directly from the `/ardupilot/Tools/autotest` directory.

##### Map and/or console module(s)

If you wish to start the simulator with the console and/or the map, it is required to install some additional packages. Please change the `pip` version according to your Python version. Don't worry if the installation for `wxpython` is taking long, this is normal.

```bash
pip install -U --user --verbose wxpython console map
```

Then start the simulator using the `--console` and/or the `--map` parameters.

```bash
sim_vehicle.py -v ArduCopter --console --map
```

### Cleanup

Finally, finish by automatically removing all unused packages.

```bash
sudo apt autoremove -y
```

### Utilities

#### X11 forwarding

For viewing sensors, media and other things with a graphical user interface, we can use X11 forwarding to forward the interfaces over SSH.

##### Setup

Most Linux distributions should already have X11 forwarding enabled by default. If it is not enabled, X11 forwarding can be enabled in the `sshd` configuration file.

```bash
sudo nano /etc/ssh/sshd_config
```

Look for a parameter called `X11Forwarding`, ensure that the value is set to `yes`. It should look like this.

    X11Forwarding yes

##### Connecting

To connect to the Jetson Nano with X11 support, use the following command:

```bash
ssh -X -Y USERNAME@IP
```

##### Viewing videos

A common use case for X11 within the drone project is to watch back debugging videos. To review debugging videos from the drone via X11, you can use the `mplayer` package, which is a lightweight media player that also works in combination with X11 forwarding.

Install `mplayer`

```bash
sudo apt install mplayer
```

Example of `mplayer` usage

```bash
mplayer video.mp4
```
