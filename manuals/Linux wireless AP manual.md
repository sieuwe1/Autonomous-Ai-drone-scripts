# Linux wireless AP manual

Author: Jelle Maas
Date of writing: September 20th, 2021

## Table of contents

[[toc]]

## Context

One of the first tasks for the autonomous drone project, was finding improvements to the current communication setup. The current workflow uses a router as an intermediate point to connect the drone and a laptop.

To slightly improve this workflow, I opted to remove this intermediate point by making the autonomous drone act as a wireless access point. This way, a laptop can connect directly to the autonomous drone without the need for a router.

This document contains a manual to more easily reproduce the process of turning the drone into a wireless access point.

## Prerequisites

To start this guide, your system environment must meet a few conditions:

- Installed a Debian based Linux distribution for the `apt` package manager (e.g. Debian, Ubuntu, Raspberry Pi OS)
- Installed a wireless network card that is recognized as a wireless interface (e.g. wlan0)

## Manual

By the end of these instructions you will have a fully working wireless access point on your Debian based Linux system.

### Update the system

Start by updating the system packages.

```bash
sudo apt update && sudo apt upgrade -y
```

### Setting up the DHCP server

To assign IP addresses to connecting clients, we need to setup a DHCP server. In this manual we use the `dnsmasq` package for the DHCP server.

#### Install dnsmasq

Install the `dnsmasq` package.

```bash
sudo apt install dnsmasq
```

#### Configure the DHCP server

Create a backup for the current `dnsmasq` configuration and then create a new configuration for our custom configuration.

```bash
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.orig && sudo nano /etc/dnsmasq.conf
```

> This example uses wlan0 as the wireless interface. To use a different wireless interface, simply change the `interface` parameter.

    interface=wlan0
    dhcp-range=192.168.10.11,192.168.10.30,255.255.255.0,24h

In this example, IP addresses in the range from `192.168.10.11` to `192.168.10.30` can be assigned by the DHCP server. If you have more clients, increase the range.

### Configure static IP

Configuring a static IP address can be done differently across different systems. For almost all Linux systems you can use [systemd-networkd](https://wiki.archlinux.org/title/systemd-networkd). For some other operating systems (Like Raspberry Pi OS), you can use the [dhcpcd](https://wiki.archlinux.org/title/dhcpcd) daemon.

#### Option 1: systemd-networkd

Create a network file for the wireless interface.

```bash
sudo nano /etc/systemd/network/accesspoint.network
```

> This example uses wlan0 as the wireless interface. To use a different wireless interface, simply change the `Name=` parameter.

Configure the static IP address by setting the content of the configuration file to the following content:

    [Match]
    Name=wlan0

    [Network]
    Address=192.168.10.10/24

#### Option  2: dhcpcd

Modify the `dhcpcd` configuration file.

```bash
sudo nano /etc/dhcpcd.conf
```

> This example uses wlan0 as the wireless interface. To use a different wireless interface, simply change the `interface` parameter.

Configure a static IP address by appending the following content to the end of the configuration file:

    interface wlan0
    static ip_address=192.168.10.10/24
    nohook wpa_supplicant # Disable using the same wireless interface as internet

### Setting up the access point

Now we are going to setup the wireless access point itself. This process is made very easy by a package called `hostapd`. This only requires little configuration to setup a fully working wireless access point.

#### Install hostapd

Install the `hostapd` package.

```bash
sudo apt install hostapd
```

#### Configure hostapd

Create a daemon configuration file for `hostapd`.

```bash
sudo nano /etc/hostapd/hostapd.conf
```

> This example uses wlan0 as the wireless interface. To use a different wireless interface, simply change the `interface` parameter.

In this configuration file, we configure the paramaters for the access point. Add the following content to the configuration file:

    interface=wlan0
    hw_mode=g
    channel=1
    macaddr_acl=0
    auth_algs=1
    ignore_broadcast_ssid=0
    ssid=<network_name>
    wpa=2
    wpa_passphrase=<network_password>
    wpa_key_mgmt=WPA-PSK
    wpa_pairwise=TKIP
    rsn_pairwise=CCMP

Don't forget to change the `ssid` and `wpa_passphrase`.

##### Pointing hostapd to our daemon configuration

We still have to show the system the location of the configuration file. Start by modifying the default `hostapd` configuration.

```bash
sudo nano /etc/default/hostapd
```

In this file, track down the line that says `#DAEMON_CONF=””` – remove the **#** and put the path to our daemon configuration file in the quotes, example.

    DAEMON_CONF="/etc/hostapd/hostapd.conf"

#### Start hostapd service

Before we can start the `hostapd` service, we have to unmask it.

```bash
sudo systemctl unmask hostapd
```

After following all the above instructions and unmasking the service, start the `hostapd` service.

```bash
sudo systemctl start hostapd
```

##### Running the hostapd service on startup

If you want to start the hostapd service on startup, we have to enable the service.

```bash
sudo systemctl enable hostapd
```

### Cleaning up

Finish by removing all unused packages.

```bash
sudo apt autoremove -y
```
