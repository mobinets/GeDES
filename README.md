## Prerequisites
This project uses [CMake](https://cmake.org/) (3.0+) and Ubuntu 22.04 for building and testing.
It also requires dependencies of [NVIDIA GPU Driver 575+](https://www.nvidia.com/en-us/geforce/drivers/) and [CUDA 11.08+](https://developer.nvidia.com/cuda-11-8-0-download-archive).

### Installing NVIDIA GPU Driver and CUDA
Detailed steps can be found [here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

```shell
$ sudo apt update
$ sudo apt install nvidia-driver-575-server
$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
$ sudo sh cuda_11.8.0_520.61.05_linux.run
```

After the installation, please modify the following two lines in `CMakeLists.txt` accordingly.

```cmake
include_directories(
    "/path/to/source/directory/GeDES_code/code/c++/include/"
    "/usr/local/cuda/include"
    "/usr/include"
    "/usr/local/include"
)
set(SOURCE_DIR /path/to/source/directory/GeDES_code/code/c++/src/)
```

## Quick Evaluation
We provide a quick evaluation script to run the simulation.
```shell
$ sudo apt install python3
$ sudo apt install python3-pip
$ pip3 install -r requirements.txt
$ python3 script.py
```

The `data/` directory holds simulation raw data, and the `img/` directory contains generated visualizations including figures and tables.

## Build source code

We use cmake for GeDES 

```shell
$ cd GeDES_code
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j 18
```
To run the simulation:

```shell
$ ./GeDES --ft_k=xx \
    --packet_pool_size=xx \
    --output=xx \
    --average_flow_size=xx \
    --flow_time_range=xx
```



## The orgnization of the code
The list of the files in the project:
```
├── GeDES_code/
│   ├── CMakeLists.txt
│   ├── main.cc
│   └── code/
│       └── c++/
│           ├── include/
│           │   ├── arp_protocol.h
│           │   ├── component.h
│           │   ├── conf.h
│           │   ├── frame_decapsulation.h
│           │   ├── frame_encapsulation.h
│           │   ├── gpu_packet_manager.h
│           │   ├── gpu_queue.cuh
│           │   ├── ipv4_controller.h
│           │   ├── ipv4_decapsulation.h
│           │   ├── ipv4_encapsulation.h
│           │   ├── memory_pool.h
│           │   ├── p2p_channel.h
│           │   ├── packet_cache.h
│           │   ├── packet_definition.h
│           │   ├── protocol_type.h
│           │   ├── switch.h
│           │   ├── tcp_controller.h
│           │   ├── thread_pool.h
│           │   ├── topology.h
│           │   └── vdes_timer.h
│           └── src/
│               ├── cache/
│               │   ├── packet_cache.cc
│               │   └── packet_cache.cu
│               ├── channel/
│               │   ├── p2p_channel.cc
│               │   └── p2p_channel.cu
│               ├── components/
│               │   ├── component.cc
│               │   ├── topology.cc
│               │   ├── vdes_timer.cc
│               │   └── vdes_timer_kernel.cu
│               ├── config/
│               │   └── config.cc
│               ├── example/
│               │   ├── network_demo.cc
│               │   └── node-switch-node.cc
│               ├── helper/
│               ├── ipv4/
│               │   ├── ipv4_controller.cc
│               │   └── ipv4_controller.cu
│               ├── link_layer/
│               │   ├── switch.cc
│               │   └── switch.cu
│               ├── memory_management/
│               │   ├── gpu_packet_manager.cc
│               │   └── packet_management.cu
│               ├── middleware/
│               │   ├── frame_decapsulation.cc
│               │   ├── frame_decapsulation.cu
│               │   ├── frame_encapsulation.cc
│               │   ├── frame_encapsulation.cu
│               │   ├── ipv4_decapsulation.cc
│               │   ├── ipv4_decapsulation.cu
│               │   ├── ipv4_encapsulation.cc
│               │   └── ipv4_encapsulation.cu
│               ├── transport/
│               │   ├── tcp_controller.cc
│               │   └── tcp_controller.cu
│               └── utils/
│                   ├── gpu_queue.cu
│                   └── thread_pool.cc
```
The relationship between the header files:
```
├── topology.h 
│   ├── component.h  \\base component class for all network components
│   ├── thread_pool.h  \\thread pool management for parallel execution
│   ├── memory_pool.h  \\memory management for packet allocation
│   ├── gpu_packet_manager.h  \\GPU-side packet management
│   └── switch.h  \\network switch implementation
│       ├── p2p_channel.h  \\point-to-point communication channels
│       ├── ipv4_controller.h  \\IPv4 network layer control
│       │   ├── ipv4_encapsulation.h  \\IPv4 packet encapsulation
│       │   └── ipv4_decapsulation.h  \\IPv4 packet decapsulation
│       ├── tcp_controller.h  \\TCP transport layer control
│       ├── frame_encapsulation.h  \\frame-level encapsulation
│       ├── frame_decapsulation.h  \\frame-level decapsulation
│       ├── packet_cache.h  \\packet caching mechanisms
│       └── arp_protocol.h  \\ARP protocol implementation
```
The impl files contain the implementation of the classes in the header files.
