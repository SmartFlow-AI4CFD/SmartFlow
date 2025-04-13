# SmartFlow
**SmartFlow** is an open-source framework developed as a joint effort among researchers from various institutions to advance research in turbulence modeling, flow control, and numerical algorithm development through multi-agent deep reinforcement learning (DRL). SmartFlow leverages the [SmartSim](https://github.com/CrayLabs/SmartSim) infrastructure library to efficiently launch and manage computational fluid dynamics (CFD) simulations. Data exchange between CFD simulations (implemented in Fortran or C++) and DRL models (implemented in Python) is seamlessly handled by [SmartRedis](https://github.com/CrayLabs/SmartRedis) clients, ensuring efficient and scalable communication. The framework is optimized for deployment on high-performance computing (HPC) platforms, including both CPU clusters and GPU-accelerated architectures, enabling scalable and efficient training for computationally demanding problems. Built on top of [Relexi](https://github.com/flexi-framework/relexi) and [SmartSOD2D](https://github.com/b-fg/SmartSOD2D), SmartFlow offers the following two improvements:

1. **CFD-solver-agnostic framework:** To simplify the integration of diverse CFD solvers with the DRL framework, we have developed a data communication library [SmartRedis-MPI](https://github.com/soaringxmc/smartredis-mpi) that can be easily linked to various CFD solvers. As an example, **only five lines of code** are needed to enable coupling between the [CaLES](https://github.com/soaringxmc/CaLES) solver and the SmartFlow framework.

2. **PyTorch-based [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/):** Reinforcement learning algorithms are implemented using the widely adopted Stable-Baselines3 library, which is much easier to use, compared to the TensorFlow-based TF-Agents library used in previous implementations.

## Publications
Please cite us if you find this framework useful!
  - [Maochao Xiao, Francisco Alcántara-Ávila, Bernat Font, Marius Kurz, Di Zhou, Yuning Wang, Ting Zhu, Ricardo Vinuesa, Johan Larsson, and Sergio Pirozzoli, "SmartFlow: An open-source framework for deep reinforcement learning in turbulence modeling, flow control and numerical algorithm development," presented at the 2nd European Fluid Dynamics Conference (EFDC2), Dublin, Ireland, 26–29 August 2025.](https://www.overleaf.com/read/jycvfjgqdpfm#91e56a)

## Table of Contents
- [SmartFLow](#smartflow)
- [Publications](#publications)
- [How it works](#how-it-works)
- [Installation](#installation)
  - [1. SmartSim and SmartRedis](#1-smartsim-and-smartredis)
  - [2. Stable-Baselines3](#2-stable-baselines3)
  - [3. SmartFlow](#3-smartflow)
  - [4. SmartRedis-MPI](#4-smartredis-mpi)
  - [5. CFD Solver](#5-cfd-solver)
- [Run a case](#run-a-case)
  - [Running on a standalone machine](#running-on-a-standalone-machine)
  - [Running with SLURM on a cluster](#running-with-slurm-on-a-cluster)
- [Example](#example)
- [Contributors](#contributors)
- [Acknowledgements](#acknowledgements)

## How it works
Most high-performance CFD solvers are written in low-level languages such as C/C++/Fortran, while ML libraries are typically available in Python or other high-level languages. This creates a "two-language problem" when data needs to be shared across different processes - a common challenge during online training of ML models where continuous data exchange occurs between the model and the CFD solver.

While this challenge can be addressed using Unix sockets or message-passing interface (MPI), SmartSim provides an elegant solution through a workflow library that facilitates communication between different instances via an in-memory Redis database. SmartRedis, which provides client libraries for C/C++/Fortran/Python, enables seamless communication with applications written in any of these languages.

The SmartRedis clients offer high-level API calls such as `put_tensor` and `get_tensor` that simplify sending and receiving data arrays, significantly reducing the overall software complexity. Additionally, SmartSim can manage processes, such as starting or stopping multiple CFD simulations, further streamlining workflow orchestration.

## Installation
The following components are required to use the SmartFlow framework:
1. SmartSim and SmartRedis
2. Stable-Baselines3
3. SmartFlow
4. SmartRedis-MPI
5. CFD solver

For comprehensive installation instructions, please refer to the official documentation for each component:
- [SmartSim and SmartRedis](https://www.craylabs.org/docs/installation_instructions/basic.html)
- [SmartRedis-MPI](https://github.com/soaringxmc/smartredis-mpi)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)

If you plan to contribute advanced features to the framework, we recommend thoroughly reviewing both the [SmartSim](https://www.craylabs.org/docs/overview.html) and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) documentation.

Below is a quick installation guide for all components. Note that Python 3.9-3.11 is required for compatibility. We recommend starting with a fresh virtual environment:

```sh
python -m venv /path/to/smartflow/environment
source /path/to/smartflow/environment/bin/activate
```

### 1. SmartSim and SmartRedis
SmartSim provides the infrastructure for launching and managing simulations, while SmartRedis provides client interfaces for interacting with the in-memory Redis database.

#### Python Installation
```sh
# Install SmartSim
pip install smartsim
# Build SmartSim with CPU support and Dragon
smart build --device cpu --dragon

# Install SmartRedis Python client
pip install smartredis
```

#### C, C++, and Fortran Client Libraries
For compiled language applications, the SmartRedis client libraries need to be built from source. We currently use version 0.5.3 for compatibility.

> **Important:** The CFD solver, SmartRedis, and SmartRedis-MPI must be compiled with the same compiler to ensure proper linking.

##### Using GCC Compilers:
```sh
git clone https://github.com/CrayLabs/SmartRedis --depth=1 --branch=v0.5.3 smartredis
cd smartredis
make lib-with-fortran CC=gcc CXX=g++ FC=gfortran
```

After compilation, add the library path to your environment:
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/smartredis/lib
# Add this to your .bashrc or .bash_profile for persistence
```

##### Using NVIDIA Compilers (for GPU-enabled applications):
```sh
cd smartredis
make lib-with-fortran CC=nvc CXX=nvc++ FC=nvfortran
```

### 2. Stable-Baselines3
Stable-Baselines3 is a Python library that provides implementations of reinforcement learning algorithms. It can be installed with the following command:
```sh
pip install stable-baselines3[extra]
```

### 3. SmartFlow
SmartFlow is installed with the following commands:
```sh
cd SmartFlow
pip install -e .
```
This will mark the current package as editable, so it can be modified and the changes will be automatically available to the Python environment.

### 4. SmartRedis-MPI
Before installing CFD solver, build the SmartRedis-MPI library that will be linked by MPI-based parallel cfd solver:

```sh
git clone https://github.com/soaringxmc/smartredis-mpi.git
cd smartredis-mpi
```

Edit the `Makefile` to set the correct paths to your SmartRedis installation:
```sh
# Adjust the include and lib paths in the Makefile
make
```

Add the library path to your environment:
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/smartredis-mpi/lib
# Add this to your .bashrc or .bash_profile for persistence
```

### 5. CFD Solver
The advantage of SmartFlow is that it can be easily integrated with any CFD solver. Only several lines of code need to be added to the CFD solver to enable the communication with the SmartFlow framework. As an example, we only added five lines of code to the [CaLES](https://github.com/soaringxmc/CaLES) solver to enable its coupling with the SmartFlow framework. If you want to use CaLES as your CFD solver or to simply test the workflow of the SmartFlow framework, please refer to the [CaLES](https://github.com/soaringxmc/CaLES) documentation for installation instructions.

## Run a case
### Running on a standalone machine
To run a case on a standalone machine, we can use:
```sh
python main.py
```

### Running with SLURM on a CPU cluster
To run a case on a CPU cluster, we can use a SLURM script such as:
```sh
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --job-name=smartflow
#SBATCH --account=user_account
#SBATCH --qos=qos_name
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

python main.py
```
The script is submitted with the following command:
```sh
sbatch slurm.sh
```
For this setting, we allocate 2 nodes with 32 tasks per node and 1 CPU per task (for a total of 64 tasks). The job is submitted to the `qos_name` queue under the `user_account` account. The job is expected to run for 48 hours. The output and error logs are saved in the `slurm-%j.out` and `slurm-%j.err` files, respectively, where `%j` represents the job ID.


### Running with SLURM on a GPU-accelerated cluster
To run a case on a GPU-accelerated cluster, we can use a SLURM script such as:
```sh
#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --job-name=smartflow
#SBATCH --account=user_account
#SBATCH --qos=qos_name
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

python main.py
```
For this setting, we allocate 2 nodes with 32 tasks per node and 8 CPUs per task, along with 4 GPUs per node for acceleration. The job is submitted to the `qos_name` queue under the `user_account` account. The job is expected to run for 48 hours. The output and error logs are saved in the `slurm-%j.out` and `slurm-%j.err` files, respectively, where `%j` represents the job ID.

## Quick start example
*An example of using SmartFlow will be added here soon to demonstrate its practical application.*

## Contributors
SmartFlow is developed and maintained by a dedicated team of researchers and developers. We are grateful for the contributions of:

- **Maochao Xiao** ([@soaringxmc](https://github.com/soaringxmc)) - Library design and code implementation
- **Bernat Font** ([@b-fg](https://github.com/b-fg)) - Library design and code implementation
- **Marius Kurz** ([@m-kurz](https://github.com/m-kurz)) - Library design and code implementation
- **Ziyu Zhou** ([@zz-blue](https://github.com/zz-blue)) - Documentation and code implementation
- **You!** ([@your-username](https://github.com/your-username)) - *Psst, that's you! The amazing developer reading this right now* - Code enhancements and brilliant ideas

We are always looking for new contributors to join us to improve the framework. If you are interested in contributing, please feel free to submit a pull request or contact us in the [SmartFlow GitHub Issues](https://github.com/soaringxmc/SmartFlow/issues) or [SmartRedis-MPI GitHub Issues](https://github.com/soaringxmc/smartredis-mpi).

## Acknowledgements

SmartFlow has benefited from the expertise and support of many individuals in the computational fluid dynamics and high-performance computing communities. We are grateful for the valuable discussions with:

- Francisco Alcántara Ávila
- Di Zhou
- Pol Suárez Morales
- Yuning Wang
- Ting Zhu

Special thanks for the guidance and support provided by:
- Ricardo Vinuesa
- Johan Larsson
- Sergio Pirozzoli
