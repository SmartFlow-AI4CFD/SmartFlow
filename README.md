# SmartFlow
**SmartFlow** is an open-source framework developed as a joint effort among researchers from various institutions to advance research in turbulence modeling, flow control, and numerical algorithm development through multi-agent deep reinforcement learning (DRL). SmartFlow leverages the [SmartSim](https://github.com/CrayLabs/SmartSim) infrastructure library to efficiently launch and manage computational fluid dynamics (CFD) simulations. Data exchange between CFD simulations (implemented in Fortran or C++) and DRL models (implemented in Python) is seamlessly handled by [SmartRedis](https://github.com/CrayLabs/SmartRedis) clients, ensuring efficient and scalable communication. The framework is optimized for deployment on high-performance computing (HPC) platforms, including both CPU clusters and GPU-accelerated architectures, enabling scalable and efficient training for computationally demanding problems. Built on top of [Relexi](https://github.com/flexi-framework/relexi) and [SmartSOD2D](https://github.com/b-fg/SmartSOD2D), SmartFlow offers the following two improvements:

1. **CFD-solver-agnostic framework:** To simplify the integration of diverse CFD solvers with the DRL framework, we have developed a data communication library [SmartRedis-MPI](https://github.com/soaringxmc/smartredis-mpi) that can be easily linked to various CFD solvers. As an example, **only five lines of code** are needed to enable coupling between the [CaLES](https://github.com/soaringxmc/CaLES) solver and the SmartFlow framework.

2. **PyTorch-based [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/):** Reinforcement learning algorithms are implemented using the widely adopted Stable-Baselines3 library, which is much easier to use, compared to the TensorFlow-based TF-Agents library used in previous implementations.

## How it works
Most high-performance CFD solvers are written in low-level languages such as C/C++/Fortran, while ML libraries are typically available in Python or other high-level languages. This creates a "two-language problem" when data needs to be shared across different processes - a common challenge during online training of ML models where continuous data exchange occurs between the model and the CFD solver.

While this challenge can be addressed using Unix sockets or message-passing interface (MPI), SmartSim provides an elegant solution through a workflow library that facilitates communication between different instances via an in-memory Redis database. SmartRedis, which provides client libraries for C/C++/Fortran/Python, enables seamless communication with applications written in any of these languages.

The SmartRedis clients offer high-level API calls such as `put_tensor` and `get_tensor` that simplify sending and receiving data arrays, significantly reducing the overall software complexity. Additionally, SmartSim can manage processes, such as starting or stopping multiple CFD simulations, further streamlining workflow orchestration.

## Learn More about SmartFlow

For comprehensive information on how to use and contribute to SmartFlow, please visit the webpage [**SmartFlow documentation**](https://smartflow.readthedocs.io/en/latest/).

The documentation includes:

- **How it works** – An overview of SmartFlow and data communication strategy  
- **Installation** – Step-by-step instructions for setting up the framework on various platforms  
- **Run a case** – Guidelines for launching and managing CFD-DRL workflows  
- **Example** – Demonstrative cases to help users get started quickly  
- **Contributors** – Information about the developers and collaborators involved in the project  
- **Acknowledgements** – Recognition of individuals from the CFD and HPC communities who supported this work 
- **Bibliography** – Relevant academic publications and citation information 