# 6.S894 Final Project
### Group Members
* Aryan Kumar
* Kenneth Choi
* Siddhant Mukherjee
* Tasmeem Reza

### Project Description
We plan to implement some of the optimizations mentioned in the paper “Efficiently Processing Large Relational Joins on GPUs” (https://arxiv.org/pdf/2312.00720). The paper discusses 3 primitives provided by NVIDIA’s cub library (radix-partition, sort-pairs, gather), as well as four algorithms to use based on whether the join is wide, whether there are 8-byte keys, and whether there is a low match ratio or skewed foreign key. We will first implement the join algorithms using the cub library and then, time-permitting, we will explore further optimizations. Specifically, the four algorithms we plan to implement are SMJ-UM, SMJ-OM, PHJ-UM, and PHJ-OM. We will benchmark the combined optimizations against the authors’ source code and aim to beat their implementation on the TPC-H/DS benchmarks.

### How to run
Copy your `auth.json` and `connection.json` into the `final_project_devctr-main/` directory.

Run `./run_experiment.sh` to run the experiment. Use the `--compile` flag to recompile cached binaries.

Depending on your computer architecture, you may need to run `export DOCKER_DEFAULT_PLATFORM=linux/amd64` to compile the binaries for x86. If your computer has ARM architecture, run OrbStack; else, run Docker.