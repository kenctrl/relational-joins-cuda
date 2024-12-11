# 6.S894 Final Project
### Group Members
* Aryan Kumar
* Kenneth Choi
* Siddhant Mukherjee
* Tasmeem Reza

### Project Description
Relational joins are essential to SQL-based data processing, playing a key role in applications such as analytics and database management. Recently, there has been an increasing presence of data-intensive workloads; however, GPU computation has also increased in availability. As a result, optimizing join algorithms for GPU architectures has become a critical research focus. GPUs provide significant parallelism and memory bandwidth, making them suitable for these operations, yet challenges such as materialization costs and workload distribution persist.

We build on the work of Wu et al. in “Efficiently Processing Large Relational Joins on GPUs” (https://arxiv.org/pdf/2312.00720), who proposed an optimized framework for processing large relational joins on GPUs. Their Gather-From-Transformed-Relations (GFTR) technique reduces materialization costs by leveraging sequential memory accesses, outperforming the traditional Gather-From-Untransformed-Relations (GFUR) approach. We implement and benchmark four join algorithms—Sort Merge Join and Partitioned Hash Join, each with and without GFTR optimization—under diverse workloads characterized by varying join width, input relation sizes, and data skewness. Our results validate the state-of-the-art improvements of the GFTR framework, demonstrating significant throughput gains. We also discuss various trade-offs in Sort Merge Join and Partitioned Hash Join implementations. Overall, we enhance the understanding of GPU-accelerated database operations, contributing insights that facilitate the design of efficient query engines for GPU-enabled environments.

### How to run
Copy your `auth.json` and `connection.json` into the `final_project_devctr-main/` directory.

Run `./run_experiment.sh` to run the experiment. Use the `--compile` flag to recompile cached binaries.

Depending on your computer architecture, you may need to run `export DOCKER_DEFAULT_PLATFORM=linux/amd64` to compile the binaries for x86. If your computer has ARM architecture, run OrbStack; else, run Docker.

#### How to run our SMJ-GFUR/GFTR
Comment back in the two lines with `OurSortMergeJoin`, while commenting out the two lines with `SortMergeJoin` and `SortMergeJoinByIndex`.

Toggle correctness check in `join_exp_4b4b.cu` with the `CHECK_CORRECTNESS` flag.

#### How to run our PHJ-GFUR/GFTR
Branches `partitioner` and `phj` contain our PHJ-GFUR/GFTR implementations. `partitioner` has a fully functional paritioned hash join implementation at `final_project_devctr-main/src/src/volcano/PH.cuh`. This header file is included by the benchmarking file `join_exp_4b4b.cu` which runs both our implementation and the paper PHJ implementation. Simply running `./run-experiment.sh -c` would run and benchmark all the yaml files. The output would be dumped at `output.txt` in the `final_project_devctr-main` directory. In the output file, you can search through the initializing token `========Our PHJ Statistics========` which details out which part of the code took how much time.
