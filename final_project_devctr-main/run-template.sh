#!/bin/bash

echo "Content:"; ls
python3 exp/run_join_exp.py -s /tmp/join_exp_config.csv

echo "Run microbenchmarks from Section 5.2.1 to 5.2.7"
python3 exp/run_join_exp.py \
        -b ./bin/volcano/join_exp_4b4b \
        -c /tmp/join_exp_config.csv \
        -y {{ join_runs_path }} \
        -e 0 \
        -r 1 \
        -o exp_results/output \
        -p exp_results/gpu_join

# Copy results to out/
mkdir -p out/
cp exp_results/output out/output.txt

echo "Complete"
