#!/bin/bash

echo "Content:"; ls
pip3 install -r requirements.txt

python3 exp/run_join_exp.py -s /tmp/join_exp_config.csv
# echo "[Success] The configuration database is generated."

# Print memory usage
# nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used --format=csv

# Install the required packages (once a day)
# echo "Installing the required packages"
# python3 -m pip install -r requirements.txt

echo "Run microbenchmarks from Section 5.2.1 to 5.2.7"
python3 exp/run_join_exp.py \
        -b ./bin/volcano/join_exp_4b4b \
        -c /tmp/join_exp_config.csv \
        -y exp/join_runs.yaml \
        -e 0 \
        -r 1 \
        -o exp_results/output \
        -p exp_results/gpu_join

# Print results
# echo "Contents of exp_results/gpu_join/pkfk_vanilla.csv:"
# cat exp_results/gpu_join/pkfk_vanilla.csv

# echo "Contents of exp_results/output:"
# cat exp_results/output

# Copy results to out/
mkdir -p out/
cp exp_results/output out/output.txt

# python3 exp/run_join_exp.py \
#         -b ./bin/volcano/join_exp_8b8b \
#         # -c exp/join_exp_config.csv \
#         -y exp/join_runs.yaml \
#         -e 5 \
#         -r 1 \
#         -p exp_results/gpu_join \
#         # -d $2
    
# python3 exp/run_join_exp.py \
#         -b ./bin/volcano/join_exp_4b8b \
#         # -c exp/join_exp_config.csv \
#         -y exp/join_runs.yaml \
#         -e 6 \
#         -r 1 \
#         -p exp_results/gpu_join \
#         # -d $2

# echo "Run the sequence of joins from Section 5.2.8"
# for a in SMJ SMJI SHJ PHJ
# do 
#     for r in {1..$1}
#     do 
#         ./bin/volcano/join_pipeline $2 25 27 $a
#     done
# done

echo "Complete"
