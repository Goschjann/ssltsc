#!/bin/bash
#SBATCH -p gpu -n 1 --gres=gpu:tesla:1
export GIT_PYTHON_REFRESH=quiet


seed=$1
split=$2
config=$3
config_file="experiments/config_files/tune_arch/$config.yaml"

echo "Running single evaluation with seed=$seed and split=$split and config_file=${config_file}"
python -m experiments.run --config $config_file \
                          --mlflow_name $config \
                          --num_workers 0 \
                          --num_labels $split \
                          --seed $seed