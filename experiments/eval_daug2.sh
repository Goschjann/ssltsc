#!/bin/bash

DATASET='sits'
MLFLOWNAME='DAUG2_SITS'

for SEED in 182 1717 29292; do
    for POOL in 'large' 'medium' 'small'; do
        python run.py --config config_files/daug2_fixmatch.yaml --dataset $DATASET --pool $POOL --mlflow_name $MLFLOWNAME --seed $SEED
        python run.py --config config_files/daug2_vat.yaml --dataset $DATASET --pool $POOL --mlflow_name $MLFLOWNAME --seed $SEED
        python run.py --config config_files/daug2_supervised.yaml --dataset $DATASET --pool $POOL --mlflow_name $MLFLOWNAME --seed $SEED
    done
done

