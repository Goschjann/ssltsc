#!/bin/bash

seeds='1461 1645 7233 5284 5765'
splits='50 100 250 500 1000'

datasets="electricdevices pamap2 sits"
algorithms="supervised vat mixmatch"

architectures="CNNLSTM inceptiontime"

# Seeds
for seed in $seeds
do
    # Algorithm
    for algorithm in $algorithms
    do

        # Datasets
        for dataset in $datasets
        do
            # Architectures
            for architecture in $architectures
            do
                # Splits
                for split in $splits
                do
                    config="${algorithm}_${dataset}_${architecture}"

#                   ./run_single_evaluation.sh $seed $split $config
                    sbatch run_single_evaluation.sh $seed $split $config
                done
            done
        done
    done
done