#! /bin/bash

for BACKGROUND in glia KC
do
    CELLTYPES=$(cat data/backgrounds/${BACKGROUND}.csv | tail +2 | cut -f 1 -d,)
    INPUT_H5AD="results/${BACKGROUND}-data/${BACKGROUND}-no-marker-genes.h5ad"
    for CELLTYPE in ${CELLTYPES[@]}
    do
        # train a classifier of the categories listed in simple_condition_mapping
        # use full training set (still leave a test set)
        # training configuration specified in training_params.yaml
	sleep-models-sweep ${CELLTYPE} --output results/${BACKGROUND}-sweeps/ --sweep-config data/sweeps/2022-02-25_sweep.conf --model-name NeuralNetwork --input results/${BACKGROUND}-models/NeuralNetwork/${CELLTYPE}/random-state_1000_fraction_1.0/  > logs/sweep_${CELLTYPE}.log
    done
done


