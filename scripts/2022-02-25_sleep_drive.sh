#! /bin/bash

## Generate a dataset only with cells from the given cluster family (background)
#make-dataset --h5ad-input data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad  --seed 1500 --background "data/backgrounds/KC.csv" --batch-genes-file  "data/batch_effects.xlsx" --shuffles 1 --h5ad-output results/KC-data/KC-raw.h5ad
#make-dataset --h5ad-input data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad  --seed 1500 --background "data/backgrounds/glia.csv" --batch-genes-file  "data/batch_effects.xlsx" --shuffles 1 --h5ad-output results/glia-data/glia-raw.h5ad


for BACKGROUND in KC glia
do
    CELLTYPES=$(cat data/backgrounds/${BACKGROUND}.csv | tail +2 | cut -f 1 -d,)
    INPUT_H5AD="results/${BACKGROUND}-data/${BACKGROUND}-no-marker-genes.h5ad"
    for CELLTYPE in ${CELLTYPES[@]}
    do
        # train a classifier of the categories listed in simple_condition_mapping
        # use full training set (still leave a test set)
        # training configuration specified in training_params.yaml
        train-torch-model ${CELLTYPE} --output results/${BACKGROUND}-models/NN/${CELLTYPE} --seed 1000 --model-name NeuralNetwork --h5ad-input ${INPUT_H5AD} --training-config training_params.yaml --label-mapping data/templates/simple_condition_mapping.yaml --fractions 1.0  --n-neurons 100 10 > logs/train_torch_model_${CELLTYPE}.log
        # plot the confusion table
    
        for DATATYPE in ${CELLTYPES[@]}
        do
            test-torch-model ${CELLTYPE} --output results/${BACKGROUND}-models/NN/${CELLTYPE} --seed 1000 --model-name NeuralNetwork  --cluster-data ${DATATYPE} > logs/test_torch_model_${CELLTYPE}_${DATATYPE}.log
        done
    done
done


# sweep
sleep-models-sweep y --output results/KC-sweeps/ --sweep-config data/sweeps/2022-02-25_sweep.conf --model-name NeuralNetwork --input results/KC-models/NN/y/random-state_1000_fraction_1.0/
