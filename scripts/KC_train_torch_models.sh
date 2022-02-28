#! /bin/bash

for CELLTYPE in y ab abp
do
  echo ${CELLTYPE}
  train-torch-model ${CELLTYPE} --output results/KC-models/NN/${CELLTYPE} --seed 1000 --model-name NeuralNetwork --h5ad-input results/KC-data/KC-raw.h5ad --training-config training_params.yaml --label-mapping data/templates/simple_condition_mapping.yaml --fractions 0.25 0.5 1.0 --trim
  test-torch-model  ${CELLTYPE} --output results/KC-models/NN/${CELLTYPE} --seed 1000 --model-name NeuralNetwork    --cluster-data ${CELLTYPE} 
done
