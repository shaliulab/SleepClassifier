#! /bin/bash

for CELLTYPE in EGN+- EGN++ SPG PNG ALG+ CXG
do
  echo ${CELLTYPE}
  train-torch-model ${CELLTYPE} --output results/glia-models/NN/${CELLTYPE} --seed 1000 --model-name NeuralNetwork --h5ad-input results/glia-data/glia-raw.h5ad --training-config training_params.yaml --label-mapping data/templates/simple_condition_mapping.yaml --fractions 0.25 0.5 0.75 1.0
  test-torch-model  ${CELLTYPE} --output results/glia-models/NN/${CELLTYPE} --seed 1000 --model-name NeuralNetwork    --cluster-data ${CELLTYPE}
done

