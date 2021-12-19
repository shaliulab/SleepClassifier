#! /bin/bash

BACKGROUND=$1

#snakemake results/${BACKGROUND}-data/${BACKGROUND}-no-marker-genes.h5ad   --cores all --allowed-rules   remove_marker_genes
#snakemake results/${BACKGROUND}-models/EBM/random-state_1000   --cores all --allowed-rules   train_models 
##snakemake results/${BACKGROUND}-models/EBM/random-state_1000/accuracy.csv   --cores all --allowed-rules   predict_sleep
#snakemake   --cores all -R   predict_sleep
snakemake results/${BACKGROUND}-models/EBM/matrixplot.png   --cores all --allowed-rules  make_matrixplot 
