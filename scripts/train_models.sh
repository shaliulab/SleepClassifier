#! /bin/bash

# --h5ad-input files need to point to an h5ad produced with remove-marker-genes
# so the cluster separtion is gone
train-models --h5ad-input data/h5ad/Preloom/KC_mapping_wo-marker-genes_log2FC_threshold-2.6.h5ad --background data/backgrounds/KC_mapping.csv --seed 1000 2000 3000 --ncores 5 --verbose 10
train-models --h5ad-input data/h5ad/Preloom/glia_mapping_wo-marker-genes_log2FC_threshold-6.0.h5ad --background data/backgrounds/glia_mapping.csv --seed 1000 2000 3000 --ncores 5 --verbose 10
train-models --h5ad-input data/h5ad/Preloom/peptides_mapping_wo-marker-genes_log2FC_threshold-2.0.h5ad --background data/backgrounds/peptides_mapping.csv --seed 1000 2000 3000 --ncores 5 --verbose 10

