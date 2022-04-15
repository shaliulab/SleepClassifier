#! /bin/bash

make-umapplot  --h5ad-input results/2021_treatment_predictions/KC-data/KC.h5ad  --threshold inf --input /Users/Antonio/SleepML/results/2021_treatment_predictions/KC-homogenization --output results/2021_treatment_predictions/KC-homogenization/2022-01-21_UMAP-plots/KC

make-umapplot  --h5ad-input results/2021_treatment_predictions/KC-data/KC.h5ad  --threshold 7.0 --input /Users/Antonio/SleepML/results/2021_treatment_predictions/KC-homogenization --output results/2021_treatment_predictions/KC-homogenization/2022-01-21_UMAP-plots/KC

make-umapplot  --h5ad-input results/2021_treatment_predictions/KC-data/KC.h5ad  --threshold 3.0 --input /Users/Antonio/SleepML/results/2021_treatment_predictions/KC-homogenization --output results/2021_treatment_predictions/KC-homogenization/2022-01-21_UMAP-plots/KC

make-umapplot  --h5ad-input results/2021_treatment_predictions/KC-data/KC.h5ad  --threshold 2.6 --input /Users/Antonio/SleepML/results/2021_treatment_predictions/KC-homogenization --output results/2021_treatment_predictions/KC-homogenization/2022-01-21_UMAP-plots/KC

make-umapplot  --h5ad-input results/2021_treatment_predictions/glia-data/glia.h5ad  --threshold 7.0 --input /Users/Antonio/SleepML/results/2021_treatment_predictions/glia-homogenization --output results/2021_treatment_predictions/glia-homogenization/2022-01-21_UMAP-plots/glia --ignore-cell-types SPG

make-umapplot  --h5ad-input results/2021_treatment_predictions/glia-data/glia.h5ad  --threshold 6.5 --input /Users/Antonio/SleepML/results/2021_treatment_predictions/glia-homogenization --output results/2021_treatment_predictions/glia-homogenization/2022-01-21_UMAP-plots/glia  --ignore-cell-types SPG

make-umapplot  --h5ad-input results/2021_treatment_predictions/glia-data/glia.h5ad  --threshold 6.0 --input /Users/Antonio/SleepML/results/2021_treatment_predictions/glia-homogenization --output results/2021_treatment_predictions/glia-homogenization/2022-01-21_UMAP-plots/glia  --ignore-cell-types SPG
