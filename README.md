# Sleep models pipeline

Requirements:
===================

`sleep-models==2.0.0`

Instructions
==========================

This pipeline has 5 steps, executed like so:

```
# if you dont have the module
# pip install sleep-models==2.0.0

python 01-pipeline/01_make_dataset.py
python 01-pipeline/02_get_marker_genes.py
python 01-pipeline/03_remove_marker_gens.py
python 01-pipeline/04_train_models.py
python 01-pipeline/05_predict.py
```

Settings are read from the configuration file `config.yaml`:

## 1 make_dataset

### Parameters used:

    **data_dir**
    **background**
    **temp_data_dir**
    **h5ad_input**
    **shuffles**
    **pinned_columns**
    **raw**
    **highly_variable_genes**
    **template**


### Description
Creates a cached `anndata.AnnData` (.h5ad) only with cells from the desired background.
A background is defined by a list of clusters (cell types) that share some simmilarities (Kenyon Cells, glia, etc)

### Detailed explanation

1. These backgrounds are expected to be stored in a folder called backgrounds under the `data_dir` folder specified in the configuration file
2. The cells are filtered according to whether their CellType annotation in the obs data frame of the AnnData matches one of the cell types in the background

A background file looks [like this](https://github.com/shaliulab/sleepml-data/blob/master/backgrounds/KC.csv). Only the cluster column is needed.

3. The input AnnData in serialized form (.h5ad) is expected under in the `data_dir` folder with name `h5ad_input`.
4. The new anndata.AnnData will be saved in a folder with name as stated in `temp_data_dir`.
5. An extra anndata.AnnData will be saved for every shuffle performed (`shuffles`). Columns in the parameter `pinned_columns` will not be shuffled.
    In our analysis we pinned the CellType, so that the shuffled datasets have cells retaining their cell type annotation, but not their batch.
6. The features fed to the pipeline will be the raw counts or not depending on the value of the `raw` parameter. In our analysis we set it to True.
7. If the param `highly_variable_genes` is True, only those will be used. In our analysis we set it to True.
8. If a `template` is provided, labels are merged or transformed into pseudo-labels. We did not use this feature in this analysis (can be ignored)

[These genes](https://github.com/shaliulab/sleepml-data/blob/master/batch_effects.txt) were ignored in the analysis because they were found to correlate with batches

## 2 get_marker_genes


### Parameters used

### Description

Iteratively removes markers genes with a fold change above an increasingly low threshold, so the cell types in the background become as homogeneous as possible. A Dimensionality Reduction plot is generated at each threshold

### Detailed explanation



## 3 remove_marker_genes

### Parameters used


### Description

Set the genes with a threshold higher than the selected threshold to 0

### Detailed explanation


## 4 train_models

### Parameters used


### Description

Have one or more of the supported models train using the transcriptomic data to predict sleep and wake

### Detailed explanation


## 5 predict

### Parameters used


### Description

predict the sleep / wake status of a cell from the same cell type, or a different one

### Detailed explanation
