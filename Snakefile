# type: ignore
## SNAKEFILE
## Snakefile for analysis of scRNA-seq dataset in Dopp. et al 2021
## 
## A Snakefile instructs Snakemake http://snakemake.readthedocs.io/
## how to run a pipeline written in Python/R/Julia
##
## It is invoked by running `snakemake --cores NCORES` on the directory where this file lives
## See software requirements 
import argparse
import datetime
import os.path
import sys

import pandas as pd

ROOT_DIR = os.getcwd()

configfile: "config.yaml"

BACKGROUND = list(config["background"].keys())[0]


config["background"][BACKGROUND] = pd.read_csv(
    os.path.join(
        "data", "backgrounds", BACKGROUND + ".csv"
    )
)["cluster"].tolist()



from sleep_models.bin.make_dataset import make_dataset
from sleep_models.bin.get_marker_genes import get_marker_genes 
from sleep_models.bin.remove_marker_genes import remove_marker_genes 
from sleep_models.bin.train_models import train_models 
from sleep_models.bin.crosspredict import predict_sleep 
from sleep_models.bin.make_matrixplot import make_matrixplot 

rule make_single_cell_dataset:
    input:
        h5ad = "data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad",
        background = f"data/backgrounds/{BACKGROUND}.csv"
    output:
        h5ad = f"results/{BACKGROUND}-data/{BACKGROUND}.h5ad"
    threads: 1
    log:
        f"logs/make_dataset_{BACKGROUND}.log"
    run:
        make_dataset(
            h5ad=input.h5ad,
            output=output.h5ad,
            background=input.background,
            seed=1000,
            raw=config["raw"],
            logfile=log[0],
            verbose=20,
            template_file=None
        )

rule get_marker_genes:
    """
    Given:
        1. an input h5ad for a whole background
        2. lists of marker genes for each cell type in the background
        3. a list of FChange thresholds
    find the genes that are to be considered marker genes of the background
    and save them to 
    """
    input:
        h5ad = f"results/{BACKGROUND}-data/{BACKGROUND}.h5ad"
    output:
        directory(f"results/{BACKGROUND}-homogenization/")
    log:
        f"logs/get_markers_{BACKGROUND}.log"
    run:

        thresholds = config["log2FC_thresholds"][BACKGROUND]

        print(f"Snakemake is computing the marker genes at following thresholds: {thresholds} ...")

        get_marker_genes(
            h5ad=input[0],
            output=output[0],
            thresholds = thresholds,
            min_clusters = config["min_clusters"][BACKGROUND]
        )


rule remove_marker_genes:
    """
    Remove the passed marker genes from the dataset
    """

    input:
        h5ad = f"results/{BACKGROUND}-data/{BACKGROUND}.h5ad"
    output:
        f"results/{BACKGROUND}-data/{BACKGROUND}-no-marker-genes.h5ad"
    log:
        f"logs/remove_markers_{BACKGROUND}.log"
    run:

        remove_marker_genes(
            h5ad = input[0],
            marker_gene_file = os.path.join(
                "results",
                f"{BACKGROUND}-homogenization",
                f"threshold-{config['user_defined_log2FC_threshold'][BACKGROUND]}",
                "marker_genes.txt"
            ),
            output = output[0]
        )


rule train_models:
    """
    Train a Sleep Wake classifier on a particular cell type of the background
    """

    input:
        h5ad = f"results/{BACKGROUND}-data/{BACKGROUND}-no-marker-genes.h5ad",
        background = f"data/backgrounds/{BACKGROUND}.csv"
    output:
        directory(f"results/{BACKGROUND}-models/" + config["model"])
    log:
       f"logs/train_models_{BACKGROUND}.log"
    run:

        kwargs = {
            "h5ad_input": input.h5ad,
            "exclude_genes_file":  config["exclude_genes_file"],
            "highly_variable_genes": config["highly_variable_genes"],
            "output" : output[0],
            "model_name": config["model"],
            "verbose": 20
            "seeds": config["seeds"]
        }
        
        train_models(
            background = input.background,
            clusters=None, #all
            ncores=-2,
            **kwargs
        )


rule all:
    input: rules.make_single_cell_dataset.input.h5ad
    output: rules.train_models.output[0]

# rule predict_sleep:
#     """
#     Given a set of trained models, each on a given cell type,
#     try performing the same type of prediction
#     on other cells of this cell type AND also other cell types  
#     """
#     pass


# rule make_matrixplot:
#      """
    #    Plot the results of the pipeline
#      """
