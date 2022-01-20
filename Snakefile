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


configfile: "config.yaml"
BACKGROUND = list(config["background"].keys())[0]
LOG_FOLDER = "results/logs"
INPUT_H5AD = "data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad"


# write to the config which cell types belong to the
# background selected
# this information is available at "data/backgrounds/BACKGROUND.csv
config["background"][BACKGROUND] = pd.read_csv(
    os.path.join(
        "data", "backgrounds", BACKGROUND + ".csv"
    )
)["cluster"].tolist()




#rule homogenize:
#    input:
#        f"results/{BACKGROUND}-homogenization/"
#
#rule train_and_plot:
#    input:
#        #os.path.join(f"results/{BACKGROUND}-models", config['model']),
##        expand(os.path.join(f"results/{BACKGROUND}-models", config['model'], "random-state_{seed}", "accuracy.csv"), seed=config["seeds"]),
#        expand(os.path.join(f"results/{BACKGROUND}-models", config["model"], "matrixplot.{ext}"), ext=config["extensions"])


rule make_single_cell_dataset:
    input:
        h5ad = INPUT_H5AD,
        background = f"data/backgrounds/{BACKGROUND}.csv"
    output:
        h5ad = f"results/{BACKGROUND}-data/{BACKGROUND}.h5ad"
    threads: 1
    log:
        f"{LOG_FOLDER}/make_dataset_{BACKGROUND}.log"
    run:
        from sleep_models.bin.make_dataset import make_dataset

        make_dataset(
            h5ad=input.h5ad,
            output=output.h5ad,
            background=input.background,
            seed=1000,
            batch_genes_file=config["batch_genes_file"],
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
    priority: 1
    log:
        f"{LOG_FOLDER}/get_markers_{BACKGROUND}.log"
    run:
        from sleep_models.bin.get_marker_genes import get_marker_genes 

        thresholds = config["log2FC_thresholds"][BACKGROUND]

        print(f"Snakemake is computing the marker genes at following thresholds: {thresholds} ...")

        get_marker_genes(
            h5ad=input[0],
            output=output[0],
            thresholds = thresholds,
            max_clusters = config["max_clusters"][BACKGROUND]
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
        f"{LOG_FOLDER}/remove_markers_{BACKGROUND}.log"
    run:
        from sleep_models.bin.remove_marker_genes import remove_marker_genes 

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
        #h5ad = f"results/{BACKGROUND}-data/{BACKGROUND}-no-marker-genes.h5ad",
        h5ad = rules.remove_marker_genes.output,
        background = f"data/backgrounds/{BACKGROUND}.csv"
    output:
        #directory(f"results/{BACKGROUND}-models/" + config["model"]),
        directory(expand(f"results/{BACKGROUND}-models/" + config["model"] + "/random-state_{seed}", seed=config["seeds"]))
    log:
       f"{LOG_FOLDER}/train_models_{BACKGROUND}.log"
    run:

        kwargs = {
            "exclude_genes_file":  config["exclude_genes_file"],
            "highly_variable_genes": config["highly_variable_genes"],
            "model_name": config["model"],
            "verbose": 20,
            "seeds": config["seeds"]
        }

        from sleep_models.bin.train_models import train_models 
       
        train_models(
            h5ad_input = input.h5ad[0],
            output = os.path.dirname(output[0]),
            background = input.background,
            clusters=None, #all
            ncores=config["ncores"],
            **kwargs
        )


rule predict_sleep:
    """
    Given a set of trained models, each on a given cell type,
    try performing the same type of prediction
    on other cells of this cell type AND also other cell types  
    """
    input:
        #f"results/{BACKGROUND}-models/" + config['model']
        rules.train_models.output
        #expand(os.path.join(f"results/{BACKGROUND}-models", config['model'], "random-state_{seed}"), seed=config["seeds"])
    #output:
        #expand(os.path.join(f"results/{BACKGROUND}-models", config['model'], "random-state_{seed}", "accuracy.csv"), seed=config["seeds"])
    run:

        from sleep_models.bin.crosspredict import predict_sleep 

        predict_sleep(
            training_output = os.path.dirname(input[0]),
            ncores = config["ncores"],
            prediction_output = os.path.dirname(input[0]),
        )


rule make_matrixplot:
     """
     Plot the results of the pipeline
     """
     input:
       expand(os.path.join(f"results/{BACKGROUND}-models", config['model'], "random-state_{seed}", "accuracy.csv"), seed=config["seeds"])
     output:
       expand(os.path.join(f"results/{BACKGROUND}-models", config["model"], "matrixplot.{ext}"), ext=config["extensions"])
     run:

       from sleep_models.bin.make_matrixplot import make_matrixplot_main as make_matrixplot

       plotting_kwargs = config["plotting_kwargs"]

       make_matrixplot(
           prediction_results = os.path.dirname(os.path.dirname(input[0])),
           **plotting_kwargs
       )

