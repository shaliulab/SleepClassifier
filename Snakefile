## SNAKEFILE
## Snakefile for analysis of scRNA-seq dataset in Dopp. et al 2021
## 
## A Snakefile instructs Snakemake http://snakemake.readthedocs.io/
## how to run a pipeline written in Python/R/Julia
##
## It is invoked by running `snakemake --cores NCORES` on the directory where this file lives
## See software requirements 
import datetime
import os.path
import sys
ROOT_DIR = os.getcwd()


celltypes = ["y_KCs", "a_b_KCs", "a_b_prime_KCs", "Surface_Glia", "Astrocytes", "Ensheathing_Glia_17", "Ensheathing_Glia_19"]
targets = ["Run", "Genotype", "Treatment"]

###################################################
# Rules
###################################################

include : "rules/01-MLP.smk"

###################################################
# Expected outputs of the pipeline
###################################################

rule all:
    input:
        # Filter loom files with select_cells
        expand(rules.MLP.output.summary_csv, celltype=celltypes, target=targets)

