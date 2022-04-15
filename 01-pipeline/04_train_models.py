#
# train-models --h5ad-input data/h5ad/Preloom/KC_mapping_wo-marker-genes_log2FC_threshold-2.6.h5ad --background data/backgrounds/KC_mapping.csv --seed 1000 2000 3000 --ncores 5 --verbose 10
import yaml
import pandas as pd
import os.path

from sleep_models.bin.train_model import train

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]


def get_celltypes(background):

    return pd.read_csv(
        os.path.join(
            DATA_DIR, "backgrounds", f"{background}.csv"
        )
    )["cluster"].tolist()



for background in config["background"]:
    for arch in config["arch"]:
        for seed in [1000, 2000, 3000]:
            celltypes = get_celltypes(background)
            for celltype in celltypes:
                train(
                    h5ad_input=os.path.join(DATA_DIR, f"h5ad/{background}-no-marker-genes.h5ad"),
                    arch=arch,
                    cluster=celltype,
                    output=os.path.join(RESULTS_DIR, f"{background}-train"),
                    random_state=seed,
                    highly_variable_genes=True
                )

