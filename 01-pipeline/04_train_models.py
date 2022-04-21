# train-torch-model y --output results/KC-models/500_neurons_NN/ --seed 1000 --model-name NeuralNetwork --h5ad-input results/KC-data/KC-raw.h5ad --training-config training_params.yaml --label-mapping data/templates/simple_condition_mapping.yaml

#
import yaml
import pandas as pd
import os.path

from sleep_models.bin.train_model import train

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

TEMP_DATA_DIR = config["temp_data_dir"]
DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]
CONDITION_TEMPLATE = config["template"]

if CONDITION_TEMPLATE == "None":
    label_mapping=None
else:
    label_mapping=os.path.join(DATA_DIR, "templates", CONDITION_TEMPLATE)

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
                    h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}-no-marker-genes.h5ad"),
                    arch=arch,
                    cluster=celltype,
                    output=os.path.join(RESULTS_DIR, f"{background}-train"),
                    random_state=seed,
                    highly_variable_genes=config["highly_variable_genes"], # used to be True always
                    label_mapping=label_mapping,
                )
