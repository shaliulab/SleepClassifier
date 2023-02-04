# train-torch-model y --output results/KC-models/500_neurons_NN/ --seed 1000 --model-name NeuralNetwork --h5ad-input results/KC-data/KC-raw.h5ad --training-config training_params.yaml --label-mapping data/templates/simple_condition_mapping.yaml

#
import pandas as pd
import os.path
import logging
import joblib

from sleep_models.utils.utils import load_pipeline_config, save_pipeline_config
from sleep_models.bin.train_model import train
from sleep_models.utils.utils import load_pipeline_config
config = load_pipeline_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TEMP_DATA_DIR = config["temp_data_dir"]
DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]
TEMPLATE_FILE = config["template"]
TARGET = config["target"]
SECONDARY_TARGET = config["secondary_target"]
N_JOBS=1
train_models_input = config["train_models_input"]
ALLOW_WITH_MARKERS=config.get("allow_with_markers", False)
STRATIFY=config.get("stratify", True)



if TEMPLATE_FILE is None:
    template_file=None
else:
    template_file=os.path.join(DATA_DIR, "templates", TEMPLATE_FILE)


def get_celltypes(background):

    return pd.read_csv(
        os.path.join(
            DATA_DIR, "backgrounds", f"{background}.csv"
        ), index_col=0, comment="#"
    ).index.tolist()


def train_loop(background, arch, seed):
    celltypes = get_celltypes(background)
    for celltype in celltypes:
        output=os.path.join(RESULTS_DIR, f"{background}-train")

        CONFIG_DOCUMENTATION = os.path.join(output, "04_train_models.yml")
        h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}{train_models_input}.h5ad")
        if ALLOW_WITH_MARKERS and not os.path.exists(h5ad_input):
            h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}.h5ad")
        print(f"Reading {h5ad_input}")

        train(
            h5ad_input=h5ad_input,
            arch=arch,
            output=output,
            cluster=celltype,
            random_state=seed,
            template_file=template_file,
            target=TARGET,
            stratify=STRATIFY,
        )
        save_pipeline_config(config, dest=CONFIG_DOCUMENTATION)


        for i in range(config["shuffles"]):
            h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}_shuffled_{i}{train_models_input}.h5ad")
            if ALLOW_WITH_MARKERS and not os.path.exists(h5ad_input):
                h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}.h5ad")

            print(f"Reading {h5ad_input}")
            train(
                h5ad_input=h5ad_input,
                arch=arch,
                cluster=celltype,
                output=os.path.join(RESULTS_DIR, f"{background}_shuffled_{i}-train"),
                random_state=seed,
                template_file=template_file,
                target=TARGET,
                stratify=STRATIFY,
            )

    if SECONDARY_TARGET is not None:
        for target in SECONDARY_TARGET:
            output = os.path.join(RESULTS_DIR, target, f"{background}-train")
            CONFIG_DOCUMENTATION = os.path.join(output, "04_train_models_secondary-target.yml")
            h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}{train_models_input}.h5ad")
            if ALLOW_WITH_MARKERS and not os.path.exists(h5ad_input):
                h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}.h5ad")

            print(f"Reading {h5ad_input}")
            train(
                h5ad_input=h5ad_input,
                arch=arch,
                cluster=None,
                output=output,
                random_state=seed,
                template_file=template_file,
                target=target,
                stratify=STRATIFY,
            )
            for i in range(config["shuffles"]):
                h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}_shuffled_{i}{train_models_input}.h5ad")
                if ALLOW_WITH_MARKERS and not os.path.exists(h5ad_input):
                    h5ad_input=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}.h5ad")


                print(f"Reading {h5ad_input}")
                train(
                    h5ad_input=h5ad_input,
                    arch=arch,
                    cluster=None,
                    output = os.path.join(RESULTS_DIR, target, f"{background}_shuffled_{i}-train"),
                    random_state=seed,
                    template_file=template_file,
                    target=target,
                    shuffle=i,
                   stratify=STRATIFY,
                )                
            save_pipeline_config(config, dest=CONFIG_DOCUMENTATION)



def main():
    for background in config["background"]:
        for arch in config["arch"]:
            joblib.Parallel(n_jobs=N_JOBS)(
                joblib.delayed(
                    train_loop
                )(
                    background, arch, seed
                ) for seed in config["seeds"]
            )
            

if __name__ == "__main__":
    main()