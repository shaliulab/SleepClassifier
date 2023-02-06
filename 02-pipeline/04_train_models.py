# train-torch-model y --output results/KC-models/500_neurons_NN/ --seed 1000 --model-name NeuralNetwork --h5ad-input results/KC-data/KC-raw.h5ad --training-config training_params.yaml --label-mapping data/templates/simple_condition_mapping.yaml

#
import os.path
import logging

from sleep_models.utils.utils import load_pipeline_config, save_pipeline_config
from sleep_models.bin.train_model import train
from sleep_models.utils.utils import load_pipeline_config
from sleep_models.preprocessing.preprocessing import load_adata
from sleep_models.utils.data import sort_celltypes

config = load_pipeline_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
print(__name__)
TEMP_DATA_DIR = config["temp_data_dir"]
DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]
CONDITION_TEMPLATE = config["template"]
TARGET = config["target"]
SECONDARY_TARGET = config.get("secondary_target", None)

train_models_input = config.get("train_models_input", None)
if train_models_input is None:
    train_models_input = ""

if CONDITION_TEMPLATE is None:
    label_mapping=None
else:
    label_mapping=os.path.join(DATA_DIR, "templates", CONDITION_TEMPLATE)

h5ad_input = os.path.join(DATA_DIR, config["h5ad_input"])

if config["exclude_genes_file"] is None:
    exclude_genes_file=None
else:
    exclude_genes_file=os.path.join(DATA_DIR, config["exclude_genes_file"])

print(f"Loading {h5ad_input}")
adata = load_adata(
    h5ad_input,
    exclude_genes_file=exclude_genes_file,
)
print(" ... Done")

celltypes = list(set(adata.obs[config["celltype"]].tolist()))
celltypes = sort_celltypes(celltypes)


for celltype in celltypes:
    for arch in config["arch"]:
        for seed in config["seeds"]:
            output=os.path.join(RESULTS_DIR, f"{celltype}-train")
            output_file = os.path.join(output, arch, f"random-state_{seed}", f"{celltype}_confusion_table.csv")
            if os.path.exists(output_file):
                logger.warning(f"{output_file} exists")
                continue


            os.makedirs(output, exist_ok=True)
            print(f"{celltype} -> {output}")
            CONFIG_DOCUMENTATION = os.path.join(output, "04_train_models.yml")
            if celltype == "Bulk":
                adata_ = adata.copy()
            else:
                adata_ = adata[adata.obs[config["celltype"]] == celltype]

            if adata.shape[0] < 10:
                logger.warning(f"Cell type {celltype} has size {adata.shape[0]}. Ignoring")
                continue

            print(celltype)
            train(
                h5ad_input=adata_,
                arch=arch,
                output=output,
                cluster=celltype,
                random_state=seed,
                label_mapping=label_mapping,
                target=TARGET,
            )
            save_pipeline_config(config, dest=CONFIG_DOCUMENTATION)
