# Create a dataset of single cells sharing a background
# e.g. KC only or glia only
# Optionally removes manually curated batch effect genes

import yaml
import os.path
from sleep_models.bin.make_dataset import make_dataset

SHUFFLES=0
H5AD_INPUT="data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad"

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

backgrounds = config["background"]
DATA_DIR = config["data_dir"]

for background in backgrounds:
    ## for KC
    make_dataset(
        h5ad_input=H5AD_INPUT,
        h5ad_output=os.path.join(DATA_DIR, "h5ad", f"{background}.h5ad"),
        random_state=1500,
        background=os.path.join(DATA_DIR, "backgrounds", f"{background}.csv"),
        batch_genes_file=os.path.join(DATA_DIR, "batch_effects.xlsx"),
        shuffles=SHUFFLES,
        raw=False,
    )