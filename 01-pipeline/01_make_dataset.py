# Create a dataset of single cells sharing a background
# e.g. KC only or glia only
# Optionally removes manually curated batch effect genes

import os.path
import logging
from sleep_models.bin.make_dataset import make_dataset
from sleep_models.constants import HIGHLY_VARIABLE_GENES
from sleep_models.utils.utils import load_pipeline_config, save_pipeline_config
from sleep_models.utils.logging import apply_logging_config
apply_logging_config()

logger=logging.getLogger(__name__)

config = load_pipeline_config()

DATA_DIR = config["data_dir"]
TEMP_DATA_DIR = config["temp_data_dir"]
H5AD_INPUT = config["h5ad_input"]
SHUFFLES = config["shuffles"]
RAW=config["raw"]
HIGHLY_VARIABLE_GENES=config["highly_variable_genes"]
TEMPLATE_FILE=config["template"]
TEMPLATE_FILE=config["template"]
FEATURE_SELECTION_FILE=config.get("feature_selection_file", None)
if FEATURE_SELECTION_FILE is not None:
    FEATURE_SELECTION_FILE=os.path.join(DATA_DIR, FEATURE_SELECTION_FILE)


if TEMPLATE_FILE is None or not config["template_from_beginning"]:
    template_file=None
else:
    template_file=os.path.join(DATA_DIR, "templates", TEMPLATE_FILE)



os.makedirs(
    os.path.join(TEMP_DATA_DIR, 'h5ad'),
    exist_ok=True
)


for background in config["background"]:

    CONFIG_DOCUMENTATION = os.path.join(TEMP_DATA_DIR, "01_make_dataset.yml")

    logger.debug(f"Configuration: {config}")
    make_dataset(
        h5ad_input=os.path.join(DATA_DIR, H5AD_INPUT),
        h5ad_output=os.path.join(TEMP_DATA_DIR, "h5ad", f"{background}.h5ad"),
        random_state=1500,
        background=os.path.join(DATA_DIR, "backgrounds", f"{background}.csv"),
        batch_genes_file=os.path.join(DATA_DIR, config["batch_genes_file"]),
        feature_selection_file=FEATURE_SELECTION_FILE,
        shuffles=SHUFFLES,
        raw=RAW,
        highly_variable_genes=HIGHLY_VARIABLE_GENES,
        pinned_columns=config["pinned_columns"],
        # NOTE
        # Passing this here
        # means everything downstream (marker gene detection, DR, training, etc)
        # will happen only with cells that match the template file
        template_file=template_file,
    )
    save_pipeline_config(config, dest=CONFIG_DOCUMENTATION)
