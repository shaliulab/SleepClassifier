# Progressively remove marker genes starting from those with highest logFC
# Use a list of logFC thresholds, given in the config
# Only consider a marker gene if it's not shared by most celltypes in the background


import os.path
import yaml
from sleep_models.bin.get_marker_genes import get_marker_genes

with open("config.yaml", "r") as filehandle:
    config = yaml.load(filehandle, yaml.SafeLoader)

backgrounds = config["background"]
DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]
cache = config["cache"]

for background in backgrounds:
    print(background)
    max_clusters_per_marker = config["max_clusters"][background]
    thresholds = config["log2FC_thresholds"][background]

    get_marker_genes(
        h5ad_input=os.path.join(DATA_DIR, f"h5ad/{background}.h5ad"),
        output=os.path.join(RESULTS_DIR, f"{background}_get-marker-genes/"),
        max_clusters=max_clusters_per_marker,
        thresholds=thresholds,
        ncores=1,
        cache=cache,
    )
