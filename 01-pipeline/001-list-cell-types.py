import pandas as pd
import tqdm
import yaml
import os.path

from sleep_models.utils.utils import load_pipeline_config, save_pipeline_config
import sleep_models.preprocessing as pp


config = load_pipeline_config()

DATA_DIR = config["data_dir"]
H5AD_INPUT = config["h5ad_input"]

def list_cell_types(h5ad_input):

    adata  = pp.read_h5ad(h5ad_input)
    annotation=set(adata.obs["Cluster_ID_res8"].tolist())
    config["background"] = {}
    for cluster in tqdm.tqdm(annotation):

        background = pd.DataFrame({
            "cluster": [f"Cluster_{cluster}"],
            "louvain_res": ["Cluster_ID_res8"],
            "idx": [cluster]
        })
        background.to_csv(os.path.join(DATA_DIR, "backgrounds", f"Cluster_{cluster}.csv"), index=False)

        config["background"][f"Cluster_{cluster}"] = None

    save_pipeline_config(config, dest=None)

if __name__ == "__main__":
    list_cell_types(
        h5ad_input=os.path.join(DATA_DIR, H5AD_INPUT),
    )
