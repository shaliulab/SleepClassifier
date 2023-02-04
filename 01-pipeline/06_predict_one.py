import os.path
import argparse
from sleep_models.utils.utils import load_pipeline_config, save_pipeline_config
from sleep_models.predict import load_and_predict

config = load_pipeline_config()

DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--background", required=True)
    ap.add_argument("--cluster1", required=True)
    ap.add_argument("--cluster2", required=True)
    ap.add_argument("--random_state", type=int, required=True)
    return ap

def main():
    
    ap = get_parser()
    args = ap.parse_args()

    background_folder = os.path.join(RESULTS_DIR, f"{args.background}-train")
    arch_folder = os.path.join(background_folder, args.arch)
    replicate_folder=os.path.join(arch_folder, f"random-state_{args.random_state}")
    load_and_predict(replicate_folder, clusters=[args.cluster2], cluster_name=args.cluster1)


if __name__ == "__main__":
    main()
