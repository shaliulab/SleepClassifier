import os.path
from sleep_models.utils.utils import load_pipeline_config, save_pipeline_config
from sleep_models.predict import replicate
from sleep_models.bin.make_matrixplot import make_matrixplot_main as make_matrixplot

config = load_pipeline_config()

DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]
SECONDARY_TARGET = config.get("secondary_target", None)

def predict(background_folder, architectures, random_states, predict=True, metrics=["accuracy", "recall", "f_sleep"], save_results=True, **kwargs):

    for arch in architectures:
        arch_folder = os.path.join(background_folder, arch)

        if predict:
            for random_state in random_states:

                replicate_folder=os.path.join(arch_folder, f"random-state_{random_state}")

                if not os.path.exists(replicate_folder):
                    raise Exception(f"Please make sure {replicate_folder} exists")

                # TODO: Change this ugly code to something nice!
                background = os.path.basename(background_folder.rstrip("/")).replace("-train", "")
                # TODO Change how we remove the _shuffled_0 instead of doing this crap
                background = background.split("_")[0]
                tables = replicate(
                    background=os.path.join(DATA_DIR, "backgrounds", f"{background}.csv"),
                    replicate_folder=replicate_folder,
                    ncores=3
                )

                if save_results:
                    for metric in tables:
                        # save to disk
                        tables[metric].to_csv(
                            os.path.join(replicate_folder, f"{metric}.csv")
                        )

        # for metric in tables:
        for metric in metrics:
            plotting_kwargs = kwargs.copy()
            if metric != "accuracy":
                plotting_kwargs["barlimits"]=None

            fig = make_matrixplot(
                prediction_results=arch_folder,
                metric=metric,
                **plotting_kwargs
            )


def main():

    for background in config["background"]:
        background_folder = os.path.join(RESULTS_DIR, f"{background}-train")
        CONFIG_DOCUMENTATION = os.path.join(background_folder, "05_predict.yml")
        predict(
            background_folder, config["arch"],
            config["seeds"], metrics=config["metrics"],
            predict=config["predict"],
            barlimits=config["barlimits"][background],
            **config["plotting_kwargs"]
        )
        save_pipeline_config(config, dest=CONFIG_DOCUMENTATION)

        for i in range(config["shuffles"]):
            background_folder = os.path.join(RESULTS_DIR, f"{background}_shuffled_{i}-train")
            predict(
                background_folder, config["arch"],
                config["seeds"], metrics=config["metrics"],
                predict=config["predict"], barlimits=config["barlimits"][background],
                **config["plotting_kwargs"]
            )

if __name__ == "__main__":
    main()
