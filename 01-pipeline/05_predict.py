import os.path
from sleep_models.utils.utils import load_pipeline_config, save_pipeline_config
from sleep_models.predict import predict
from sleep_models.bin.make_matrixplot import make_matrixplot_main as make_matrixplot

config = load_pipeline_config()

DATA_DIR = config["data_dir"]
RESULTS_DIR = config["results_dir"]
SECONDARY_TARGET = config.get("secondary_target", None)

def predict_all_replicates(background, models, random_states, do_predict=True, metrics=["accuracy", "recall", "f_sleep"], save_results=True, shuffle=-1, **kwargs):
    f"""
    Predict the mapping between transcriptome and some label using the model and data stored in the background folder
    This is done for each replicate folder found
    The work is actually achived by the predict function

    Arguments:

        background (str): Name of a group of cell types with some common characteristics e.g. Kenyon Cells, glia, etc
        models (str): List of model names which should be evaluated
        random_states (tuple): List of seeds for which a replicate is available in the background's results folder
            The background folder must exist under RESULTS_DIR with name background-train or background_shuffled_X-train if it's a shuffled dataset
        
        do_predict (bool): If False, no models are loaded or evaluated, and the output plots are just remade
        metrics (list): Name of metrics to evaluate the models with
        save_results (bool): If True, the .csv with the value of each metric for a model trained on cell type X and evaluated on cell type Y are saved
            in the model_folder (under background folder) 

        shuffle (int): Whether the dataset is shuffled (>= 0) or not

    See sleep_models.predict.predict to learn how the model evaluation is performed
    """

    if shuffle >= 0:
        background_folder = os.path.join(RESULTS_DIR, f"{background}_shuffled_{shuffle}-train")
    else:
        background_folder = os.path.join(RESULTS_DIR, f"{background}-train")

    
    CONFIG_DOCUMENTATION = os.path.join(background_folder, "05_predict.yml")

    for model_name in models:
        model_folder = os.path.join(background_folder, model_name)

        if do_predict:
            for random_state in random_states:

                replicate_folder=os.path.join(model_folder, f"random-state_{random_state}")

                if not os.path.exists(replicate_folder):
                    raise Exception(f"Please make sure {replicate_folder} exists")

                # TODO: Change this ugly code to something nice!
                background = os.path.basename(background_folder.rstrip("/")).replace("-train", "")
                # TODO Change how we remove the _shuffled_0 instead of doing this
                background = background.split("_")[0]
                tables = predict(
                    background_path=os.path.join(DATA_DIR, "backgrounds", f"{background}.csv"),
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
                prediction_results=model_folder,
                metric=metric,
                **plotting_kwargs
            )

    save_pipeline_config(config, dest=CONFIG_DOCUMENTATION)



def main():

    for background in config["background"]:
        predict_all_replicates(
            background, config["arch"],
            config["seeds"], metrics=config["metrics"],
            do_predict=config["predict"],
            barlimits=config["barlimits"][background],
            **config["plotting_kwargs"],
            shufle=-1
        )

        for i in range(config["shuffles"]):
            predict_all_replicates(
                background, config["arch"],
                config["seeds"], metrics=config["metrics"],
                do_predict=config["predict"],
                barlimits=config["barlimits"][background],
                **config["plotting_kwargs"],
                shufle=i
            )


if __name__ == "__main__":
    main()
