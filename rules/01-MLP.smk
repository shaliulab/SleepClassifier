rule MLP:
    """
    Build a MultiLayer Perceptron model
    to perform binary classification of single cell sequencing datasets
    """

    input:
        loomfile = "/1TB/Cloud/Lab/Projects/SleepSignature/workflow/results/20201224/scran/{celltype}/{celltype}_grouping-Condition.loom"

    output:
        os.path.join(ROOT_DIR, "results", "{celltype}_{target}.h5")
        , os.path.join(ROOT_DIR, "results", "{celltype}_{target}.json")
        , summary_csv =  os.path.join(ROOT_DIR, "results", "{celltype}_{target}.csv")
        , test_str_txt = os.path.join(ROOT_DIR, "results", "{celltype}_{target}.txt")

    params:
        target = "{target}"
        , model =     os.path.join(ROOT_DIR, "results", "{celltype}_{target}")
        , random_labels = True

    notebook:
        os.path.join(ROOT_DIR, "notebooks", "01-MLP.py.ipynb")
       
