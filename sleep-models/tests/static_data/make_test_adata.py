import random

random.seed(1000)

import anndata
import numpy as np


N_CELLS = 500
N_GENES = 10

PATH_TO_ANNDATA = "../data/h5ad/Preloom/KC_mapping.h5ad"


def main():
    adata = anndata.read_h5ad(PATH_TO_ANNDATA)

    test_adata = adata[
        random.sample(range(adata.shape[0]), N_CELLS),
        random.sample(range(adata.shape[1]), N_GENES),
    ]

    new_adata = anndata.AnnData(X=test_adata.X, obs=test_adata.obs, raw=test_adata.raw)

    new_adata.write_h5ad("tests/static_data/test_adata.h5ad")
    # print(len(np.unique(new_adata.obs["Condition"])))


if __name__ == "__main__":
    main()
