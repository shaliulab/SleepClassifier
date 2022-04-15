Pipeline to train a model on different cell types, learning the transcriptomic signature of sleep vs wake
and have it predict it on the same and all other celltypes
Produces a matrixplot that summarises the accuracy of the model depending on which cell type's data was used to train it,
and which cell type's data was used to predict.
A diagonal is expected to arise if the sleep signature has some cell specific component.
 