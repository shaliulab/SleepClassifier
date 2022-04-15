MAX_CLUSTERS_PER_MARKER=(3 4 5)
i=0
for CLUSTER in "KC" "glia" "peptides"; do
    CMD="get-marker-genes --h5ad-input data/h5ad/Preloom/${CLUSTER}_mapping.h5ad --background data/backgrounds/${CLUSTER}_mapping.csv --max-clusters-per-marker ${MAX_CLUSTERS_PER_MARKER[$i]} --thresholds  7 6 5 4 3 2.9 2.8 2.7 2.6 2.5 2.3 2.2 2.1 2    --output 2021-11-05_get-marker-genes_${CLUSTER} --ncores 1 &"
    echo $CMD
    #eval $CMD
    let i++
done
