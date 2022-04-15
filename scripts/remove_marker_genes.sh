THRESHOLDS=(2.6 6.0 2.0)
MAX_CLUSTERS_PER_MARKER=(3 4 5)
TIMESTAMP=$(date "+%Y-%m-%d")


i=0
for BACKGROUND in "KC" "glia" "peptides"; do
    CMD="remove-marker-genes --working-directory  ${TIMESTAMP}_get-marker-genes_${BACKGROUND}  --h5ad-output data/h5ad/Preloom/${BACKGROUND}_mapping_wo-marker-genes_log2FC_threshold-${THRESHOLDS[$i]}.h5ad --h5ad-input  data/h5ad/Preloom/${BACKGROUND}_mapping.h5ad --threshold ${THRESHOLDS[$i]} --background data/backgrounds/${BACKGROUND}_mapping.csv --max-clusters-per-marker ${MAX_CLUSTERS_PER_MARKER[$i]} &"
    echo $CMD
    let i++
done
