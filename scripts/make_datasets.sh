# preprocess
SHUFFLES=0


for BACKGROUND in "KC" "glia" "peptides"
do

    make-dataset \
        --h5ad-input "data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad" \
        --seed 1500 \
        --background "data/backgrounds/${BACKGROUND}_mapping.csv" \
        --batch-genes-file  "data/batch_effects.xlsx" \
        --template-file "data/template.json" \
        --shuffles $SHUFFLES &
done

## for KC
#make-dataset --h5ad-input data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad  --seed 1500 --background "data/backgrounds/KC_mapping.csv" --batch-genes-file  "data/batch_effects.xlsx" --shuffles $SHUFFLES &
## for Glia
#make-dataset --h5ad-input data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad  --seed 1500 --background "data/backgrounds/glia_mapping.csv" --batch-genes-file  "data/batch_effects.xlsx" --shuffles $SHUFFLES &
## for peptide clusters
#make-dataset --h5ad-input data/h5ad/Preloom/All_Combined_No_ZT2_Wake.h5ad  --seed 1500 --background "data/backgrounds/peptides_mapping.csv" --batch-genes-file  "data/batch_effects.xlsx" --shuffles $SHUFFLES &

