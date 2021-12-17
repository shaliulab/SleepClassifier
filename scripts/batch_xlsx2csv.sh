xlsx2csv data/batch_effects.xlsx  | tail -n +2 | cut -f 1 -d, > data/batch_effects.txt
