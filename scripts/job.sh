#! /bin/bash

ARGS=$1 
ARGS_ARR=($ARGS)
H5AD_INPUT=${ARGS_ARR[0]}
CLUSTER=${ARGS_ARR[1]}
BACKGROUND=${ARGS_ARR[2]}
EXCLUDE=${ARGS_ARR[3]}
RESULTS=${ARGS_ARR[4]}
SEED=${ARGS_ARR[5]}
LOGFILE=$(echo "${ARGS}.log" | tr "/" "-" | tr " " "_")
echo $LOGFILE

CMD="source ~/.bashrc_conda && conda activate TF && train-model  $CLUSTER --h5ad-input $H5AD_INPUT --background $BACKGROUND --exclude-genes-file $EXCLUDE --results $RESULTS --seed $SEED --verbose 20 --logfile logs/$LOGFILE"

echo $CMD
eval $CMD
