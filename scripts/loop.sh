#! /bin/bash


#BACKGROUND="data/backgrounds/KC_mapping.csv"
BACKGROUND=$1
EXCLUDE_GENES=$2
H5AD=$3
H5AD_RANDOM=$4
TIMESTAMP_TODAY=$(date "+%Y-%m-%d")
I=0
cat $BACKGROUND  | tail -n +2 | cut -f 1 -d, > /tmp/clusters.txt
mapfile -t CLUSTERS < /tmp/clusters.txt

OUTPUT=$5



PARALLEL=1
SGE=0
rm -rf $OUTPUT
touch $OUTPUT

for H5AD_INPUT in "$H5AD" "$H5AD_RANDOM"; do
    for CLUSTER in ${CLUSTERS[*]}; do
        for SEED in 1000 2000 3000 ; do

            if [ $I -eq 0 ]; then
                RESULTS="${TIMESTAMP_TODAY}_notshuffled"
            elif [ $I -eq 1 ]; then
                RESULTS="${TIMESTAMP_TODAY}_shuffled"
            fi

            if [ $SGE -eq 1 ]; then
                bash job.sh  $H5AD_INPUT $CLUSTER $BACKGROUND $RESULTS $SEED
            elif [ $PARALLEL -eq 1 ]; then
                echo "$H5AD_INPUT $CLUSTER $BACKGROUND $EXCLUDE_GENES  $RESULTS $SEED" >> $OUTPUT
            fi
        done
    done
    let I++
done


#if [ $PARALLEL -eq 1 ]; then
#    parallel -P 5 ./job.sh :::: parallel_args.txt




