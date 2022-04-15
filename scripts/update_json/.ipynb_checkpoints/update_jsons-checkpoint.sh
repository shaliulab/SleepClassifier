#! /bin/bash

# This script generates a json file that each notebook can load
# to generate a snakemake-like object with a set of default values

# Takes:
# * a ROOT_DIR where the project lives


# From ROOT_DIR it derives:
# * a directory where rule files live -> $ROOT_DIR/rules/foo.smk
# * a json file in the same directory as the rule and with the same name (only .json instead of .smk) -> $ROOT_DIR/rules/foo.json
# The json should contain default values for wildcards used in the rule
# * a Snakefile -> $ROOT_DIR/Snakefile

# It repeats the same operation for a list of rules
# These rules should be described in the rule files in the $ROOT_DIR/rules
# The rule name should appear within the 10 first rows of the rule file

ROOT_DIR=$1
RULES_DIR="rules"

for RULE in "MLP";
do
    # the Snakemake rule filename and the rule inside it must be called the same
    RULECOUNT=$(head -v -n 10 rules/*.smk | grep "$RULES_DIR/.*$RULE.smk" | sed 's/==> '$RULES_DIR'\/\(.*\)-'$RULE'.smk <==/\1/g')
    printf "%-5s" $RULECOUNT 
    printf "%-30s"  " - $RULE"
    # This will put together a new json in the notebooks dir containing a full set of snakemake inputs, outputs, params etc
    # based off the wildcard mapping presented on the derived json.
    CMD="python update_json.py $ROOT_DIR/notebooks/$RULECOUNT-$RULE.json --snakefile $ROOT_DIR/Snakefile  --rule $RULE --root $ROOT_DIR --json $ROOT_DIR/$RULES_DIR/$RULECOUNT-$RULE.json"
    echo ""
    echo $CMD
    eval $CMD
    printf " Done\n"
done
