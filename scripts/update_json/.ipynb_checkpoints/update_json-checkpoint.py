"""
Take as input a Snakemake rule (smk) and its wildcards (json)
Return a new json with all inputs and outputs populated so it can be loaded in a notebook,
providing Snakemake-like interactivity
"""

import argparse
import json

import snakemk_util

ap = argparse.ArgumentParser()
ap.add_argument("--snakefile")
ap.add_argument("--rule")
ap.add_argument("--root", default=".")
ap.add_argument("--json", default=None)
ap.add_argument("output", help="Output json file")

args = ap.parse_args()

us = snakemk_util.UniversalSnakemake(args.snakefile)

if args.json is None:
    defaults = None
else:
    with open(args.json, "r") as fh:
        defaults = json.load(fh)

us.export_rule(args.rule, args.output, defaults=defaults, root=args.root)
