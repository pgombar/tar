#!/usr/bin/python

import argparse

parser = argparse.ArgumentParser(
    description='Recover rankings from SVM Rank Classifier output.',
    epilog='Example: ./recover_rankings.py ../data/trial-dataset/ predictions rankings')

parser.add_argument('input_dir', help='Samples directory')
parser.add_argument('predictions', help='Output of the classifier')
parser.add_argument('rankings_file', help='Output file - Substitutions rankings')

args = parser.parse_args()
input_dir = args.input_dir
predictions = args.predictions
rankings_file = args.rankings_file

import utils

tasks = utils.parse_input_file(input_dir)
preds = open(predictions).read().split('\n')[:-1];

rankings = []
cur = 0
for (sentence, idx), subs in tasks:
    scores = []
    for sub in subs:
        scores.append((float(preds[cur]), sub))
        cur += 1 

    ranked = []
    for _, sub in sorted(scores):
        ranked.append(sub)
    rankings.append(ranked)
assert cur == len(preds)

utils.output(rankings_file, rankings)

