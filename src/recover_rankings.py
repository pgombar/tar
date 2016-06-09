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

rankings = utils.recover_rankings(tasks, preds)
utils.output(rankings_file, rankings)

