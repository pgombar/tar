#!/usr/bin/python

import argparse

parser = argparse.ArgumentParser(
    description='Features extractor.',
    epilog='Example: ./extract_features.py ../data/trial-dataset/ train.dat')

parser.add_argument('input_dir', help='Samples directory')
parser.add_argument('output_file', help='Output file')

args = parser.parse_args()
input_dir = args.input_dir
output_file = args.output_file


import utils
import gensim
import numpy as np
from nltk.corpus import wordnet
from scorer import *

tasks = utils.parse_input_file(input_dir)

scorers = [
   ScorerInvLength(),
   ScorerWordNet(),
   ScorerSWFreqs(),
#   ScorerWFreqs(),
#   ScorerContextSimilarity(),
]

def features((sentence, idx), sub):
    return map(lambda scorer: scorer.score((sentence, idx), sub), scorers)

gold_file = input_dir + 'substitutions.gold-rankings'
gold_rankings = utils.parse_rankings_file(gold_file)
utils.output_features_file(output_file, gold_rankings, tasks, features)

