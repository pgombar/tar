#!/usr/bin/python

import utils
import gensim
import numpy as np
from scorer import Scorer

Word2Vec = gensim.models.Word2Vec

input_dir = '../data/trial-dataset/'
output_file = 'main_ranking'
simple_wiki_freqs_file = '../simple_wiki/freqs.txt'
wiki_freqs_file = '../wiki/freqs.txt'
model_file = '../models/glove.6B.200d.txt'
svm_file = 'train.dat'
predictions_file = 'predictions'

tasks = utils.parse_input_file(input_dir)

preds = open(predictions_file).read().split('\n')[:-1];

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
    
utils.output(output_file, rankings)

