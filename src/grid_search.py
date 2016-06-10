#!/usr/bin/python

input_dir = '../data/trial-dataset/'
output_file = 'rankings'

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import math
from scorer import *

tasks = utils.parse_input_file(input_dir)


gold_file = input_dir + 'substitutions.gold-rankings'
gold_rankings = utils.parse_rankings_file(gold_file)

scorers = [
    ScorerInvLength(),
    ScorerWordNet(),
    ScorerSWFreqs(),
#    ScorerWFreqs(),
#    ScorerFreqsRatio(),
    ScorerContextSimilarity(),
#    ScorerSemanticSimilarity(),
]

def get_features(tasks):
    features = []
    for (sentence, idx), subs in tasks:
        for sub in subs:
            features.append(map(lambda scorer: scorer.score((sentence, idx), sub), scorers))
    return features

print 'initialization done'

coefs = np.zeros(len(scorers))
features = get_features(tasks)
transformer = MinMaxScaler().fit(features)
features = transformer.transform(features)

best = 0
best_coefs = []

K = 10

def rec(i):
    if i == len(scorers):
        global features
        rankings = utils.recover_rankings(tasks, -np.dot(features, coefs))
        score = utils.evaluate(rankings, gold_rankings)
        global best, best_coefs
        if score > best:
            print coefs, score
            best = score
            best_coefs = np.array(coefs, copy=True)
        return

    coefs[i] = 0
    while coefs[i] <= K:
        rec(i + 1)
        coefs[i] += 1

rec(0)
print best
print best_coefs
    
test_dir = '../data/test-data/'
test_tasks = utils.parse_input_file(test_dir)
test_gold_file = test_dir + 'substitutions.gold-rankings'
test_gold_rankings = utils.parse_rankings_file(test_gold_file)

features = get_features(test_tasks)
features = transformer.transform(features)

rankings = utils.recover_rankings(test_tasks, -np.dot(features, best_coefs))
print utils.evaluate(rankings, test_gold_rankings)
utils.output('grid_search_rankings', rankings)
