#!/usr/bin/python

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import utils
import math
import numpy as np
from scorer import *


scorers = [
    ScorerInvLength(),
    ScorerWordNet(),
    ScorerSWFreqs(),
    ScorerCorpusComplexity(),
    ScorerContextSimilarity(),
]

def get_features(tasks):
    features = []
    for (sentence, idx), subs in tasks:
        for sub in subs:
            features.append(map(lambda scorer: scorer.score((sentence, idx), sub), scorers))
    return features

print 'initialization done'

coefs = [1,1,1,1,1]
    
test_dir = '../data/test-data/'
test_tasks = utils.parse_input_file(test_dir)
test_gold_file = test_dir + 'substitutions.gold-rankings'
test_gold_rankings = utils.parse_rankings_file(test_gold_file)

features = get_features(test_tasks)
transformer = MinMaxScaler().fit(features)
features = transformer.transform(features)

rankings = utils.recover_rankings(test_tasks, -np.dot(features, coefs))
print utils.evaluate(rankings, test_gold_rankings)
utils.output('unsupervised_rankings', rankings)
