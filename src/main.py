#!/usr/bin/python

import utils
import word2vec
import numpy as np
from scorer import Scorer

input_dir = '../data/trial-dataset/'
output_file = 'main_ranking'
simple_wiki_freqs_file = '../simple_wiki/freqs.txt'

tasks = utils.parse_input_file(input_dir)

class ScorerInvLength(Scorer):
    def score(self, _, sub):
        return 1.0 / len(sub)

    
class ScorerSWFreqs(Scorer):
    freqs = {}
    def __init__(self):
        self.freqs = utils.read_freqs_file(simple_wiki_freqs_file)
    def score(self, _, sub):
        if sub in self.freqs:
            return self.freqs[sub]
        return 0

    
class ScorerContextSimilarity(Scorer):
    model = 0
    def __init__(self):
        self.model = word2vec.load('../simple_wiki/dump.bin')

    cache = {}
    def vector(self, w):
        if w not in self.cache:
            if w in self.model.vocab:
                self.cache[w] = self.model.get_vector(w)
            else:
                self.cache[w] = 100*[0.0]
        return self.cache[w]

    def score(self, (sentence, idx), sub):
        ret = 0
        for i, w in enumerate(sentence):
            if i != idx:
                ret += np.dot(self.vector(w), self.vector(sub))
        return ret

    
scorer = ScorerSWFreqs()
rankings = utils.rank_everything(scorer, tasks)
utils.output(output_file, rankings)

