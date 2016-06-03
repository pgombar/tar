#!/usr/bin/python

import utils
import word2vec
import numpy as np
from scorer import Scorer

input_dir = '../data/trial-dataset/'
output_file = 'main_ranking'
simple_wiki_freqs_file = '../simple_wiki/freqs.txt'
wiki_freqs_file = '../wiki/freqs.txt'

tasks = utils.parse_input_file(input_dir)


class ScorerInvLength(Scorer):
    def score(self, _, sub):
        return 1.0 / len(sub)


class ScorerCorpusComplexity(Scorer):
    sw_freqs = {}
    w_freqs = {}
    def __init__(self):
        self.sw_freqs = utils.read_freqs_file(simple_wiki_freqs_file)
        self.w_freqs = utils.read_freqs_file(wiki_freqs_file, separator='\t')
    def score(self, _, sub):
        if sub in self.w_freqs and sub in self.sw_freqs:
            return 1.0 / (self.w_freqs[sub] / (self.sw_freqs[sub] * 1.0))
        return 0


class ScorerWFreqs(Scorer):
    freqs = {}
    def __init__(self):
        self.freqs = utils.read_freqs_file(wiki_freqs_file, separator='\t')
    def score(self, _, sub):
        if sub in self.freqs:
            return self.freqs[sub]
        return 0        
    

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
#            if i != idx:
#                ret += np.dot(self.vector(w), self.vector(sub))
            ret += np.dot(self.vector(w), self.vector(sub))
        return ret


class ScorerBiran(Scorer):
    s1 = 0
    s2 = 0
    def __init__(self):
        self.s1 = ScorerCorpusComplexity()
        self.s2 = ScorerInvLength()

    def score(self, (sentence, idx), sub):
        return self.s1.score((sentence, idx), sub) * self.s2.score((sentence, idx), sub)


class ScorerComb(Scorer):
    s1 = 0
    s2 = 0
    def __init__(self):
        self.s1 = ScorerSWFreqs()
        self.s2 = ScorerContextSimilarity()

    def score(self, (sentence, idx), sub):
        return self.s1.score((sentence, idx), sub) * self.s2.score((sentence, idx), sub)


scorer = ScorerContextSimilarity()
rankings = utils.rank_everything(scorer, tasks)
utils.output(output_file, rankings)

