import utils
import gensim
import numpy as np
from nltk.corpus import wordnet


Word2Vec = gensim.models.Word2Vec

simple_wiki_freqs_file = '../simple_wiki/freqs.txt'
wiki_freqs_file = '../wiki/freqs.txt'
w2v_model_file = '../models/glove.6B.200d_reduced.txt'


class Scorer:
    def __init__(self):
        return
    
    def score(self, (sentence, idx), sub):
        assert False
        
    def rank(self, (sentence, idx), subs):
        scores = map(lambda sub: -self.score((sentence, idx), sub), subs)
        subs = sorted(zip(scores, subs))
        return map(lambda (_, sub): sub, subs)

    def rankMultiple(self, tasks):
        return map(lambda ((a, b), c): self.rank((a, b), c), tasks)

    
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
        self.model = Word2Vec.load_word2vec_format(w2v_model_file, binary=False)

    def similarity(self, a, b):
        if a in self.model and b in self.model:
            return self.model.similarity(a, b)
        return 0
    
    def score(self, (sentence, idx), sub):
        ret = 0
        for i, w in enumerate(sentence):
            self.similarity(w, sub)
        return ret
    
class ScorerSemanticSimilarity(Scorer):
    model = 0
    def __init__(self):
        self.model = Word2Vec.load_word2vec_format(w2v_model_file, binary=False)

    def similarity(self, a, b):
        if a in self.model and b in self.model:
            return self.model.similarity(a, b)
        return 0
    
    def score(self, (sentence, idx), sub):
        return self.similarity(sentence[idx], sub)

class ScorerWordNet(Scorer):
    def score(self, _, sub):
        return len(wordnet.synsets(sub))

