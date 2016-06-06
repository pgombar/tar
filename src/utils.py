#!/usr/bin/python

import nltk

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

import xml.etree.ElementTree as ET
import numpy as np

import re


def parse_text(text):
    """ 
    1. tokenize
    2. convert to lowercase
    3. remove stopwords
    4. return list of words
    """

    stops = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = map(lambda f: f.lower(), tokens)
    tokens = filter(lambda f: f not in stops, tokens)
    return tokens


def parse_input_file(input_dir):
    """ Parses 'contexts.xml' and 'substitutions' files.
    
    Takes the directory of input files the only argument.
    
    Returns list of tasks in form of pairs (sentence, list of possible subs).
    Each sentence is a pair (list of strings, 0-based index of a string to be replaced).
    """

    tree = ET.parse(input_dir + 'contexts.xml')
    root = tree.getroot()
    
    ret = []
    for x in root.iter(tag='instance'):
        text = [c.strip() for c in x.itertext() if c.strip()]
        
        if len(text) == 3:
            pre = parse_text(text[0])
            sub = [' '.join(parse_text(text[1]))]
            suf = parse_text(text[2])
            ret.append((pre + sub + suf, len(pre)))
        else:
            if len(text[0]) < len(text[1]):
                sub = [' '.join(parse_text(text[0]))]
                suf = parse_text(text[1])
                ret.append((sub + suf, 0))
            else:
                pre = parse_text(text[0])
                sub = [' '.join(parse_text(text[1]))]
                ret.append((pre + sub, len(pre)))
            
    subs = open(input_dir + 'substitutions').read()
    subs = subs.split('\n')
    if subs[-1] == '':
        subs.pop()
    assert len(subs) == len(ret)

    assert all([x[-1] == ';' for x in subs])
    subs = map(lambda x: x.split(';')[:-1], subs)
    
    return zip(ret, subs)
    
def output(filepath, rankings):
    """ Writes rankings to file """
    
    f = open(filepath, 'w')
    for idx, ranking in enumerate(rankings):
        out = map(lambda x: '{' + x + '}', ranking)
        out = ' '.join(out)
        f.write('Sentence {} rankings: {}\n'.format(idx + 1, out))


def read_freqs_file(filepath, separator=' '):
    """ Reads word frequencies file and returns dictionary
    where words are keys and frequencies are values.
    """
    f = open(filepath, 'r')
    ret = {}
    for line in f.read().split('\n'):
        tokens = line.split(separator)
        if len(tokens) == 2:
            ret[tokens[0]] = int(tokens[1])
    return ret
    

def lemmatizate_freqs_file(inp_file, out_file):
    lemmatizater = WordNetLemmatizer()
    freqs = read_freqs_file(inp_file)

    nfreqs = {}
    for word in freqs:
        nword = lemmatizater.lemmatize(word)
        if nword not in nfreqs:
            nfreqs[nword] = 0
        nfreqs[nword] += freqs[word]

    f = open(out_file, 'w')
    for word in sorted(nfreqs.keys()):
        f.write('{} {}\n'.format(word, nfreqs[word]))
    

def rank_everything(scorer, tasks):
    return map(lambda ((a, b), c): scorer.rank((a, b), c), tasks)
    

def get_svm_line(gold, qid, sub, features):
    """ Produce a string of the following SVM input format:
        3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A
    """
    s = "{} qid:{}".format(gold, qid)
    for idx, feature in enumerate(features):
        s += " {}:{}".format(idx + 1, feature)
    s += " # {}_{}\n".format(qid, sub)
    return s


def output_svm_file(filepath, gold_rankings, tasks, features_fun):
    """ Output a file in SVM rank format.
        features_fun is a function that takes (sentence, idx) and sub as 
        arguments and returns list of floats (features).
    """
    f = open(filepath, 'w')
    
    assert len(tasks) == len(gold_rankings)
    
    for qid, (gold, task) in enumerate(zip(gold_rankings, tasks)):
        (sentence, idx), subs = task
        for sub in subs:
            features = features_fun((sentence, idx), sub)
            f.write(get_svm_line(gold[sub], qid, sub, features))
    f.close()

    
def parse_rankings_file(filename):
    pattern = re.compile('.*?\{(.*?)\}(.*)')
    
    allContextRankings = []
    for line in open(filename):
	rest = line
	currentContextRanking = {}
	counter = 1
	while rest:
	    match = pattern.search(rest)
	    currentRank = match.group(1)
	    individualWords = currentRank.split(', ')
	    for word in individualWords:
		word = re.sub('\s$','',word)
		currentContextRanking[word] = counter
	    rest = match.group(2)
	    counter += 1
		
	allContextRankings.append(currentContextRanking)
    
    return allContextRankings
