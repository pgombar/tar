#!/usr/bin/python

import nltk

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

import xml.etree.ElementTree as ET
import numpy as np


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
    """ Read word frequencies file and returns dictionary
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
    

def get_svm_line(gold, qid, features):
    """ Produce a string of the following SVM input format:
        3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A
    """
    features = [1, 2, 3]
    s = str(gold) + " qid:" + str(qid) + " "
    for i,f in enumerate(features):
        s += str(i+1) + ":" + str(f) + " "
    s += "# " + str(qid) + "\n"
    return s


def get_sentence_data(gold_subs, subs):
    """ Retrieve qid (sentence number) and the target value (gold for each sub).
        Gold is a string in the format of:
        "Sentence 1 rankings: {intelligent} {clever} {smart} {bright}",
        subs is a list of possible substitutions.
        Returns a list of (gold, qid) for each substitution.
    """
    ret = []
    qid = gold_subs.split()[1]
    gold_rank = map(lambda f: f.strip(), gold_subs.replace('}','').split('{')[1:])
    for sub in subs:
        # TODO skuziti kako parsirati {rijec, rijec2} jer ovdje nastaje problem
        try:
            gold = gold_rank.index(sub) + 1
        except ValueError:
            gold = -1
        ret.append((gold, qid))
    return ret


def output_svm_file(filepath, gold_subs_file, subs_file, features):
    """ Read gold rankings file and substitutions, parse them and
        output a file in SVM format.
        Features is a list of floats, filepath is the output file.
    """
    f = open(filepath, 'w')
    
    gold_subs_list = open(gold_subs_file).read()
    gold_subs_list = gold_subs_list.split('\n')
    if gold_subs_list[-1] == '':
        gold_subs_list.pop()
    
    subs_list = open(subs_file).read()
    subs_list = subs_list.split('\n')
    if subs_list[-1] == '':
        subs_list.pop()

    assert len(subs_list) == len(gold_subs_list)
    assert all([x[-1] == ';' for x in subs_list])
    
    subs_list = map(lambda x: x.split(';')[:-1], subs_list)
    
    for gold, sub in zip(gold_subs_list, subs_list):
        sentence_data = get_sentence_data(gold, sub)
        for gold, qid in sentence_data:
            # TODO promijeniti u features[i]
            f.write(get_svm_line(gold, qid, features))
    f.close()

