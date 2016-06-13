#!/usr/bin/python

import argparse
import random

input_dir = '../data/test-data/'
rankings = 'trainer_rbf_rankings'

import utils

tasks = utils.parse_input_file(input_dir)
rankings = utils.parse_rankings_file(rankings)
assert len(tasks) == len(rankings)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


random.seed(1)
filtered = [1000, 715, 741, 692, 98, 1142]

while True:
    choice = raw_input("> ")

    if choice == 'a':
        filtered = range(0, len(tasks) - 1)
    else:
        if len(choice):
            break


    i = random.choice(filtered)
    while True:
        (sentence, idx), subs = tasks[i]
        if len(sentence) < 13:
            break
        i = random.choice(filtered)

    (sentence, idx), subs = tasks[i]
    def getOut(caption, sentence, idx, sub, color):
        out = caption
        while len(out) < 20:
            out += " "
        out += ": "
        for i, word in enumerate(sentence):
            if i == idx:
                out += color + sub + bcolors.ENDC
            else:
                out += word
            out += " "
        return out
    
    print getOut('Input', sentence, idx, sentence[idx], bcolors.FAIL)
    for i, sub in enumerate(rankings[i]):
        print getOut(str(i+1) + '.', sentence, idx, sub, bcolors.OKGREEN)
