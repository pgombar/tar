#!/usr/bin/python

import utils

input_dir = '../data/trial-dataset/'
output_file = 'main_ranking'

tasks = utils.parse_input_file(input_dir)

def rank_stupid(s):
    ret = []
    for _, subs in s:
        ret.append(sorted(subs, key=len))
    return ret

rankings = rank_stupid(tasks)

utils.output(output_file, rankings)
    
