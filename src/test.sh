#!/bin/bash

./main.py
./test-script/rank-scorer.py -i main_ranking -g ../data/trial-dataset/substitutions.gold-rankings
