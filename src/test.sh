#!/bin/bash

./main.py > main_ranking
./test-script/rank-scorer.py -i main_ranking -g ../data/trial-dataset/substitutions.gold-rankings
