#!/bin/bash

./main.py
../svm_rank_linux64/svm_rank_learn train.dat model.dat
../svm_rank_linux64/svm_rank_classify train.dat model.dat predictions
./recover_rankings.py
./test-script/rank-scorer.py -i main_ranking -g ../data/trial-dataset/substitutions.gold-rankings
