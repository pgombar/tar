#!/bin/bash

./extract_features.py ../data/trial-dataset/ train.dat
../svm_rank_linux64/svm_rank_learn -c 200 -t 2 train.dat model.dat
../svm_rank_linux64/svm_rank_classify train.dat model.dat predictions
./recover_rankings.py ../data/trial-dataset/ predictions rankings
./test-script/rank-scorer.py -i rankings -g ../data/trial-dataset/substitutions.gold-rankings
