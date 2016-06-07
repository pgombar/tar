#!/bin/bash

./extract_features.py ../data/test-data/ test.dat
../svm_rank_linux64/svm_rank_classify test.dat model.dat predictions_test
./recover_rankings.py ../data/test-data/ predictions_test rankings_test
./test-script/rank-scorer.py -i rankings_test -g ../data/test-data/substitutions.gold-rankings
