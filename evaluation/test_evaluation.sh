#!/bin/bash


# PROJECT_PATH will be changed to the master branch of your repo
PROJECT_PATH='/rap/jvb-000-aa/COURS2019/etudiants/ift6759/projects/humanware/b1phut3/code'

RESULTS_DIR='/rap/jvb-000-aa/COURS2019/etudiants/submissions/b1phut3/evaluation/'
DATA_DIR='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/test_sample'
METADATA_FILENAME='/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/test_sample_metadata.pkl'

cd $PROJECT_PATH/evaluation
python eval.py --dataset_dir=$DATA_DIR --results_dir=$RESULTS_DIR --metadata_filename=$METADATA_FILENAME