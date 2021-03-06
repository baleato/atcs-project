#!/bin/bash

# Emotion classification; data/semeval18_task1_class
if [[ ! -f SemEval18.zip ]]; then
  curl -Lo SemEval18.zip 'https://drive.google.com/uc?export=download&id=1GjvfEPGhCPc4dCLAohwZNNtG1vLIEIav'
  unzip SemEval18.zip
fi

# Sarcasm detection; data/twitter
if [[ ! -f sarcasm_twitter.zip ]]; then
  curl -Lo sarcasm_twitter.zip 'https://drive.google.com/uc?export=download&id=1eDJrerQdcMY2nGNOkMOi0vsAke2h_jEB'
  unzip sarcasm_twitter.zip
fi


# Offensive Language Identification Dataset (OLID); data/OLIDv1.0
if [[ ! -f OLIDv1.0.zip ]]; then
  curl -Lo OLIDv1.0.zip 'https://sites.google.com/site/offensevalsharedtask/olid/OLIDv1.0.zip?attredirects=0&d=1'
  unzip OLIDv1.0.zip -d OLIDv1.0
fi

# Abusive language detection; data/tweet_wassem
if [[ ! -f WaseemHovy.zip ]]; then
  curl -Lo WaseemHovy.zip 'https://drive.google.com/uc?export=download&id=1Dbvn7yJkS3Iurx92-7sfe0DsDgD18Dbf'
  unzip WaseemHovy.zip
fi

# Sentiment Classification; data/sem_eval_2015
if [[ ! -f tweets_output.zip ]]; then
  curl -Lo tweets_output.zip 'https://drive.google.com/uc?export=download&id=10G4Owb12txUo1HZqkwaxJ69-GHyWvPKN'
  unzip tweets_output.zip  -d sem_eval_2015
fi

# Irony subtask A; data/sem_eval_2018
if [[ ! -f SemEval2018-T3-train-taskA.txt.zip ]]; then
  curl -Lo SemEval2018-T3-train-taskA.txt.zip 'https://drive.google.com/uc?export=download&id=1ihc9XKBvSIgoGpsp8hrkL-8hEUMi-Tuv'
  unzip SemEval2018-T3-train-taskA.txt.zip -d sem_eval_2018
fi

# Irony subtask B; data/sem_eval_2018
if [[ ! -f SemEval2018-T3-train-taskB.txt.zip ]]; then
  curl -Lo SemEval2018-T3-train-taskB.txt.zip 'https://drive.google.com/uc?export=download&id=1qFwQ6LPfLIYRe0C9z8rGYDHFBzKT2ivv'
  unzip SemEval2018-T3-train-taskB.txt.zip -d sem_eval_2018
fi

# Stanford Politeness Corpus; data/stanford_politeness_2013
if [[ ! -f stanford_politeness_2013.zip ]]; then
  curl -Lo stanford_politeness_2013.zip 'https://drive.google.com/uc?export=download&id=1UvmaP6iwTHeNm4LVhgIraPFUMQDg8Tic'
  unzip stanford_politeness_2013.zip -d stanford_politeness_2013
fi
