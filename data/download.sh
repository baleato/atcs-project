#!/bin/bash

# Emotion classification; data/semeval18_task1_class
if [[ ! -f SemEval18.zip ]]; then
  curl -Lo SemEval18.zip 'https://drive.google.com/uc?export=download&id=1GjvfEPGhCPc4dCLAohwZNNtG1vLIEIav'
  unzip SemEval18.zip
fi

# Sarcasm detection; data/twitter
if [[ ! -f twitter_training.zip ]]; then
  curl -Lo twitter_training.zip 'https://github.com/EducationalTestingService/sarcasm/releases/download/v1.0/twitter_training.zip'
  unzip twitter_training.zip
fi

if [[ ! -f twitter_testing.zip ]]; then
  curl -Lo twitter_testing.zip 'https://github.com/EducationalTestingService/sarcasm/releases/download/v2.0/twitter_test.zip'
  unzip twitter_testing.zip -d twitter
  mv twitter/twitter_test.jsonl twitter/sarcasm_twitter_test.jsonl
fi

# Offensive language detection; data/offenseval
if [[ ! -f OffensEval.zip ]]; then
  curl -Lo OffensEval.zip 'https://drive.google.com/uc?export=download&id=1-ynErP5o7NeV_ZLH_RfQFrw1siJd15tt'
  unzip OffensEval.zip
fi

# Abusive language detection; data/tweet_wassem
if [[ ! -f WaseemHovy.zip ]]; then
  curl -Lo WaseemHovy.zip 'https://drive.google.com/uc?export=download&id=1Dbvn7yJkS3Iurx92-7sfe0DsDgD18Dbf'
  unzip WaseemHovy.zip
fi
