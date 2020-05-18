## Introduction

This repository performs meta-learning across multiple NLP tasks focusing on pragmatics and social meaning (emotion, sarcasm, politeness, abusive language detection).

## Dependencies

Python 3, PyTorch, Transformers, Tokenizers

## Tasks & datasets

- Emotion classification - [SemEval-2018 Task 1: Affect in Tweets (AIA-2018)](https://competitions.codalab.org/competitions/17751)
- Sarcasm detection - [The Shared Task on Sarcasm Detection (ACL 2020)](https://competitions.codalab.org/competitions/22247)
- Sentiment classification
  - [ ] Find a suitable dataset which might fit nicely with the other datasets. We can take a look for instance to [The Big Bad NLP Database](https://datasets.quantumstat.com/)
- Abusive language detection: [data-twitter-wh; Waseem and Hovy 2016](https://github.com/zeerakw/hatespeech)
- Politeness: [Stanford Politeness Corpus](http://www.cs.cornell.edu/~cristian/Politeness.html).
  :exclamation:	This dataset is based on web-forum interactions and not Twitter data. It might not fit nicely with the other datasets.

To download the datasets go to the data directory and execute the download script:

```sh
$ cd data
$ sh download.sh
```

Overview of datasets files:
```tree
data/
├── download.sh
├── OLIDv1.0
│   ├── README.txt
│   ├── labels-levela.csv
│   ├── labels-levelb.csv
│   ├── labels-levelc.csv
│   ├── olid-annotation.txt
│   ├── olid-training-v1.0.tsv
│   ├── testset-levela.tsv
│   ├── testset-levelb.tsv
│   └── testset-levelc.tsv
├── semeval18_task1_class
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
├── tweet_wassem
│   └── twitter_data_waseem_hovy.csv
└── twitter
    └── sarcasm_detection_shared_task_twitter_training.jsonl
```

## Running examples

TBD

## Authors

- Ard Snijders
- Christoph Hönes
- Daniel Rodríguez Baleato
- Tamara Czinczoll
