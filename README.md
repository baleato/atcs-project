## Introduction

This repository performs meta-learning across multiple NLP tasks focusing on pragmatics and social meaning (emotion, sarcasm, politeness, abusive language detection). Performance can be evaluated in a k-shot fashion.

## Dependencies

Python 3, PyTorch, Transformers, Tokenizers
Consult the requirements.txt for a full list of dependency packages.
The dependencies can be installed by running:
```sh
$ pip install -r requirements.txt
```

## Tasks & datasets  
### Training Tasks
- Emotion classification - [SemEval-2018 Task 1: Affect in Tweets (AIA-2018)](https://competitions.codalab.org/competitions/17751)
- Sarcasm detection - [The Shared Task on Sarcasm Detection (ACL 2020)](https://competitions.codalab.org/competitions/22247)
- Offensive Language Identification (OLID) - [SemEval 2019  Task 6: OffensEval](https://sites.google.com/site/offensevalsharedtask/olid)   
### Validation Task
- Sentiment classification - [SemEval-2015 Task 10: Sentiment Analysis in Twitter](http://alt.qcri.org/semeval2015/task10/)  
### Testing Tasks
- Irony Detection (Task A & B) - [SemEval-2018 Task 3: Irony detection in English tweets](https://competitions.codalab.org/competitions/17468)  
- Abusive language detection - [data-twitter-wh; Waseem and Hovy 2016](https://github.com/zeerakw/hatespeech)
- Politeness - [Stanford Politeness Corpus](http://www.cs.cornell.edu/~cristian/Politeness.html).
  :exclamation:	This dataset is based on web-forum interactions and not Twitter data.

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
├── sem_eval_2015
│   └── tweets_output.txt
├── sem_eval_2018
│   ├── SemEval2018-T3-train-taskA.txt
│   └── SemEval2018-T3-train-taskB.txt
├── semeval18_task1_class
│   ├── dev.txt
│   ├── test.txt
│   └── train.txt
├── stanford_politeness_2013
│   ├── README.txt
│   ├── stack-exchange-politeness-corpus.csv
│   └── wikipedia-politeness-corpus.csv
├── tweet_wassem
│   └── twitter_data_waseem_hovy.csv
└── twitter
    └── sarcasm_detection_shared_task_twitter_training.jsonl
```

## Models

All models share the encoder (composed by BERT and a multi-layer perceptron on top). Training is performed on the MLP and the last 2 layers of BERT.

- MultiTaskLearner: adds task-dependent single linear layers on top of the _encoder_.
- PrototypeLearner: uses the _encoder_ to learn a set of embeddings to classify an example based on class centroids previously calculated.
- ProtoMAMLLearner: extends the _PrototypeLearner_ with an task-independent linear layer.

The pre-trained models can be found in the following google drive folder:
https://drive.google.com/drive/u/0/folders/1bfTU8SV0xNzlpr_h4CjaUWqNMmZ28Dl8

## Running examples

To train a model just execute the corresponding line in a terminal:  

Multitask model
```sh
$ python train.py --save_path "path_to_checkpoint_dir"
```
ProtoNet
```sh
$ python train_prototype.py --save_path "path_to_checkpoint_dir"
```
ProtoMAML
```sh
$ python meta_train.py --save_path "path_to_checkpoint_dir"
```
There are various parameters that can be passed to the model via a flag (e.g --lr 1e-4). For a detailed overview od all flags please refer to the help function of the argument parser. 
```sh
$ python train.py -h
```
The model checkpoints are then savedd to the specified ```--save_path```. There will be three checkpoints based on the best validation task accuracy, the best training task performance and a checkpoint after the last training iteration.

To perform k-shot testing for a model run the following command:
```sh
$ python k_shot_testing.py --model_path "model_checkpoint_path.pt" --model "model_name" --task "testing_task_name" --k 4 \
                           --episodes "path_to_my_episodes.pkl"
```
The testing will print the mean test accuracy and the standard deviation over episodes once the evaluation is completed.  
If you do not specify saved episodes the testing will just randomly sample episodes from the test task training set.  
Please check the other possible arguments in the argparser help to get an overview over the the functionality.  
:exclamation: Note: if you change the MLP architecture then you also have to specify ```--mlp_dims``` with the right amount of layers and hidden neurons.

## Authors

- Ard Snijders
- Christoph Hönes
- Daniel Rodríguez Baleato
- Tamara Czinczoll
