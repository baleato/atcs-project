import pandas as pd
from convokit import Corpus, download

# This scripts downloads the following datasets using ConvKit (https://convokit.cornell.edu/)
# - Stanford Politeness Corpus (Wikipedia)
# - Stanford Politeness Corpus (Stack Exchange)
# This code is based on the following notebook:
# - https://github.com/CornellNLP/Cornell-Conversational-Analysis-Toolkit/blob/master/examples/conversations-gone-awry/Conversations_Gone_Awry_Prediction.ipynb
for dataset_name in ['stack-exchange-politeness-corpus', 'wikipedia-politeness-corpus']:
    corpus = Corpus(filename=download(dataset_name))

    kept_conversations = {c.id: c for c in corpus.iter_conversations()}
    kept_utterances = {}
    for convo_id in kept_conversations:
        for utterance in kept_conversations[convo_id].iter_utterances():
            kept_utterances[utterance.id] = utterance

    corpus.conversations = kept_conversations
    corpus.utterances = kept_utterances
    print('{}: {} utterances'.format(dataset_name, len(corpus.utterances)))

    texts = [ corpus.utterances[id].text for id in iter(corpus.utterances) ]
    labels = [ corpus.utterances[id].meta['Binary'] for id in iter(corpus.utterances) ]
    df = pd.DataFrame(data={
        'text': texts,
        'label': labels
    })
    df.to_csv('./{}.csv'.format(dataset_name), index=False)
