import pandas as pd

NUM_EXAMPLES_PER_CLASS = 1089

# IronySubtaskA
df_irony = pd.read_csv('./data/sem_eval_2018/SemEval2018-T3-train-taskA.txt', sep='\t', header=0, names=['Tweet_index', 'Label', 'Tweet_text'])
pd.concat([
    df_irony[df_irony.Label == 0].sample(n=NUM_EXAMPLES_PER_CLASS),
    df_irony[df_irony.Label == 1].sample(n=NUM_EXAMPLES_PER_CLASS)
]).sort_values('Tweet_index').to_csv('./data/sem_eval_2018/IronySubtaskA-T3-train-taskA_mod.txt', header=['Tweet index', 'Label', 'Tweet text'], index=False, sep='\t', encoding='utf-8')

# SarcasmDetection
df_sarcasm = pd.read_json('data/atcs_sarcasm_data/sarcasm_twitter_train.json', lines=True, encoding='utf8')
pd.concat([
    df_sarcasm[df_sarcasm.label == 'SARCASM'].sample(n=NUM_EXAMPLES_PER_CLASS),
    df_sarcasm[df_sarcasm.label == 'NOT_SARCASM'].sample(n=NUM_EXAMPLES_PER_CLASS)
]).sort_index().to_json('data/atcs_sarcasm_data/sarcasm_twitter_train_mod.json', lines=True, orient='records')

# Politeness
df_politeness = pd.read_csv('data/stanford_politeness_2013/wikipedia-politeness-corpus.csv')
pd.concat([
    df_politeness[df_politeness.label == -1].sample(n=NUM_EXAMPLES_PER_CLASS),
    df_politeness[df_politeness.label ==  0].sample(n=NUM_EXAMPLES_PER_CLASS),
    df_politeness[df_politeness.label ==  1].sample(n=NUM_EXAMPLES_PER_CLASS)
]).to_csv('data/stanford_politeness_2013/wikipedia-politeness-corpus_mod.csv')

# SemEval18: Fear, Optimism & Sadness
df_semeval18 = pd.read_csv('./data/semeval18_task1_class/train.txt', sep='\t').set_index('ID')
df_semeval18_fear_holdout = df_semeval18[df_semeval18.fear == 1].sample(n=NUM_EXAMPLES_PER_CLASS)
df_semeval18_optimism_holdout = df_semeval18[df_semeval18.optimism == 1].sample(n=NUM_EXAMPLES_PER_CLASS)
df_semeval18_sadness_holdout = df_semeval18[df_semeval18.sadness == 1].sample(n=NUM_EXAMPLES_PER_CLASS)
df_rest = df_semeval18.drop(pd.concat([df_semeval18_fear_holdout, df_semeval18_optimism_holdout, df_semeval18_sadness_holdout]).index)

# Fear
df_nofear_holdout = df_rest[df_rest.fear == 0].sample(n=NUM_EXAMPLES_PER_CLASS)
df_semeval18_fear = pd.concat([df_semeval18_fear_holdout, df_nofear_holdout])
df_semeval18_fear.to_csv('./data/semeval18_fear_train.txt', sep='\t', columns=['Tweet', 'fear'])
df_rest = df_rest.drop(df_nofear_holdout.index)

# Optimism
df_nooptimism_holdout = df_rest[df_rest.optimism == 0].sample(n=NUM_EXAMPLES_PER_CLASS)
df_semeval18_optimism = pd.concat([df_semeval18_optimism_holdout, df_nooptimism_holdout])
df_semeval18_optimism.to_csv('./data/semeval18_optimism_train.txt', sep='\t', columns=['Tweet', 'optimism'])
df_rest = df_rest.drop(df_nooptimism_holdout.index)

# Sadness
df_nosadness_holdout = df_rest[df_rest.sadness == 0].sample(n=NUM_EXAMPLES_PER_CLASS)
df_semeval18_sadness = pd.concat([df_semeval18_sadness_holdout, df_nosadness_holdout])
df_semeval18_sadness.to_csv('./data/semeval18_sadness_train.txt', sep='\t', columns=['Tweet', 'sadness'])
