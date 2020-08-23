import time
from transformers import BertTokenizer

from util import get_training_tasks
from tasks import *
from argparse import ArgumentParser
from collections import defaultdict


TASK_NAMES = [
    'Abuse', 'IronySubtaskA', 'IronySubtaskB', 'Offenseval', 'Politeness',
    'SarcasmDetection', 'SemEval18', 'SentimentAnalysis',
]


if __name__ == '__main__':
    # define console arguments
    parser = ArgumentParser()
    parser.add_argument('--training_tasks', nargs='*', choices=TASK_NAMES,
                        default=['SemEval18', 'Offenseval', 'SarcasmDetection'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--detail', nargs=2, type=int, default=[0, 4])
    parser.add_argument('--task', type=bool, default=True)
    parser.add_argument('--text', type=bool, default=True)
    parser.add_argument('--pretty', type=bool, default=True)
    parser.add_argument('--token_id', type=bool, default=False)
    parser.add_argument('--mask', type=bool, default=False)
    parser.add_argument('--label', type=bool, default=True)
    parser.add_argument('--mlp_dims', nargs='*', default=[768])
    args = parser.parse_args()

    # print config
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))

    # prepare tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # setup task sampler and generate iterator
    tasks = get_training_tasks(args)
    sampler = TaskSampler(tasks, method='random', custom_task_ratio=None, supp_query_split=True)
    train_iter = sampler.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)

    # initiallize statistics containers
    task_ratio = defaultdict(int)
    sup_label_ratio = defaultdict(int)
    quer_label_ratio = defaultdict(int)
    quater_sentence_word = defaultdict(int)
    mask_average = 0

    # generate some batches
    for i in range(args.num_batches):
        # prepare support and query set
        batch = next(train_iter)
        support = batch[:3]
        query = batch[3:]

        current_task = sampler.get_name()

        # record task ratio
        task_ratio[current_task] += 1
        for s in range(support[0].shape[0]):
            sup_label_ratio[support[2][s]] += 1
            sup_text = support[0][s]
            quater_sentence_word[sup_text[len(sup_text)//4]] += 1
            mask_average += sum(support[1][s]) / len(sup[1][s])
        for q in range(query[0].shape[0]):
            quer_label_ratio[query[2][q]] += 1
            quer_text = query[0][q]
            quater_sentence_word[quer_text[len(quer_text)//4]] += 1
            mask_average += sum(query[1][q]) / len(query[1][q])

        # print details (explicit data) for specified batches
        if args.detail[0] <= i <= args.detail[1]:
            if args.task:
                print("Task: {}".format(current_task))

            print("Support: ")
            for s in range(support[0].shape[0]):
                if args.text:
                    print("Text:")
                    print(tokenizer.decode(support[0][s], clean_up_tokenization_spaces=args.pretty))
                if args.token_id:
                    print("Token ID: ")
                    print(support[0][s])
                if args.mask:
                    print("Mask: ")
                    print(support[1][s])
                if args.label:
                    print("Label: {}".format(support[2][s]))

            print("Query: ")
            for q in range(query[0].shape[0]):
                if args.text:
                    print("Text:")
                    print(tokenizer.decode(query[0][q], clean_up_tokenization_spaces=args.pretty))
                if args.token_id:
                    print("Token ID: ")
                    print(query[0][q])
                if args.mask:
                    print("Mask: ")
                    print(query[1][q])
                if args.label:
                    print("Label: {}".format(query[2][q]))

        # print data statistics
        task_ratio_results = ""
        for t in args.training_tasks:
            task_ratio_results += "{}: {} ".format(t, task_ratio[t]/args.num_batches)
        print("\nTask distribution:\n{}".format(task_ratio_results))
        print("Support Label Ratio: ")

        sup_label_ratio_results = ""
        for s in sup_label_ratio.keys():
            sup_label_ratio_results += \
                "Label {}: {} ".format(s, sup_label_ratio[s] / (args.num_batches * args.batch_size))
        print(sup_label_ratio_results)

        quer_label_ratio_results = ""
        for q in quer_label_ratio.keys():
            quer_label_ratio_results += \
                "Label {}: {} ".format(q, quer_label_ratio[q] / (args.num_batches * args.batch_size))
        print(quer_label_ratio_results)

        print("Average Sentence Length (based on unmasked tokens): {}".format(
            mask_average / (args.num_batches * args.batch_size)))

        quater_sentence_word_results = ""
        for w in quater_sentence_word.keys():
            quater_sentence_word_results += \
                "{}: {}\n".format(tokenizer.decode([w]), quater_sentence_word[w] / (args.num_batches * args.batch_size))
        print("Token diversity (at quater position):")
        print(quater_sentence_word_results)