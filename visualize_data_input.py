import time
import statistics
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
    parser.add_argument('--num_batches', type=int, default=3000)
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
    start = time.time()
    sampler = TaskSampler(tasks, method='random', custom_task_ratio=None, supp_query_split=True)
    train_iter = sampler.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)
    end = time.time()
    print("Preparation Time for Task Sampler: {} seconds".format(end-start))

    # initiallize statistics containers
    task_ratio = defaultdict(int)
    sup_label_ratio = defaultdict(int)
    quer_label_ratio = defaultdict(int)
    quater_sentence_word = defaultdict(int)
    avg_sup_set_examples = []
    avg_quer_set_examples = []
    mask_average = []
    mask_batch_std = []
    sampling_time = []

    # generate some batches
    for i in range(args.num_batches):
        # prepare support and query set
        start = time.time()
        batch = next(train_iter)
        support = batch[:3]
        query = batch[3:]
        end = time.time()
        sampling_time.append(end-start)

        current_task = sampler.get_name()

        # record task ratio
        task_ratio[current_task] += 1
        num_sup_examples = support[0].shape[0]
        avg_sup_set_examples.append(num_sup_examples)
        mask_average_batch = []
        for s in range(num_sup_examples):
            sup_label_ratio[support[1][s].item()] += 1
            sup_text = support[0][s]
            quater_sentence_word[sup_text[len(sup_text)//4].item()] += 1
            mask_average_batch.append(sum(support[2][s]).item())
        num_quer_examples = query[0].shape[0]
        avg_quer_set_examples.append(num_quer_examples)
        for q in range(num_quer_examples):
            quer_label_ratio[query[1][q].item()] += 1
            quer_text = query[0][q]
            quater_sentence_word[quer_text[len(quer_text)//4].item()] += 1
            mask_average_batch.append(sum(query[2][q]).item())

        mask_average.append(statistics.mean(mask_average_batch))
        if len(mask_average_batch) > 1:
            mask_batch_std.append(statistics.stdev(mask_average_batch))
        else:
            mask_batch_std = mask_average_batch

        # print details (explicit data) for specified batches
        if args.detail[0] <= i <= args.detail[1]:
            if args.task:
                print("Task: {}\n".format(current_task))

            print("Support: \n")
            for s in range(support[0].shape[0]):
                if args.text:
                    print("Text:")
                    print(tokenizer.decode(support[0][s], clean_up_tokenization_spaces=args.pretty))
                    print()
                if args.token_id:
                    print("Token ID: ")
                    print(support[0][s])
                    print()
                if args.mask:
                    print("Mask: ")
                    print(support[2][s])
                    print()
                if args.label:
                    print("Label: {}".format(support[1][s]))
                print()

            print("Query: \n")
            for q in range(query[0].shape[0]):
                if args.text:
                    print("Text:")
                    print(tokenizer.decode(query[0][q], clean_up_tokenization_spaces=args.pretty))
                    print()
                if args.token_id:
                    print("Token ID: ")
                    print(query[0][q])
                    print()
                if args.mask:
                    print("Mask: ")
                    print(query[2][q])
                    print()
                if args.label:
                    print("Label: {}".format(query[1][q]))
                print()

    # print data statistics
    print("Average sampling time per batch: {} seconds".format(statistics.mean(sampling_time)))

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

    print("Query Label Ratio: ")
    quer_label_ratio_results = ""
    for q in quer_label_ratio.keys():
        quer_label_ratio_results += \
            "Label {}: {} ".format(q, quer_label_ratio[q] / (args.num_batches * args.batch_size))
    print(quer_label_ratio_results)

    avg_mask_len = "Average Sentence Length (based on unmasked tokens): Mean: {}, intra-batch std.: {}".format(
        statistics.mean(mask_average), statistics.mean(mask_batch_std))
    if len(mask_average) > 1:
        avg_mask_len += ", inter-batch std.: {}".format(statistics.stdev(mask_average))
    print(avg_mask_len)

    print("Average Examples per batch: Support: {}, Query: {}".format(
        sum(avg_sup_set_examples) / len(avg_sup_set_examples),
        sum(avg_quer_set_examples) / len(avg_quer_set_examples)))

    quater_sentence_word_results = ""
    for w in quater_sentence_word.keys():
        quater_sentence_word_results += \
            "{}: {}\n".format(tokenizer.decode([w]), quater_sentence_word[w] / (args.num_batches * args.batch_size))
    print("Token diversity (at quater position):")
    print(quater_sentence_word_results)