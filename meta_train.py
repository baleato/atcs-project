import os
import time
import glob
from datetime import timedelta

import torch.nn as nn
import torch
from transformers import BertTokenizer, AdamW, get_cosine_schedule_with_warmup
import random

from util import get_args_meta, get_pytorch_device, load_model, get_training_tasks, get_validation_task
from tasks import *
from torch.utils.tensorboard import SummaryWriter
from models import ProtoMAMLLearner
from k_shot_testing import k_shot_testing
from itertools import chain

from datetime import datetime
import torch.optim as optim


def meta_train(tasks, model, args, device, method='random', meta_iters=10000, num_updates=5, meta_batch_size=5):
    """
    We'll start with binary classifiers (2-way classification)
    for step in range(num_steps):
        # Training
        for i in num_samples:
            task_batch_train := Sample tasks based on meta_batch_size (training set) (and task frequencies)
            for task in task_batch_train:
                forward
                loss
                backward

        # Meta-training
        if step % meta_every == 0:
            task_batch_test :=  Sample tasks not included in task_batch_train
                                meta_batch_test_size (> meta_batch_size, all?)
            for task in task_batch_test:
                forward
                loss
                backward

    params:
        - tasks
        - method: method of the task sampling sequential, custom probabilities or proportional to sqrt of data size
        - custom_task_ratio: default None only pass if custom task probabilities as sampling method
        - meta_iters: number of meta-training iterations
        - num_updates: number of updates in inner loop on same task_batch
        [NOT needed!?: num_classes: number of classes (N in N-way classification.). Default 2.]
        - meta_batch_size: number of N-way tasks per meta-batch (meta-update)
    """
    # Define logging
    os.makedirs(args.save_path, exist_ok=True)
    writer = SummaryWriter(
        os.path.join(args.save_path, 'runs', '{}'.format(datetime.now()).replace(":", "_")))

    header = '      Time      Task      Iteration      Loss      Accuracy'
    log_template = '{:>10} {:>25} {:10.0f} {:10.6f} {:10.6f}'
    test_template = 'Test mean: {}, Test std: {}'

    print(header)
    start = time.time()

    # Define optimizers, lr schedulers and loss function
    optimizer_bert = AdamW(params=model.proto_net.encoder.bert.parameters(), lr=args.bert_lr)
    optimizer = optim.Adam(params=chain(model.proto_net.encoder.mlp.parameters(),
                                   model.output_layer.parameters()),
                           lr=args.lr)
    scheduler_bert = get_cosine_schedule_with_warmup(optimizer_bert, 200, meta_iters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, meta_iters)
    # ProtoNets always have CrossEntropy loss due to softmax output
    cross_entropy = nn.CrossEntropyLoss()

    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    special_tokens_dict = {'additional_special_tokens': ["[MNT]", "[URL]"]}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    model.proto_net.encoder.bert.resize_token_embeddings(len(tokenizer))
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

    # setup task sampler and task model
    sampler = TaskSampler(tasks, method=method, custom_task_ratio=args.custom_task_ratio, supp_query_split=True)
    task_model = type(model)(args)
    task_model.proto_net.encoder.bert.resize_token_embeddings(len(tokenizer))

    iterations = 0
    # Iterate over the data
    train_iter = sampler.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)
    model.train()

    # setup validation task and episodes for evaluation
    val_task = get_validation_task(args)
    episodes = torch.load(args.episodes)

    # dummy data to overwrite old values of task model output layer
    dummy_w = torch.randn((args.mlp_dims[-1], 2))
    dummy_b = torch.randn(2)

    average_query_loss = 0
    best_query_loss = 1e+9
    best_test_mean = -1
    best_test_last = -1
    convergence_tolerance_cnt = 0
    # outer loop (meta-iterations)
    for i in range(meta_iters):
        grads = []
        task_losses_inner = {}
        task_accuracies_inner = {}
        task_losses_outer = {}
        task_accuracies_outer = {}
        # inner loop (sample different tasks)
        for task_sample in range(meta_batch_size):
            # clone original model
            task_model.proto_net.load_state_dict(model.proto_net.state_dict())
            task_model.initialize_classifier(nn.Parameter(dummy_w), nn.Parameter(dummy_b), hard_replace=True)
            task_model.to(device)
            task_model.train()

            # new optimizer for every new task model
            task_optimizer_bert = optim.SGD(params=task_model.proto_net.encoder.bert.parameters(), lr=args.bert_lr)
            task_optimizer = optim.SGD(params=chain(task_model.proto_net.encoder.mlp.parameters(),
                                                    task_model.output_layer.parameters()),
                                       lr=args.inner_lr)

            # prepare support and query set
            batch = next(train_iter)
            support = batch[:3]
            query = batch[3:]

            # setup output layer (via meta-model's prototype network)
            proto_embeddings = model.proto_net(support[0].to(device), attention_mask=support[2].to(device))
            prototypes = model.proto_net.calculate_centroids((proto_embeddings, support[1]), sampler.get_num_classes())
            W, b = task_model.calculate_output_params(prototypes.detach())
            task_model.initialize_classifier(W, b)

            # train some iterations on support set
            for update in range(num_updates):
                task_optimizer_bert.zero_grad()
                task_optimizer.zero_grad()
                predictions = task_model(support[0].to(device), attention_mask=support[2].to(device))
                task_loss = cross_entropy(predictions, support[1].long().squeeze().to(device))
                task_loss.backward()
                task_optimizer.step()
                task_optimizer_bert.step()

            # record task losses and accuracies for logging
            task_losses_inner[sampler.get_name()] = task_loss.item()
            task_accuracies_inner[sampler.get_name()] = sampler.calculate_accuracy(predictions, support[1].to(device))

            # trick to add prototypes back to computation graph
            W = 2 * prototypes + (W - 2 * prototypes).detach()
            b = -prototypes.norm(dim=1)**2 + (b + prototypes.norm(dim=1)**2).detach()
            task_model.initialize_classifier(W, b, hard_replace=True)

            # calculate gradients for meta update on the query set
            predictions = task_model(query[0].to(device), attention_mask=query[2].to(device))
            query_loss = cross_entropy(predictions, query[1].long().squeeze().to(device))
            query_loss.backward()

            # record task losses and accuracies for logging
            task_losses_outer[sampler.get_name()] = query_loss.item()
            task_accuracies_outer[sampler.get_name()] = sampler.calculate_accuracy(predictions, query[1].to(device))
            average_query_loss += query_loss.item()

            # register W and b parameters again to avoid error in weight update
            W = nn.Parameter(W)
            b = nn.Parameter(b)
            task_model.initialize_classifier(W, b, hard_replace=True)

            # save gradients of first task model
            if task_sample == 0:
                for param in task_model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grads.append(param.grad.clone())
            # add the gradients of all task samples
            else:
                p = 0
                for param in task_model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grads[p] += param.grad.clone()
                        p += 1

        # perform meta update
        # first load/add the calculated gradients in the meta-model
        # (already contains gradients from prototype calculation)
        p = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad += grads[p]
                p += 1
        # update model parameters according to the gradients from inner loop (clear gradients afterwards)
        optimizer.step()
        optimizer_bert.step()
        scheduler.step()
        scheduler_bert.step()
        optimizer.zero_grad()
        optimizer_bert.zero_grad()

        iterations += 1
        if iterations % args.log_every == 0:
            average_query_loss /= (args.log_every*meta_batch_size)
            iter_loss = sum(task_losses_outer.values()) / len(task_losses_outer.values())
            iter_acc = sum(task_accuracies_outer.values()) / len(task_accuracies_outer.values())
            writer.add_scalar('Meta_Average/Loss/outer'.format(sampler.get_name()), iter_loss, iterations)
            writer.add_scalar('Meta_Average/Accuracy/outer'.format(sampler.get_name()), iter_acc, iterations)
            for t in tasks:
                task_name = t.get_name()
                if task_name in task_losses_inner.keys():
                    writer.add_scalar('{}/Loss/inner'.format(task_name), task_losses_inner[task_name], iterations)
                    writer.add_scalar('{}/Accuracy/inner'.format(task_name), task_accuracies_inner[task_name], iterations)
                    writer.add_scalar('{}/Loss/outer'.format(task_name), task_losses_outer[task_name], iterations)
                    writer.add_scalar('{}/Accuracy/outer'.format(task_name), task_accuracies_outer[task_name], iterations)
            print(log_template.format(
                str(timedelta(seconds=int(time.time() - start))),
                sampler.get_name(),
                iterations,
                iter_loss,
                iter_acc))

            # save best snapshot
            if average_query_loss < best_query_loss:
                best_query_loss = average_query_loss
                average_query_loss = 0
                snapshot_prefix = os.path.join(args.save_path, 'best_query')
                snapshot_path = (
                        snapshot_prefix +
                        '_loss_{:.5f}_iter_{}_model.pt'
                ).format(best_query_loss, iterations)
                model.save_model(snapshot_path)
                # Keep only the best snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        # evaluate in k shot fashion
        if iterations % args.eval_every == 0:
            task_model.proto_net.load_state_dict(model.proto_net.state_dict())
            task_model.initialize_classifier(nn.Parameter(dummy_w), nn.Parameter(dummy_b), hard_replace=True)
            test_mean, test_std = k_shot_testing(task_model, episodes, val_task, device, num_updates=args.inner_updates,
                                                 num_test_batches=args.num_test_batches)
            writer.add_scalar('{}/Acc'.format(val_task.get_name()), test_mean, iterations)
            writer.add_scalar('{}/STD'.format(val_task.get_name()), test_std, iterations)
            print(test_template.format(test_mean, test_std), flush=True)
            if test_mean > best_test_mean:
                best_test_mean = test_mean
                snapshot_prefix = os.path.join(args.save_path, 'best_test_{}'.format(val_task.get_name()))
                snapshot_path = (
                        snapshot_prefix +
                        '_acc_{:.5f}_iter_{}_model.pt'
                ).format(best_test_mean, iterations)
                model.save_model(snapshot_path)
                # Keep only the best snapshot
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)
            
            if test_mean > best_test_last:
                best_test_last = best_test_mean
                convergence_tolerance_cnt = 0
            else:
                convergence_tolerance_cnt += 1

            if convergence_tolerance_cnt == args.convergence_tolerance:
                break


        # saving redundant parameters
        # Save model checkpoints.
        if iterations % args.save_every == 0:
            iter_loss = sum(task_losses_outer.values()) / len(task_losses_outer.values())
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = (
                    snapshot_prefix +
                    '_iter_{}_loss_{}_model.pt'
            ).format(iterations, iter_loss)
            logging.debug('Saving model...')
            model.save_model(snapshot_path)
            # Keep only the last snapshot
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

    writer.close()


if __name__ == '__main__':
    args = get_args_meta()
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))
    device = get_pytorch_device(args)
    # set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.resume_snapshot:
        print("Loading models from snapshot")
        # TODO find way to pass right number of hidden layers when loading from snapshot
        model = ProtoMAMLLearner(args)
        model = load_model(args.resume_snapshot, model, args.unfreeze_num, device)
    else:
        model = ProtoMAMLLearner(args)

    model.to(device)
    tasks = get_training_tasks(args)
    meta_train(tasks, model, args, device, meta_iters=args.num_iterations,
               num_updates=args.inner_updates, meta_batch_size=args.meta_batch_size)
