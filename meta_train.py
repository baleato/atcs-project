import os
import time
import glob
from datetime import timedelta

import torch.nn as nn
import torch
from transformers import BertTokenizer, AdamW

from util import get_args_meta, get_pytorch_device, load_model
from tasks import *
from torch.utils.tensorboard import SummaryWriter
from models import ProtoMAMLLearner
from itertools import chain

from datetime import datetime
import torch.optim as optim


def meta_train(tasks, model, args, device, method='random', custom_task_ratio=None, meta_iters=10000, num_updates=5, meta_batch_size=5):
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

    header = '      Time      Task      Iteration      Loss'
    log_template = '{:>10} {:>25} {:10.0f} {:10.6f}'

    print(header)
    start = time.time()

    # Define optimizers and loss function
    # TODO validate if BertAdam works better and then also use in MTL training
    optimizer = AdamW(params=model.parameters(), lr=args.lr, correct_bias=False)
    # ProtoNets always have CrossEntropy loss due to softmax output
    cross_entropy = nn.CrossEntropyLoss()

    print('Loading Tokenizer..')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    print('done.')

    sampler = TaskSampler(tasks, method=method, custom_task_ratio=custom_task_ratio, supp_query_split=True)
    task_model = type(model)(args, hidden_dims=[500])

    iterations = 0
    # Iterate over the data
    train_iter = sampler.get_iter('train', tokenizer, batch_size=args.batch_size, shuffle=True)
    model.train()
    # outer loop (meta-iterations)
    for i in range(meta_iters):
        grads = []
        iteration_loss = 0
        task_losses = {}
        # inner loop (sample different tasks)
        for task_sample in range(meta_batch_size):
            # clone original model
            task_model.load_state_dict(model.state_dict())
            task_model.to(device)
            task_model.train()

            # new optimizer for every new task model
            task_optimizer_BERT = optim.SGD(params=task_model.proto_net.encoder.parameters(), lr=args.lr)
            task_optimizer = optim.SGD(params=chain(task_model.proto_net.classifier_layer.parameters(),
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
                task_optimizer_BERT.zero_grad()
                task_optimizer.zero_grad()
                predictions = task_model(support[0].to(device), attention_mask=support[2].to(device))
                task_loss = cross_entropy(predictions, support[1].long().squeeze().to(device))
                task_loss.backward()
                task_optimizer.step()
                task_optimizer_BERT.step()

            # record task losses for logging
            task_losses[sampler.get_name()] = task_loss.item()

            # trick to add prototypes back to computation graph
            W = prototypes + (W - prototypes).detach()
            b = (prototypes + (b.unsqueeze(-1) - prototypes).detach()).mean(dim=1)
            task_model.initialize_classifier(W, b, hard_replace=True)

            # calculate gradients for meta update on the query set
            predictions = task_model(query[0].to(device), attention_mask=query[2].to(device))
            query_loss = cross_entropy(predictions, query[1].long().squeeze().to(device))
            query_loss.backward()
            iteration_loss += query_loss.item()

            # register W and b parameters again to avoid error in weight update
            W = nn.Parameter(W)
            b = nn.Parameter(b)
            task_model.initialize_classifier(W, b, hard_replace=True)

            # save gradients of first task model
            if task_sample == 0:
                for param in task_model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grads.append(param.grad)
            # add the gradients of all task samples
            else:
                p = 0
                for param in task_model.parameters():
                    if param.requires_grad and param.grad is not None:
                        grads[p] += param.grad
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
        optimizer.zero_grad()

        iterations += 1
        if iterations % args.log_every == 0:
            iter_loss = iteration_loss / meta_batch_size
            writer.add_scalar('Meta_Average/Loss/train'.format(sampler.get_name()), iter_loss, iterations)
            for t in tasks:
                writer.add_scalar('{}/Loss/train'.format(t.get_name()), task_losses[t.get_name()], iterations)
            print(log_template.format(
                str(timedelta(seconds=int(time.time() - start))),
                sampler.get_name(),
                iterations,
                iter_loss))

        # saving redundant parameters
        # Save model checkpoints.
        if iterations % args.save_every == 0:
            iter_loss = iteration_loss / meta_batch_size
            snapshot_prefix = os.path.join(args.save_path, 'snapshot')
            snapshot_path = (
                    snapshot_prefix +
                    '_iter_{}_loss_{}_model.pt'
            ).format(iterations, iter_loss)
            logging.debug('Saving model...')
            model.save(args.unfreeze_num, snapshot_path)
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

    if args.resume_snapshot:
        print("Loading models from snapshot")
        # TODO find way to pass right number of hidden layers when loading from snapshot
        model = ProtoMAMLLearner(args, hidden_dims=[500])
        model = load_model(args.resume_snapshot, model, args.unfreeze_num, device)
    else:
        model = ProtoMAMLLearner(args, hidden_dims=[500])

    model.to(device)
    print("Tasks")
    tasks = []
    for emotion in SemEval18SingleEmotionTask.EMOTIONS:
        tasks.append(SemEval18SingleEmotionTask(emotion))
    tasks.append(SarcasmDetection())
    tasks.append(OffensevalTask())

    meta_train(tasks, model, args, device)
