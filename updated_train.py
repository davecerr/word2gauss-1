import gzip
import os
import sys
import ujson as json
import pandas as pd
import numpy as np
import pickle as pkl

from os.path import join
from time import time
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from model import GaussianEmbedding

################################ DATA HANDLING ################################

def _open_file(filename):
    with gzip.open(filename) as infile:
        for _, line in enumerate(infile):
            yield json.loads(line)

class Corpus(Dataset):
    def __init__(self):
        self.entity_index = defaultdict(lambda: len(self.entity_index))

    @staticmethod
    def read_corpus(corpus_list):
        self = Corpus()
        counter = Counter()
        dataset = []

        # loop through each list (of co-occurring entities) in the corpus
        for entity_list in tqdm(corpus_list):
            if args['verbose']:
                print(f"entity list = {entity_list}")
            # loop through each entity in that list
            for entity in entity_list:
                if args['verbose']:
                    print(f"entity = {entity}")
                self.entity_index[entity]
                counter[self.entity_index[entity]] += 1
                dataset.append(self.entity_index[entity])

        # entity_index is a dictionary d[idx] = entity
        self.index_entity = {v: k for k, v in self.entity_index.items()}
        # counts is a torch tensor. the value at each idx is the count of the entity @ d[idx]
        self.counts = torch.LongTensor(
                          [counter[i] for i in range(len(counter))]
                      )
        # original corpus is re-expressed as torch tensor
        # if i'th entry is idx then entity d[idx] is i'th entity in original corpus
        # note all new lines are ignored - windows can overlap different entity_lists!!!
        self.dataset = torch.LongTensor(dataset)

        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

################################### TRAINING ###################################

class PairwiseWindowIter(object):
    def __init__(self, dataset, window, batch_size):
        self.current_position = 0
        self.dataset = dataset
        self.window = window
        self.batch_size = batch_size

        half_w = window % 2 + 1
        self.order = torch.randperm(len(dataset) - half_w * 2) + half_w
        self.offset = torch.cat((torch.arange(-half_w, 0),
                                 torch.arange(1, half_w + 1)))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_position == -1:
            raise StopIteration

        i = self.current_position
        i_end = i + self.batch_size

        if i_end >= len(self.dataset):
            i_end = -1

        position = self.order[i:i_end].unsqueeze(1).repeat(1, self.window - 1)
        target = self.dataset[position]
        context = self.dataset[position + self.offset]

        self.current_position = i_end

        return target, context

def convert(batch, device):
    target, context = batch
    return target.to(device), context.to(device)


def train(model, dataset, args, device):
    model.train()
    optimizer = optim.Adam(model.parameters())
    start_time = time()

    loss_list = []

    for epoch in range(1, args['epoch'] + 1):
        train_iter = PairwiseWindowIter(dataset, args['window'], args['batch_size'])
        print('------------------------------')
        print('epoch: {}'.format(epoch))

        for i, batch in enumerate(train_iter):
            batch = convert(batch, device)
            loss = model(batch)
            loss_list.append(loss)

            elapsed = time() - start_time
            throuput = args['batch_size'] / elapsed
            prog = args['batch_size'] * (i + 1) / len(dataset) * 100
            print('\r  progress: {:.2f}% entities/s: {:.2f}'.format(
                      min(prog, 100.), throuput
                  ), end='')
            sys.stdout.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.embed.regularize_weights()

        print(f"epoch {epoch} completed in {round((time() - start_time)/3600, 2)} hours")
        start_time = time()

        print()
        print('  loss: {:.4f}'.format(loss.item()))

        if args['debug']:
            torch.save(model.state_dict(),
                       '{}_epoch{}.pt'.format(args['input_dir'].replace('.', '_'),
                                              epoch)
                       )
    return loss_list

#################################### SAVING ####################################

def dump_result(model, index_entity, args):
    model.to('cpu')
    mu_list, sigma_list = model.state_dict().values()

    filename = "model_d={}_epochs={}.txt".format(args['size'],args['epoch'])
    with open(filename, 'w') as f:
        f.write('{} {} {}\n'.format(len(index_entity),
                                    args['size'],
                                    args['covariance']))
        print("\n\n test \n\n")
        print(f)

        for i, (mu, sigma) in enumerate(zip(mu_list, sigma_list)):
            mu_str = ' '.join('{0:.7f}'.format(i) for i in mu.tolist())
            sigma_str = ' '.join('{0:.7f}'.format(i) for i in sigma.tolist())
            f.write('{} {} {}\n'.format(index_entity[i], mu_str, sigma_str))


################################## PREDICTING ##################################

def get_predictions(validation_data, model, is_round=False):
    actual = []
    predicted = []

    # iterate over full dataset
    for _, record in validation_data.iterrows():
        src = standardise_title(record["srcWikiTitle"])
        dst = standardise_title(record["dstWikiTitle"])
        act_sim = float(record["relatedness"])

        # predict similarity
        try:
            pred_sim = float(model.wv.similarity(src, dst)) * 10.
            if is_round:
                pred_sim = np.round(pred_sim)
        except KeyError:
            continue

        # add records
        actual.append(act_sim)
        predicted.append(pred_sim)

    return np.array(actual), np.array(predicted)













##################################### MAIN #####################################

def main(args):

    ############################################################################
    if args['MWE']:
        print("\n\n\n\n >>>>>>>>>> WARNING: YOU ARE USING A DATA SUBSET (MWE=1) <<<<<<<<<<")

    print("\n\n---------- ARGUMENTS ----------")
    print("\nGAUSSIAN:")
    print(f"Dimension = {args['size']}")
    print(f"Covariance = {args['covariance']}")
    print("\nTRAINING:")
    print(f"Epochs = {args['epoch']}")
    print(f"Batch Size = {args['batch_size']}")
    print(f"Window Size = {args['window']}")

    ############################################################################
    # each line in corpus is a list of co-occurring entities
    print("\n\n----------- LOADING CORPUS ----------")
    if os.path.exists("corpus.pkl"):
        start = time()
        print("loading from existing pickle")
        pickle_in = open("corpus.pkl","rb")
        corpus = pkl.load(pickle_in)
        end = time()
        print(f"loaded in {round(end - start,2)} secs")
    else:
        print("loading from gzip files")
        files = []
        for _, _, fs in os.walk(args["input_dir"]):
            files += [f for f in fs if f.endswith(".gz")]

        files = [os.path.join(args["input_dir"], f) for f in files]
        corpus = []
        for i, file in tqdm(enumerate(files)):
            sentences = list(_open_file(file))
            corpus += sentences

        pickle_out = open("corpus.pkl","wb")
        pkl.dump(corpus, pickle_out)
        pickle_out.close()

    print(f"Corpus length = {len(corpus)}")

    ############################################################################
    print("\n\n---------- SETTING DEVICE ----------")
    # cuda options
    use_cuda = args['cuda'] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print("GPU active")
    else:
        if args['cuda'] == 0:
            print("CPU active")
        else:
            print("GPU unavailable. Defaulting to CPU")

    ############################################################################
    print("\n\n---------- CREATING DATASET ----------")
    # args['MWE'] = 1 indicates build a minimum working example from data subset
    if args['MWE'] == 0:
        dataset = Corpus.read_corpus(corpus)
    else:
        dataset = Corpus.read_corpus(corpus[:2])
        print(f"\nCorpus 0 = {corpus[0]}")
        print(f"\nCorpus 1 = {corpus[1]}")

    # print details if required
    if args['verbose']:
        print(f"\nCorpus index_entity = {dataset.index_entity}")
        print(f"\nDataset = {dataset.dataset}")
        print(f"\nCounts = {dataset.counts}")

    # get counts, vocab size and corpus size
    counts = dataset.counts
    print('vocab size: {}'.format(len(counts)))
    print('words in train file: {}'.format(len(dataset)))

    ############################################################################
    print("\n\n---------- CREATING MODEL ----------")
    model = nn.Sequential()
    model.add_module('embed', GaussianEmbedding(args['size'],
                                                counts,
                                                args['window'],
                                                args['batch_size'],
                                                args['covariance'],
                                                device))
    model.to(device)
    print('Model summary:')
    print(model)

    ############################################################################
    print("\n\n---------- TRAINING ----------")
    loss_list = train(model, dataset, args, device)

    ############################################################################
    print("\n\n---------- SAVING ----------")
    dump_result(model, dataset.index_entity, args)
    print(f"Model saved to {args['output_dir']}_{args['size']}_{args['epoch']}.txt")
    pickle_loss = open("loss_list_d={}_epochs={}.pkl".format(args['size'],args['epoch']),"wb")
    pkl.dump(loss_list, pickle_loss)
    pickle_loss.close()





if __name__ == "__main__":
    import argparse

    size = 50
    window = 5
    epoch = 5
    batch_size = 128

    parser = argparse.ArgumentParser(description='Gaussian embedding')

    parser.add_argument('--input_dir', type=str, required=True, help="input directory to the data")

    parser.add_argument('--cuda', type=int, required=True,
                        help='''
                             set it to 1 for running on GPU, 0 for CPU
                             (GPU is 5x-10x slower than CPU)
                             ''')

    parser.add_argument('--MWE', type=int, default=0, help='''train a minimal working example
                                                              using only first two lists of
                                                              co-occurring entities (default: 0)''')

    parser.add_argument('--verbose', type=int, default=0, help='print dataset details (default: 0)')

    parser.add_argument('--epoch', '-e', default=epoch, metavar='N', type=int,
                        help='''
                             number of training epochs
                             (default: {})
                             '''.format(epoch))

    parser.add_argument('--size', '-s', default=size, metavar='N', type=int,
                        help='''
                             the dimension of embedding gaussian
                             (default: {})
                             '''.format(size))

    parser.add_argument('--batch_size', '-b', default=batch_size,
                        metavar='N', type=int,
                        help='''
                             minibatch size for training
                             (default:{})
                             '''.format(batch_size))

    parser.add_argument('--covariance', '-c', default='diagonal',
                        choices=['diagonal', 'spherical'],
                        help='''
                             covariance type ("diagonal", "spherical")
                             (default: diagonal)
                             ''')

    parser.add_argument('--window', '-w', default=window,
                        metavar='N', type=int,
                        help='window size (default: {})'.format(window))

    parser.add_argument('--seed', type=int, default='42', help='random seed')

    parser.add_argument('--debug', '-d', action='store_true')

    args = vars(parser.parse_args())
    main(args)
