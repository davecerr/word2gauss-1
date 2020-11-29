import gzip
import os
from os.path import join
from time import time
from tqdm import tqdm
from collections import Counter, defaultdict
import ujson as json
import pandas as pd
import numpy as np
import pickle as pkl

# from pyspark.sql import SparkSession
from scipy.stats import pearsonr, spearmanr

from model import GaussianEmbedding

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

def _open_file(filename):
    with gzip.open(filename) as infile:
        for _, line in enumerate(infile):
            yield json.loads(line)


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






class Corpus(Dataset):
    def __init__(self):
        self.entity_index = defaultdict(lambda: len(self.entity_index))

    @staticmethod
    def read_corpus(corpus_list):
        self = Corpus()
        counter = Counter()
        dataset = []

        for entity_list in tqdm(corpus_list):
            for entity in entity_list:
                self.entity_index[entity]
                counter[self.entity_index[entity]] += 1
                dataset.append(self.entity_index[entity])

        self.index_entity = {v: k for k, v in self.entity_index.items()}
        self.counts = torch.LongTensor(
                          [counter[i] for i in range(len(counter))]
                      )
        self.dataset = torch.LongTensor(dataset)

        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def main(args):

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
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print("GPU active")
    else:
        print("CPU active")

    ############################################################################
    print("\n\n---------- CREATING DATASET ----------")
    dataset = Corpus.read_corpus(corpus)
    counts = dataset.counts

    print('vocab size: {}'.format(len(counts)))
    print('words in train file: {}'.format(len(dataset)))


    ############################################################################
    print("\n\n---------- CREATING MODEL ----------")
    model = nn.Sequential()
    model.add_module('embed', GaussianEmbedding(args.size,
                                                counts,
                                                args.window,
                                                args.batch_size,
                                                args.covariance,
                                                device))
    model.to(device)
    print('Model summary:')
    print(model)
    print()

    ############################################################################
    print("\n\n---------- TRAINING ----------")
    train(model, dataset, args, device)
    dump_result(model, dataset.index_word, args)







if __name__ == "__main__":
    import argparse

    size = 50
    window = 5
    epoch = 5
    batch_size = 128

    parser = argparse.ArgumentParser(description='Gaussian embedding')

    parser.add_argument('--input_dir', type=str, required=True, help="input directory to the data")

    parser.add_argument('--output_dir', type=str, required=True,
                        help='path to save the result model')

    parser.add_argument('--cuda', type=int, required=True,
                        help='''
                             set it to 1 for running on GPU, 0 for CPU
                             (GPU is 5x-10x slower than CPU)
                             ''')

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
    print(args)
    # print(args.cuda)
    print(args["cuda"])
    main(args)
