import gzip
import os
from os.path import join
from time import time
from tqdm import tqdm
import ujson as json
import pandas as pd
import numpy as np

# from pyspark.sql import SparkSession
from scipy.stats import pearsonr, spearmanr

from lib.wikipedia import standardise_title


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


def main(args):
    # read all the files
    files = []
    for _, _, fs in os.walk(args["input_dir"]):
        files += [f for f in fs if f.endswith(".gz")]

    files = [os.path.join(args["input_dir"], f) for f in files]

    corpus = []
    for i, file in tqdm(enumerate(files)):
        sentences = list(_open_file(file))
        corpus += sentences
        # start = time()
        # # open file
        # sentences = list(_open_file(file))
        #
        # if i == 0:
        #     w2v_model.build_vocab(sentences, update=False)
        #     w2v_model.train(sentences, epochs=args["epochs"], total_examples=w2v_model.corpus_count)
        # else:
        #     w2v_model.build_vocab(sentences, update=True)
        #     w2v_model.train(sentences, epochs=args["epochs"], total_examples=w2v_model.corpus_count)
        #
        # stop = time()

    print(corpus[:5])
    print(len(corpus))

    start = time()
    w2v_model = Word2Vec(sentences=corpus,
                         min_count=args["min_count"],
                         window=args["window"],
                         size=args["size"],
                         sample=args["sample"],
                         alpha=args["alpha"],
                         min_alpha=args["min_alpha"],
                         negative=args["negative"],
                         hs=args["hs"],
                         workers=args["n_jobs"])

    w2v_model.train(corpus, epochs=args["epochs"], total_examples=w2v_model.corpus_count)

    stop = time()

    # load validation dataset
    validation_data = pd.read_csv(args["validation_filepath"], sep=",")

    # predict
    actual, predicted = get_predictions(validation_data, w2v_model, is_round=True)

    # calculate performance
    pear_r, _ = pearsonr(actual, predicted)
    spear_r, _ = spearmanr(actual, predicted)

    print("Pearson R: {},  Spearman R: {}, Vocabulary: {}, on {} word pairs, Duration: {} mins".format(pear_r, spear_r,
                                                                                                       len(
                                                                                                           w2v_model.wv.vocab),
                                                                                                       len(actual),
                                                                                                       round((
                                                                                                                     stop - start) / 60,
                                                                                                             2)))

    # save the model
    w2v_model.save(join(args["output_dir"], "model.pkl"))

    # load validation dataset
    validation_data = pd.read_csv(args["validation_filepath"], sep=",")

    # predict
    actual, predicted = get_predictions(validation_data, w2v_model)

    # calculate performance
    pear_r, pear_p = pearsonr(actual, predicted)
    spear_r, spear_p = spearmanr(actual, predicted)

    results = {}
    results["min_count"] = args["min_count"]
    results["window"] = args["window"]
    results["hs"] = args["hs"]
    results["size"] = args["size"]
    results["sample"] = args["sample"]
    results["alpha"] = args["alpha"]
    results["min_alpha"] = args["min_alpha"]
    results["negative"] = args["negative"]
    results["epochs"] = args["epochs"]

    results["pearson_r"] = pear_r
    results["pearson_p"] = pear_p
    results["spearman_r"] = spear_r
    results["spearman_p"] = spear_p

    results = pd.DataFrame([results])

    results.to_csv(join(args["output_dir"], "results.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--output-dir', type=str, required=True,
                        help="output filepath where the embedding model should be saved")
    parser.add_argument('--input-dir', type=str, required=True, help="input directory to the data")
    parser.add_argument('--n-jobs', type=int, default=10,
                        help="Number of parallel cores to be used")
    parser.add_argument('--min_count', type=int, default=1,
                        help="Number of occurences of a token to be in the batch")
    parser.add_argument('--window', type=int, default=8,
                        help="The window to be considered")
    parser.add_argument('--size', type=int, default=7,
                        help="Number of parallel cores to be used")
    parser.add_argument('--sample', type=float, default=20,
                        help="the negative sample")
    parser.add_argument('--alpha', type=float, default=0.01,
                        help="Starting learning rate")
    parser.add_argument('--min_alpha', type=float, default=7,
                        help="Minimum learning rate that alpha can get")
    parser.add_argument('--negative', type=int, default=20,
                        help="Number of negative samples")
    parser.add_argument('--epochs', type=int, default=7,
                        help="Number of epochs to train")
    parser.add_argument('--hs', type=int, default=0,
                        help="If hierarchical softmax is used for loss")
    parser.add_argument('--validation-filepath', type=str, default=7,
                        help="Filepath to the validation dataset")

    args = vars(parser.parse_args())
    main(args)