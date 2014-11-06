#!/usr/bin/env python

import os.path
import argparse
import joblib
import numpy as np
from gensim.models import Word2Vec

def make_w2v_weight_matrix(vectors_path, vectorizer_path, low_init=-0.5, high_init=0.5): 
    model = Word2Vec.load_word2vec_format(vectors_path, binary=True)
    vectorizer = joblib.load(vectorizer_path)

    nvis = len(vectorizer.vocabulary_)
    nhid = model.layer1_size

    W = np.zeros(shape=(nhid, nvis))
    for i, (word,index) in enumerate(vectorizer.vocabulary_.items()):
        try:
            W[:, i] = model[word]
        except KeyError:
            try:
                # This might be a proper name, so upcase the first character.
                W[:, i] = model[word.title()]
            except KeyError:
                print("Didn't find vector for " + word)
                W[: , i] = np.random.uniform(low=low_init, high=high_init, size=(nhid,))

    return W

def main():
    parser = argparse.ArgumentParser(
        description="create a numpy matrix from word2vec vectors")
    parser.add_argument("vectors_path", metavar="VECTORS_FILE", type=str,
        help="the path to a binary file containing word2vec vectors")
    parser.add_argument("vectorizer_path", metavar="VECTORIZER_FILE", type=str,
        help="the path to a sklearn CountVectorizer (for the vocabulary)")
    parser.add_argument("npy_path", metavar="NUMPY_OUTPUT_FILE", type=str,
        help="the path to the output file in which to save the matrix")
    parser.add_argument("--low-init", type=float, default=-0.5, 
        help="the low end of the range for initializing one of the weight matrix's columns when the word2vec file doesn't have a vector for a given word")
    parser.add_argument("--high-init", type=float, default=0.5, 
        help="the high end of the range for initializing one of the weight matrix's columns when the word2vec file doesn't have a vector for a given word")

    args = parser.parse_args()

    if ".npy" in args.npy_path:
        args.npy_path = os.path.splitext(args.npy_path)[0]

    W = make_w2v_weight_matrix(
            vectors_path=args.vectors_path,
            vectorizer_path=args.vectorizer_path,
            low_init=args.low_init,
            high_init=args.high_init)

    np.save(args.npy_path, W)

if __name__ == "__main__":
    main()
