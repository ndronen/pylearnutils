import os
import unittest
from collections import Counter

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class RottenTomatoesSentimentDataset(DenseDesignMatrix):

    def __init__(self, which_set, granularity='fine', dataset_path='/home/ndronen/proj/dissertation/projects/deeplsa/data/stanfordSentimentTreebank/', vectorizer=None, one_hot=False, task='classification'):

        if which_set not in ['train', 'dev', 'test']:
            raise ValueError('invalid which_set: ' + str(which_set) +
                    '.  should be train, dev, or test')

        if granularity not in ['fine', 'binary']:
            raise ValueError('invalid granularity: ' + str(granularity) +
                    '.  should be fine or binary')

        if not os.path.isdir(dataset_path):
            raise ValueError('dataset_path ' + str(dataset_path) +
                    ' is not a directory')

        self.__dict__.update(locals())
        del self.self

        if self.vectorizer is None:
            cv = CountVectorizer(input='content')
            idx = self._load_indices('train')
            train_sentences = self._load_sentences(idx)
            cv.fit(train_sentences)
    
        self.idx = self._load_indices(which_set)
        self.sentences = self._load_sentences(self.idx)
        self.phrase_ids = self._load_phrase_ids(self.sentences)
        self.labels = self._load_labels(self.phrase_ids)

        self.X = cv.transform(self.sentences).todense()
        self.y = np.array(self.labels)

        if granularity == 'binary':
            self._binarize_dataset()


        if self.one_hot:
            labels = dict((x, i) for (i, x) in enumerate(np.unique(self.y)))
            one_hot = np.zeros((self.y.shape[0], len(labels)), dtype='float32')
            for i in xrange(self.y.shape[0]):
                label = self.y[i]
                label_position = labels[label]
                one_hot[i, label_position] = 1.
            self.y = one_hot
        else:
            if self.task == 'regression':
                self.y = self.y.reshape((self.y.shape[0], 1))

        self._check_dataset_size()

        super(RottenTomatoesSentimentDataset, self).__init__(X=self.X, y=self.y)

    def _binarize_dataset(self):
        neutral = self.y == 3
        self.X = self.X[~neutral, :]
        self.y = self.y[~neutral]
        self.sentences = self.sentences[~neutral]
        self.phrase_ids = self.phrase_ids[~neutral]
        self.labels = self.labels[~neutral]

    def _check_dataset_size(self):
        ###################################################################
        # From the original paper (Socher et al 2013):
        #
        # Sentences in the treebank were split into a train (8544),
        # dev (1101), and test splits (2210) and tehse splits are made
        # available with the data release.  We also analyze performance
        # on only positive and negative sentences, ignoring the neutral
        # class.  This filters about 20% of the data with the three set
        # having 6920/872/1821 sentences.
        ###################################################################
        if self.granularity == 'fine':
            if self.which_set == 'train':
                assert self.X.shape[0] == 8544
                assert self.y.shape[0] == 8544
            elif self.which_set == 'dev':
                assert self.X.shape[0] == 1101
                assert self.y.shape[0] == 1101
            else:
                assert self.X.shape[0] == 2210
                assert self.y.shape[0] == 2210
        else:
            if self.which_set == 'train':
                assert self.X.shape[0] == 6920
                assert self.y.shape[0] == 6920
            elif self.which_set == 'dev':
                assert self.X.shape[0] == 872
                assert self.y.shape[0] == 872
            else:
                assert self.X.shape[0] == 1821
                assert self.y.shape[0] == 1821

    def _load_indices(self, which_set):
        split_file = self.dataset_path + 'datasetSplit.txt'

        # ID, Split ID
        splits = pd.read_csv(split_file, delimiter=',')

        if which_set == 'train':
            label = 1
        elif which_set == 'dev':
            label = 3
        elif which_set == 'test':
            label = 2
        else:
            raise ValueError('invalid which_set:' + str(which(set)))

        idx = splits[splits.splitset_label == label].sentence_index
        return idx.astype(np.int)

    def _load_sentences(self, idx=None):
        sentences_file = self.dataset_path + 'datasetSentences.txt'
        # ID, Sentence
        sentences = pd.read_csv(sentences_file, delimiter='\t')
        if idx is not None:
            sentences = sentences[sentences.sentence_index.isin(idx)]
        return sentences.sentence

    def _load_phrase_ids(self, sentences):
        dictionary_file = self.dataset_path + 'dictionary.txt'
        # 'Phrase'|Phrase ID
        dictionary = pd.read_csv(dictionary_file, delimiter='|')
        phrases = dict(zip(
            # Phrase
            dictionary.iloc[:, 0].astype(str).tolist(),
            # Phrase IDs
            dictionary.iloc[:, 1].astype(int).tolist()))

        # Return phrase ID of each sentence.
        sentence_ids = np.zeros(shape=(len(sentences)))
        for i, sentence in enumerate(sentences):
            try:
                sentence_ids[i] = phrases[sentence]
            except KeyError:
                print("no phrase ID for sentence: " + sentence)

        return sentence_ids

    def _load_labels(self, phrase_ids=None):
        labels_file = self.dataset_path + 'sentiment_labels.txt'
        # phrase ids|sentiment values
        labels = pd.read_csv(labels_file, delimiter='|')

        # There are duplicate sentences.  For all of these duplicates,
        # a given pair of identical sentences appears to map to a
        # single sentiment value, so just tolerate the duplicates
        # so as to preserve the same n as in other reported results.
        if phrase_ids is not None:
            values = []
            for phrase_id in phrase_ids:
                row_idx = labels['phrase ids'] == phrase_id
                value = labels.loc[row_idx, 'sentiment values'].values[0]
                values.append(value)
        else:
            values = labels['phrase ids'].tolist()
        values = self._map_float_to_target(values)
        return values

    def _map_float_to_target(self, floats):
        ###################################################################
        # From the dataset's README.txt:
        #
        # Note that you can recover the 5 classes by mapping the
        # positivity probability using the following cut-offs:
        #    [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
        # for very negative, negative, neutral, positive, very positive,
        # respectively.
        ###################################################################
        floats = np.array(floats)
        classes = np.ceil(5*floats)
        classes[classes > 5] = 5
        classes[classes < 1] = 1
        return classes

class TestRottenTomatoesSentimentDataset(unittest.TestCase):
    def write_sentences(self, sentences, path):
        with open(path, "w") as f:
            for sentence in sentences:
                try:
                    f.write(sentence + "\n")
                except TypeError:
                    raise TypeError("invalid sentence? " + str(sentence))

    def write_dataset(self):
        rt = RottenTomatoesSentimentDataset(which_set='train', granularity='fine')
        joblib.dump(rt.X, "sentiment-treebank-train-fine-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-train-fine-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-train-fine-sentences.txt")

        rt = RottenTomatoesSentimentDataset('dev', granularity='fine')
        joblib.dump(rt.X, "sentiment-treebank-dev-fine-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-dev-fine-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-dev-fine-sentences.txt")

        rt = RottenTomatoesSentimentDataset('test', granularity='fine')
        joblib.dump(rt.X, "sentiment-treebank-test-fine-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-test-fine-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-test-fine-sentences.txt")

        rt = RottenTomatoesSentimentDataset(which_set='train', granularity='binary')
        joblib.dump(rt.X, "sentiment-treebank-train-binary-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-train-binary-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-train-binary-sentences.txt")

        rt = RottenTomatoesSentimentDataset('dev', granularity='binary')
        joblib.dump(rt.X, "sentiment-treebank-dev-binary-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-dev-binary-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-dev-binary-sentences.txt")

        rt = RottenTomatoesSentimentDataset('test', granularity='binary')
        joblib.dump(rt.X, "sentiment-treebank-test-binary-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-test-binary-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-test-binary-sentences.txt")

    def test_map_float_to_target(self):
        rt = RottenTomatoesSentimentDataset('train', granularity='fine')

        self.assertEqual(rt._map_float_to_target([0.0]), 1)
        self.assertEqual(rt._map_float_to_target([0.2]), 1)

        self.assertEqual(rt._map_float_to_target([0.2001]), 2)
        self.assertEqual(rt._map_float_to_target([0.4]), 2)

        self.assertEqual(rt._map_float_to_target([0.4001]), 3)
        self.assertEqual(rt._map_float_to_target([0.6]), 3)

        self.assertEqual(rt._map_float_to_target([0.6001]), 4)
        self.assertEqual(rt._map_float_to_target([0.8]), 4)

        self.assertEqual(rt._map_float_to_target([0.8001]), 5)
        self.assertEqual(rt._map_float_to_target([1.]), 5)
        self.assertEqual(rt._map_float_to_target([2.]), 5)

if __name__ == '__main__':
    unittest.main()