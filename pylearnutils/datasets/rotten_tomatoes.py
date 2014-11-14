import sys
import os
import unittest
from collections import Counter

import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class RottenTomatoesLoader(object):
    def __init__(self, which_set, granularity='fine', dataset_path='/home/ndronen/proj/dissertation/projects/deeplsa/data/stanfordSentimentTreebank/'):

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

        self.idx = self._load_indices(which_set)
        self.sentences = self._load_sentences(self.idx)
        self.phrase_ids = self._load_phrase_ids(self.sentences)
        self.labels = self._load_labels(self.phrase_ids)

        if granularity == 'binary':
            self.binarize_dataset()

        self._check_dataset_size()

    def binarize_dataset(self):
        neutral = self.labels == 3
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
                assert len(self.sentences) == 8544
                assert len(self.labels) == 8544
            elif self.which_set == 'dev':
                assert len(self.sentences) == 1101
                assert len(self.labels) == 1101
            else:
                assert len(self.sentences) == 2210
                assert len(self.labels) == 2210
        else:
            if self.which_set == 'train':
                assert len(self.sentences) == 6920
                assert len(self.labels) == 6920
            elif self.which_set == 'dev':
                assert len(self.sentences) == 872
                assert len(self.labels) == 872
            else:
                assert len(self.sentences) == 1821
                assert len(self.labels) == 1821

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
        return sentences.sentence.as_matrix()

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
        sentence_idxs = np.zeros(shape=(len(sentences)))
        for i, sentence in enumerate(sentences):
            try:
                sentence_idxs[i] = phrases[sentence]
            except KeyError:
                print("no phrase ID for sentence: " + sentence)

        return sentence_idxs

    def _load_labels(self, phrase_ids=None):
        labels_file = self.dataset_path + 'sentiment_labels.txt'
        # phrase ids|sentiment values
        label_data = pd.read_csv(labels_file, delimiter='|')

        # There are duplicate sentences.  For all of these duplicates,
        # a given pair of identical sentences appears to map to a
        # single sentiment value, so just tolerate the duplicates
        # so as to preserve the same n as in other reported results.
        if phrase_ids is not None:
            labels = []
            for phrase_id in phrase_ids:
                row_idx = label_data['phrase ids'] == phrase_id
                label = label_data.loc[row_idx, 'sentiment values'].values[0]
                labels.append(label)
        else:
            labels = label_data['phrase ids'].tolist()

        return self.map_float_to_target(labels)

    def map_float_to_target(self, floats):
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
        return classes.astype(np.int)

class RottenTomatoesBagOfWordsDataset(DenseDesignMatrix):
    def __init__(self, which_set, granularity='fine', dataset_path='/home/ndronen/proj/dissertation/projects/deeplsa/data/stanfordSentimentTreebank/', vectorizer=None, one_hot=False, task='classification'):

        self.__dict__.update(locals())
        del self.self

        ###################################################################
        # This is specific to the bag-of-words model.  It appears that in
        # the literature they use the vocabulary of the entire training
        # set even when they train only on the reviews with labels that 
        # are not neutral (i.e. very positive, positive, negative,
        # very negative).
        ###################################################################
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(input='content')
            train_loader = RottenTomatoesLoader(
                'train', 'fine', dataset_path)
            self.vectorizer.fit(train_loader.sentences)
    
        self.loader = RottenTomatoesLoader(
                which_set, granularity, dataset_path)

        self.y = np.array(self.loader.labels)
        self.X = self.vectorizer.transform(self.loader.sentences).todense()

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

        super(RottenTomatoesBagOfWordsDataset, self).__init__(X=self.X, y=self.y)

    def __getattr__(self, name):
        return getattr(self.loader, name)

class Subsequence(object):
    """
    """
    def __init__(self, sequence, k):
        self.subsequences = zip(*(sequence[i:] for i in range(k)))

    def __iter__(self):
        return iter(self.subsequences)

class RottenTomatoesGroundHogIterator(object):
    """
    """
    def __init__(self, which_set, granularity='fine', dataset_path='/home/ndronen/proj/dissertation/projects/deeplsa/data/stanfordSentimentTreebank/', vectorizer=None, task='classification', seq_len=2):

        self.__dict__.update(locals())
        del self.self

        ###################################################################
        # This is specific to the sequence model.  We want to assign an
        # integer to each word in the vocabulary.  To do this, fit a
        # count vectorizer to the training set.  If a word vectorizes
        # to the 0 vector, the integer assigned to the word is the size
        # of the vocabulary.
        ###################################################################
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(input='content')
            train_loader = RottenTomatoesLoader(
                'train', 'fine', dataset_path)
            self.vectorizer.fit(train_loader.sentences)
    
        self.loader = RottenTomatoesLoader(
                which_set, granularity, dataset_path)

        self.sentence_idx = 0
        self.sentence_iter = None

    def __len__(self):
        return len(self.sentences) // self.seq_len

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        return self

    def next(self):
        """
        """
        ###################################################################
        # At this point we have a set of labels and corresponding sentences.
        # We need to convert each sentence into a sequence of subsequences
        # of some user-specified length `seq_len`.  If `seq_len` is 3, then,
        #
        #     "This is a sentence in a file"
        #
        # becomes
        #
        #   [THIS_ID, IS_ID, A_ID],
        #   [IS_ID, A_ID, SENTENCE_ID],
        #   [A_ID, SENTENCE_ID, IN_ID],
        #   ...,
        #   [IN_ID, A_ID, FILE_ID]]
        #
        # Each of these subsequences is a training example.
        # 
        # For the first pass through this implementation, I will just map
        # each sentence to a single uninterrupted (and hence variable-length)
        # sequence of term IDs.
        ###################################################################

        while True:
            # We've reached the end of the corpus.  For now just stop.
            if self.sentence_idx == len(self.sentences):
                self.sentence_iter = None
                self.sentence_idx = 0
                raise StopIteration()

            if self.sentence_iter is None:
                sentence = self.sentences[self.sentence_idx]
                # TODO: figure out how to prevent the analyzer from removing articles
                # (e.g. "a").
                tokens = self.vectorizer.build_analyzer()(sentence)
                token_ids = []
                for token in tokens:
                    try:
                        token_ids.append(self.vectorizer.vocabulary_[token])
                    except KeyError:
                        token_ids.append(len(self.vectorizer.vocabulary_))
                self.sentence_iter = iter(Subsequence(token_ids, self.seq_len))

            target = self.labels[self.sentence_idx]

            try:
                seq = self.sentence_iter.next()
                return seq, target, -1
            except StopIteration:
                # We've reached the end of the sentence or the sentence
                # is shorter than seq_len.
                self.sentence_iter = None
                self.sentence_idx += 1
                continue


class TestRottenTomatoesBagOfWordsDataset(unittest.TestCase):
    def write_sentences(self, sentences, path):
        with open(path, "w") as f:
            for sentence in sentences:
                try:
                    f.write(sentence + "\n")
                except TypeError:
                    raise TypeError("invalid sentence? " + str(sentence))

    @unittest.skip("only use this to write out the dataset to files")
    def test_write_dataset(self):
        rt = RottenTomatoesBagOfWordsDataset(
                which_set='train', granularity='fine')
        joblib.dump(rt.X, "sentiment-treebank-train-fine-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-train-fine-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-train-fine-sentences.txt")

        rt = RottenTomatoesBagOfWordsDataset('dev', granularity='fine')
        joblib.dump(rt.X, "sentiment-treebank-dev-fine-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-dev-fine-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-dev-fine-sentences.txt")

        rt = RottenTomatoesBagOfWordsDataset('test', granularity='fine')
        joblib.dump(rt.X, "sentiment-treebank-test-fine-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-test-fine-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-test-fine-sentences.txt")

        rt = RottenTomatoesBagOfWordsDataset(
                which_set='train', granularity='binary')
        joblib.dump(rt.X, "sentiment-treebank-train-binary-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-train-binary-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-train-binary-sentences.txt")

        rt = RottenTomatoesBagOfWordsDataset('dev', granularity='binary')
        joblib.dump(rt.X, "sentiment-treebank-dev-binary-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-dev-binary-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-dev-binary-sentences.txt")

        rt = RottenTomatoesBagOfWordsDataset('test', granularity='binary')
        joblib.dump(rt.X, "sentiment-treebank-test-binary-X.joblib")
        joblib.dump(rt.y, "sentiment-treebank-test-binary-y.joblib")
        self.write_sentences(rt.sentences,
            "sentiment-treebank-test-binary-sentences.txt")

class TestRottenTomatoesLoader(unittest.TestCase):

    def testmap_float_to_target(self):
        loader = RottenTomatoesLoader('train', granularity='fine')

        self.assertEqual(loader.map_float_to_target([0.0]), 1)
        self.assertEqual(loader.map_float_to_target([0.2]), 1)

        self.assertEqual(loader.map_float_to_target([0.2001]), 2)
        self.assertEqual(loader.map_float_to_target([0.4]), 2)

        self.assertEqual(loader.map_float_to_target([0.4001]), 3)
        self.assertEqual(loader.map_float_to_target([0.6]), 3)

        self.assertEqual(loader.map_float_to_target([0.6001]), 4)
        self.assertEqual(loader.map_float_to_target([0.8]), 4)

        self.assertEqual(loader.map_float_to_target([0.8001]), 5)
        self.assertEqual(loader.map_float_to_target([1.]), 5)
        self.assertEqual(loader.map_float_to_target([2.]), 5)

class TestRottenTomatoesGroundHogIterator(unittest.TestCase):

    def test_init(self):
        rt = RottenTomatoesGroundHogIterator(
                which_set='train', granularity='fine')
        iterator = iter(rt)
        while True:
            try:
                print(iterator.next())
            except StopIteration:
                print("Done")
                break

if __name__ == '__main__':
    unittest.main()
