"""
Implementation of Latent Semantic Analysis
"""
import copy
import itertools
import logging
from numbers import Number

import numpy as np
from scipy.sparse import linalg

from util import doc_word_matrix, reindex

class LSA(object):
  def __init__(self, n_topics):
    self.n_topics = n_topics

  def infer(self, documents):
    # turn words into integer indices
    (documents, word_index) = reindex(documents)
    n_docs = len(documents)
    n_words = len(word_index)
    n_topics = self.n_topics

    # create document-word count matrix
    counts = doc_word_matrix(documents, n_words)

    # run SVD
    (U, S, Vt) = linalg.svds(counts, k=n_topics)

    return {
        'topic_word': Vt,
        'doc_topic': U,
        'topic_weights': S,
        'word_index': word_index
    }
