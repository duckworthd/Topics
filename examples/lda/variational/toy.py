"""
Apply Spectral LDA inference to a randomly generated toy dataset
"""
import logging
import re

import numpy as np

from topics.lda import *

logging.basicConfig(level=logging.INFO)
np.set_printoptions(precision=4, linewidth=120, suppress=True)

# construct documents
doc_topic_prior = 1.0
topic_word_prior = 0.95
n_topics = 3
n_words = 5
n_docs = 500
doc_length = 5
n_sweeps = 10

sampler = Sampler(
    doc_topic_prior=doc_topic_prior,
    topic_word_prior=topic_word_prior,
    n_topics=n_topics,
    n_words=n_words
)
truth = sampler.sample(n_docs=n_docs, doc_length=doc_length)
documents = truth['documents']

# run LDA
lda = Variational(
    doc_topic_prior=doc_topic_prior,
    topic_word_prior=topic_word_prior,
    n_topics=n_topics
)
result = lda.infer(documents, n_sweeps=n_sweeps)

# print out results
tw_est = result['topic_word']
tw_true= truth['topic_word']

print 'Estimate:'
print tw_est
print 'Truth:'
print tw_true
