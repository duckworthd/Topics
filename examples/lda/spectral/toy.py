"""
Apply Spectral LDA inference to a randomly generated toy dataset
"""
import logging
import re

import numpy as np

from topics.lda import *

logging.basicConfig(level=logging.DEBUG)
np.set_printoptions(precision=4, linewidth=120, suppress=True)

# construct documents
sampler = LDASampler(doc_topic_prior=0.1, topic_word_prior=0.5, n_topics=3, n_words=10)
truth = sampler.sample(n_docs=10000, doc_length=3)
documents = truth['words']

# run LDA
lda = SpectralLDA(n_topics=3, doc_topic_prior=0.1)
result = lda.infer(documents)

# print out true and estimated topic-word parameters
tw_est = result['topic_word']
tw_true= truth['topic_word']

print 'Estimate:'
print tw_est
print 'Truth:'
print tw_true
