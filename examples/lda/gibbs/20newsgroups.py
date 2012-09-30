"""
Apply LDA to a subset of the 20newsgroups dataset
"""
import logging
import re

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from topics.lda import GibbsLDA

logging.basicConfig(level=logging.DEBUG)

# retrieve articles
categories = [
  'talk.religion.misc',
  'comp.graphics',
  'sci.space',
]

data = fetch_20newsgroups(subset='train', categories=categories)

# turn each document into a list of words
documents = [[w.lower() for w in re.findall('[a-zA-Z]{3,}', doc)] for doc in data.data]
documents = [d[:100] for d in documents]

# run LDA
lda = GibbsLDA(n_topics=5, doc_topic_prior=0.01, topic_word_prior=0.01)
histories = lda.infer(documents, n_sweeps=100)

# print out most popular words per topic
final_topic_word = histories[-1]['topic_word_counts']
reverse_word_index = dict( (v,k) for (k,v) in histories[-1]['word_index'].items() )
for t in range(final_topic_word.shape[0]):
  # sort word indices from most popular to least
  most_popular = np.argsort(final_topic_word[t])[::-1][:20]

  # retrieve corresponding words
  most_popular = [(reverse_word_index[i], final_topic_word[t,i]) for i in most_popular]

  # print
  print 'Topic %d' % (t,)
  for (w, p) in most_popular:
    print '%20s:%4.4f' % (w, p/np.sum(final_topic_word[t]))
