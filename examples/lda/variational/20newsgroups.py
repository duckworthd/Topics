"""
Apply Variational LDA to a subset of the 20newsgroups dataset
"""
import logging
import re

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from topics.lda import VariationalLDA

logging.basicConfig(level=logging.INFO)

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
lda = VariationalLDA(n_topics=5, doc_topic_prior=0.01, topic_word_prior=0.01)
result = lda.infer(documents)

# print out most popular words per topic
topic_word = result['topic_word']
reverse_word_index = dict( (v,k) for (k,v) in result['word_index'].items() )
for t in range(topic_word.shape[0]):
  # sort word indices from most popular to least
  most_popular = np.argsort(topic_word[t])[::-1][:20]

  # retrieve corresponding words
  most_popular = [(reverse_word_index[i], topic_word[t,i]) for i in most_popular]

  # print
  print 'Topic %d' % (t,)
  for (w, p) in most_popular:
    print '%20s:%4.4f' % (w, p/np.sum(topic_word[t]))