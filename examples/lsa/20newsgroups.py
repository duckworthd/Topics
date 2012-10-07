"""
Apply LSA to the 20newsgroups dataset
"""
import logging
import re

import numpy as np
from sklearn.datasets import fetch_20newsgroups

from topics.lsa import LSA

logging.basicConfig(level=logging.DEBUG)

# retrieve articles
categories = [
  'alt.atheism',
  'talk.religion.misc',
  'comp.graphics',
  'sci.space',
]

data = fetch_20newsgroups(subset='train', categories=categories)

# turn each document into a list of words
documents = [[w.lower() for w in re.findall('[a-zA-Z]{3,}', doc)] for doc in data.data]
documents = documents[0:500]

# run LDA
lsa = LSA(n_topics=len(categories))
results = lsa.infer(documents)

# print out most popular words per topic
reverse_word_index = dict((v,k) for (k,v) in results['word_index'].iteritems())
for t in range(len(categories)):
  topic_word_weights = results['topic_word'][t]
  most_popular = np.argsort(topic_word_weights)[::-1][:20]
  most_popular = [(reverse_word_index[i], topic_word_weights[i]) for i in most_popular]

  print 'Topic %d:' % (t,)
  for (word, weight) in most_popular:
    print '\t%s\t\t%f' % (word, weight)
