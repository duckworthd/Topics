"""
Implementations of inference methods for Latent Dirichlet Allocation.
"""
import copy
import itertools
import logging
from numbers import Number

import numpy as np


class GibbsLDA(object):
  """Latent Dirichlet Allocation with Gibbs Sampling for Inference

  Implements "Collapsed Gibbs Sampling" wherein doc-topic and topic-word
  distributions are integrated away implicitly. The only variables iterated
  over during MCMC are the individual topics of each word.

  Parameters
  ----------
  n_topics: int
    number of topics
  doc_topic_prior: float or [n_topics] array
    pseudocounts for how often a topic appears. if a float, equivalent to an
    n_topics array with the same value at every position.
  topic_word_prior: float or {str:float} dict
    pseudocounts for how often a word appears. if a float, equivalent to an
    {str:float} dict where all strings have the same float value. If a
    {str:float}, maps words to their individual pseudocounts for any topic.
    None is treated as a special key for words not in the dictionary.
  """
  def __init__(self, doc_topic_prior=0.01, topic_word_prior=0.01, n_topics=None):
    self.n_topics = n_topics
    self.doc_topic_prior = doc_topic_prior
    self.topic_word_prior = topic_word_prior
    self.logger = logging.getLogger(str(self.__class__))

  def infer(self, documents, n_sweeps=100, word_topics=None):
    r = np.random.RandomState(0)

    # initialize counts for each doc-topic and topic-word pair
    (doc_topic_counts, topic_word_counts, word_index) = self._initialize(documents)
    topic_counts = np.sum(topic_word_counts, axis=1)
    n_topics = topic_word_counts.shape[0]
    n_docs = doc_topic_counts.shape[0]
    n_words = len(word_index)

    # transform documents into lists of word indices
    documents2 = []
    for (d, doc) in enumerate(documents):
      documents2.append([word_index[w] for w in doc])
    documents = documents2

    # initialize topics for all words uniformly at random
    if word_topics is None:
      word_topics = []  # word_topics[d][i] = topic of word i in document d
      for (d, doc) in enumerate(documents):
        doc_topics = []
        for (i, word) in enumerate(doc):
          # select topic for word
          t = np.nonzero(r.multinomial(1, np.ones(n_topics)/n_topics))[0][0]
          doc_topics.append(t)
        word_topics.append(doc_topics)

    # initialize doc-topic and topic-word counts
    for (d, doc) in enumerate(documents):
      for (i, word) in enumerate(doc):
        # get topic for this word
        t = word_topics[d][i]

        # increment counts
        doc_topic_counts[d,t] += 1
        topic_word_counts[t,word] += 1
        topic_counts[t] += 1

    # resample word topics
    history = []  # state of chain after each sweep
    for sweep in range(n_sweeps):
      #self.logger.debug('starting sweep #%d' % (sweep,))
      print('starting sweep #%d' % (sweep,))
      for (d, doc) in enumerate(documents):

        if d % 100 == 0:
          print 'starting document #%d' % (d,)

        for (i, word) in enumerate(doc):
          # get topic for this word in this document
          t = word_topics[d][i]

          # remove it from counts
          doc_topic_counts[d,t] -= 1
          topic_word_counts[t,word] -= 1
          topic_counts[t] -= 1

          # calculate P(t | everything else)
          prob = [
              doc_topic_counts[d,t] * topic_word_counts[t,word] / topic_counts[t]
              for t in range(n_topics)
          ]
          prob = np.array(prob) / np.sum(prob)

          # select topic
          t = np.nonzero(r.multinomial(1, prob))[0][0]

          # increment counts
          doc_topic_counts[d,t] += 1
          topic_word_counts[t,word] += 1
          topic_counts[t] += 1
          word_topics[d][i] = t

      history.append({
        'topic_word_counts': np.copy(topic_word_counts),
        'doc_topic_counts': np.copy(doc_topic_counts),
        'word_topics': copy.deepcopy(word_topics),
        'word_index': word_index
      })

    # return final result
    return history

  def _initialize(self, documents):
    """Parse hyperparameters and initialize counts"""
    words = set(itertools.chain(*documents))
    # get actual number of topics
    n_topics = (
        self.n_topics
        if self.n_topics is not None
        else len(doc_topic_prior)
    )
    n_docs = len(documents)
    n_words = len(words)

    # initialize pseudcount matrices
    doc_topic_prior = self.doc_topic_prior
    if isinstance(doc_topic_prior, Number):
      doc_topic_prior = np.ones( (n_docs, n_topics) ) * doc_topic_prior / n_topics
    else:
      doc_topic_prior = np.atleast_1d(doc_topic_prior)
      doc_topic_prior = np.tile(doc_topic_prior, (n_docs, 1))

    topic_word_prior = self.topic_word_prior
    if isinstance(topic_word_prior, Number):
      topic_word_prior = np.ones( (n_topics, n_words) ) * topic_word_prior / n_words
    else:
      # give each unknown word a pseudocount
      for w in list(topic_word_prior.keys()):
        if w not in words:
          topic_word_prior[w] = topic_word_prior[None]

      # turn that into a vector of pseudocounts with integer indices
      topic_word_prior = [v for (k, v) in sorted(list(topic_word_prior.items()))]
      topic_word_prior = np.atleast_1d(topic_word_prior)

      # copy vector, once for each topic
      topic_word_prior = np.tile(topic_word_prior, (n_topics, 1))

    word_index = dict( (word, index) for (index, word) in enumerate(sorted(words)) )

    #self.logger.debug('n_topics = %d, n_words = %d, n_docs = %d' % (n_topics, n_words, n_docs))
    print('n_topics = %d, n_words = %d, n_docs = %d' % (n_topics, n_words, n_docs))

    return (doc_topic_prior, topic_word_prior, word_index)


class VariationalLDA(object):
  def __init__(self):
    pass

  def infer(self, words):
    pass
