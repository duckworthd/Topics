"""
Implementations of inference methods for Latent Dirichlet Allocation.
"""
import copy
import itertools
import logging
from numbers import Number

import numpy as np
from scipy import linalg
from scipy.special import psi as digamma, gammaln
from scipy.misc import logsumexp

from util import reindex, flatten_counts, default_on_nan


class GibbsLDA(object):
  """Latent Dirichlet Allocation with Gibbs Sampling for Inference

  Implements "Collapsed Gibbs Sampling" (Griffiths & Steyvers, 2004) wherein
  doc-topic and topic-word distributions are integrated away implicitly. The
  only variables iterated over during MCMC are the individual topics of each
  word.

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
      self.logger.debug('starting sweep #%d' % (sweep,))
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

    self.logger.debug('n_topics = %d, n_words = %d, n_docs = %d' % (n_topics, n_words, n_docs))

    return (doc_topic_prior, topic_word_prior, word_index)


class VariationalLDA(object):
  """Latent Dirichlet Allocation with Variational Inference

  Implements batch Variational Inference algorithm described in the original
  Latent Dirichlet Allocation (Blei, 2003) where posterior is approximated by
  the product of q(topic | word index, doc) [categorical], q(topic | doc)
  [dirichlet], and q(word | topic) [dirichlet].
  """
  def __init__(self, doc_topic_prior=0.01, topic_word_prior=0.01, n_topics=None):
    self.doc_topic_prior = doc_topic_prior
    self.topic_word_prior = topic_word_prior
    self.n_topics = n_topics
    self.logger = logging.getLogger(str(self.__class__))

  def infer(self, documents):
    # initialize priors, documents
    (doc_word_counts, doc_topic_prior, topic_word_prior, word_index) = self._initialize(documents)
    n_docs = doc_word_counts.shape[0]
    n_topics = doc_topic_prior.shape[0]
    n_words = doc_word_counts.shape[1]

    # initialize parameters
    r = np.random.RandomState(0)
    doc_topic_params = r.rand(n_docs, n_topics)
    topic_word_params = r.rand(n_topics, n_words)
    doc_word_topic_params = r.rand(n_docs, n_words, n_topics)
    while True:
      old_topic_word_params = np.copy(topic_word_params)
      self.logger.info("Starting new pass")

      for d in range(n_docs):
        if d % 100 == 0:
          self.logger.info("Processing document %d/%d" % (d, n_docs))
          try:
            ll = self.elbo(doc_word_counts, doc_word_topic_params,
                doc_topic_params, topic_word_params, doc_topic_prior,
                topic_word_prior)
            self.logger.info("Current Log Likelihood: %f" % (ll,))
          except AssertionError as e:
            logging.warn('Failed to calculate ELBO: %s' % (str(e),))

        # reset doc-topic parameters for this documents
        doc_topic_params[d] = np.ones(n_topics)
        while True:
          old_doc_topic_params = np.copy(doc_topic_params[d])

          # update q(topic | document, word)
          for w in np.nonzero(doc_word_counts[d])[0]:
            log_dwt = np.zeros(n_topics)  # do the following doc_word_topic_param update in logspace
            for t in range(n_topics):
              log_dwt[t] = (
                digamma(doc_topic_params[d,t]) - digamma(np.sum(doc_topic_params[d]))
                + digamma(topic_word_params[t,w]) - digamma(np.sum(topic_word_params[t]))
              )
            doc_word_topic_params[d,w] = np.exp(log_dwt - logsumexp(log_dwt))

          # update q(topic | doc)
          doc_topic_params[d] = doc_topic_prior
          for t in range(n_topics):
            doc_topic_params[d,t] += np.dot(doc_word_topic_params[d,:,t], doc_word_counts[d,:])


          # quit if converged
          err = linalg.norm(old_doc_topic_params - doc_topic_params[d], 1) / n_topics
          if err  < 1e-5:
            #self.logger.debug('doc-topic difference: %f' % (err,))
            #self.logger.debug('Sweet escape!')
            break
          else:
            #self.logger.debug('doc-topic difference: %f' % (err,))
            pass

      # update q(word | topic)
      for t in range(n_topics):
        topic_word_params[t] = topic_word_prior
        for w in range(n_words):
          topic_word_params[t,w] += np.dot(doc_word_topic_params[:,w,t], doc_word_counts[:,w])

      # quit if converged
      err = linalg.norm(old_topic_word_params - topic_word_params) / (n_topics * n_words)
      if err  < 1e-5:
        print 'topic-word difference: %f' % (err,)
        print 'Finally done!'
        break
      else:
        print 'topic-word difference: %f' % (err,)

    return {
        'doc_topic': doc_topic_params,
        'topic_word': topic_word_params,
        'doc_word_topic': doc_word_topic_params,
        'word_index': word_index
    }

  def _initialize(self, documents):
    # change words into indices
    (documents, word_index) = reindex(documents)

    # figure out number of topics
    n_topics = (
        self.n_topics
        if self.n_topics is not None
        else len(self.doc_topic_prior)
    )
    n_words = len(word_index)
    n_docs = len(documents)

    # build a doc-word count matrix
    doc_word_counts = np.zeros( (n_docs, n_words) )
    for (d, doc) in enumerate(documents):
      for (i, word) in enumerate(doc):
        doc_word_counts[d,word] += 1

    # build doc-topic and topic-word priors
    if isinstance(self.doc_topic_prior, Number):
      concentration = self.doc_topic_prior
      base = np.ones(n_topics) / n_topics
    else:
      concentration = 1.0
      base = self.doc_topic_prior
    doc_topic_prior = concentration * base

    if isinstance(self.topic_word_prior, Number):
      concentration = self.topic_word_prior
      base = np.ones(n_words) / n_words
    else:
      concentration = 1.0
      base = self.topic_word_prior
    topic_word_prior = concentration * base

    return (doc_word_counts, doc_topic_prior, topic_word_prior, word_index)

  def elbo(self, doc_word_counts, doc_word_topic_params, doc_topic_params,
      topic_word_params, doc_topic_prior, topic_word_prior):
    """Calculate expected lower bound of the variational approximation

    This should increase with each step in inference
    """
    n_docs = doc_word_counts.shape[0]
    n_words = doc_word_counts.shape[1]
    n_topics = topic_word_params.shape[0]

    assert np.all(doc_topic_params > 0), 'Invalid doc-topic parameters!'
    assert np.all(topic_word_params > 0), 'Invalid topic-word parameters!'

    # compute expectations
    E_doc_topic = np.zeros( (n_docs, n_topics) )
    for (d,t) in itertools.product(range(n_docs), range(n_topics)):
      E_doc_topic[d,t] = digamma(doc_topic_params[d,t]) - digamma(np.sum(doc_topic_params[d,:]))

    E_topic_word = np.zeros( (n_topics, n_words) )
    for (t,w) in itertools.product(range(n_topics), range(n_words)):
      E_topic_word[t,w] = digamma(topic_word_params[t,w]) - digamma(np.sum(topic_word_params[t,:]))

    # compile result
    result = 0.0
    for d in range(n_docs):
      for w in np.nonzero(doc_word_counts[d])[0]:
        for t in range(n_topics):
          result += default_on_nan(
              doc_word_counts[d,w] * doc_word_topic_params[d,w,t] * (
                E_doc_topic[d,t] + E_topic_word[t,w] - np.log(doc_word_topic_params[d,w,t])
              ),
              0.0
          )
      result -= gammaln(np.sum(doc_topic_params[d,:]))
      for t in range(n_topics):
        result += (doc_topic_prior[t] - doc_topic_params[d,t]) * E_doc_topic[d,t] + gammaln(doc_topic_params[d,t])
    for t in range(n_topics):
      result -= gammaln(np.sum(topic_word_params[t,:]))
      for w in range(n_words):
        result += (topic_word_prior[w] - topic_word_params[t,w]) * E_topic_word[t,w] + gammaln(topic_word_params[t,w])
    result += n_docs * gammaln(np.sum(doc_topic_prior))
    for t in range(n_topics):
      result -= n_docs * gammaln(doc_topic_prior[t])
    result += gammaln(np.sum(topic_word_prior))
    for w in range(n_words):
      result -= gammaln(topic_word_prior[w])

    # done
    return result
