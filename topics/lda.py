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

from util import categorical, default_on_nan, flatten_counts, normalize, reindex


class Sampler(object):
  """Sampler for Latent Dirichlet Allocation Model

  Warning: numpy's dirichlet sampler can't seem to handle small values for
  doc_topic_prior or topic_word_prior. Try to keep values above 0.1.
  """
  def __init__(self, doc_topic_prior=0.1, topic_word_prior=0.1, n_topics=None, n_words=None):
    self.n_topics = n_topics
    self.n_words = n_words
    self.doc_topic_prior = doc_topic_prior
    self.topic_word_prior = topic_word_prior
    self.logger = logging.getLogger(str(self.__class__))

  def sample(self, n_docs, doc_length, n_topics=None, n_words=None):
    (doc_topic_prior, topic_word_prior) = self._initialize(n_topics, n_words)
    n_topics = len(doc_topic_prior)
    n_words = len(topic_word_prior)
    r = np.random.RandomState(0)

    # sample topic-word distributions
    topic_word = r.dirichlet(topic_word_prior, n_topics)

    # sample doc-topic distributions
    doc_topic = r.dirichlet(doc_topic_prior, n_docs)

    # sample documents themselves
    all_topics = []
    all_words = []
    for d in range(n_docs):
      topics = flatten_counts(r.multinomial(doc_length, doc_topic[d]))
      words = []
      for (i, topic) in enumerate(topics):
        # sample topic for this word, word itself
        words.append(categorical(topic_word[topic], r))

      # save this document's topics, words
      all_topics.append(topics)
      all_words.append(words)

    return {
        'topic_word': topic_word,
        'doc_topic': doc_topic,
        'word_topic': all_topics,
        'documents': all_words
    }

  def _initialize(self, n_topics, n_words):
    # get true n_topics, n_words
    if n_topics is None:
      n_topics = (
          self.n_topics
          if self.n_topics is not None
          else len(self.doc_topic_prior)
      )
    if n_words is None:
      n_words = (
          self.n_words
          if self.n_words is not None
          else len(self.topic_word_prior)
      )

    # initialize numpy array for doc-topic prior and topic-word prior
    if isinstance(self.doc_topic_prior, Number):
      doc_topic_prior = self.doc_topic_prior * np.ones(n_topics) / n_topics
    else:
      doc_topic_prior = np.atleast_1d(self.doc_topic_prior)

    if isinstance(self.topic_word_prior, Number):
      topic_word_prior = self.topic_word_prior * np.ones(n_words) / n_words
    else:
      topic_word_prior = np.atleast_1d(self.topic_word_prior)

    return (doc_topic_prior, topic_word_prior)


class Gibbs(object):
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

  def infer(self, documents, n_sweeps=1000, word_topic=None):
    r = np.random.RandomState(0)

    # initialize counts for each doc-topic and topic-word pair using the prior
    (doc_topic_counts, topic_word_counts, word_index) = (
      self._initialize(documents)
    )
    topic_counts = np.sum(topic_word_counts, axis=1)
    n_topics = topic_word_counts.shape[0]
    n_docs = doc_topic_counts.shape[0]
    n_words = len(word_index)

    # transform documents into lists of word indices
    (documents, word_index) = reindex(documents)

    # initialize topics for all words uniformly at random
    if word_topic is None:
      word_topic = [
        [
          categorical(np.ones(n_topics)/n_topics, r)
          for word in doc
        ]
        for doc in documents
      ]

    # initialize doc-topic and topic-word counts
    for (d, doc) in enumerate(documents):
      for (i, word) in enumerate(doc):
        # get topic for this word
        t = word_topic[d][i]

        # increment counts
        doc_topic_counts[d,t] += 1
        topic_word_counts[t,word] += 1
        topic_counts[t] += 1

    # resample word topics
    for sweep in range(n_sweeps):
      self.logger.debug('starting sweep #%d' % (sweep,))
      for (d, doc) in enumerate(documents):

        if d % 100 == 0:
          self.logger.debug('starting document #%d' % (d,))

        for (i, word) in enumerate(doc):
          # get topic for this word in this document
          t = word_topic[d][i]

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
          t = categorical(prob, r)

          # increment counts
          doc_topic_counts[d,t] += 1
          topic_word_counts[t,word] += 1
          topic_counts[t] += 1

          # set topic for word
          word_topic[d][i] = t

      # sum of counts along each row
      topic_word_sums = topic_counts[:, np.newaxis]
      doc_topic_sums = np.sum(doc_topic_counts, axis=1)[:, np.newaxis]

      yield {
        'topic_word': np.copy(topic_word_counts) / topic_word_sums,
        'doc_topic': np.copy(doc_topic_counts) / doc_topic_sums,
        'word_topic': copy.deepcopy(word_topic),
        'word_index': word_index
      }

  def _initialize(self, documents):
    """Parse hyperparameters and initialize counts"""
    words = set(itertools.chain(*documents))
    # get actual number of topics
    n_topics = (
        self.n_topics
        if self.n_topics is not None
        else len(self.doc_topic_prior)
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


class Variational(object):
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

  def infer(self, documents, n_sweeps=20):
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

    # actual inference
    for iteration in range(n_sweeps):
      self.logger.info("Starting new pass")
      try:
        ll = self.elbo(doc_word_counts, doc_word_topic_params,
            doc_topic_params, topic_word_params, doc_topic_prior,
            topic_word_prior)
        self.logger.info("Current Log Likelihood: %f" % (ll,))
      except AssertionError as e:
        logging.warn('Failed to calculate ELBO: %s' % (str(e),))


      for d in range(n_docs):
        # print out logging info
        if d % 100 == 0:
          self.logger.info("Processing document %d/%d" % (d, n_docs))
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
            break

      # update q(word | topic)
      for t in range(n_topics):
        topic_word_params[t] = topic_word_prior
        for w in range(n_words):
          topic_word_params[t,w] += np.dot(doc_word_topic_params[:,w,t], doc_word_counts[:,w])

    return {
        'doc_topic': normalize(doc_topic_params),
        'topic_word': normalize(topic_word_params),
        'word_topic': normalize(doc_word_topic_params),
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


class Spectral(object):
  """Latent Dirichlet Allocation with Method-of-Moments Inference

  Use Singular Value Decomposition to solve for the parameters of Latent
  Dirichlet Allocation as described in "Two SVDs Suffice" (Anandkumar, 2012).
  Specifically, see Algorithm 5 ("Empirical ECA for LDA")

  Parameters
  ----------
  n_topics: int
    number of topics
  doc_topic_prior: float
    concentration parameter of the Dirichlet prior over document-topic
    distributions. Currently no way to specify base distribution.
  """
  def __init__(self, n_topics, doc_topic_prior=0.01):
    self.doc_topic_prior = doc_topic_prior
    self.n_topics = n_topics
    self.logger = logging.getLogger(str(self.__class__))

  def infer(self, documents, n_sweeps=5):
    # reindex words
    (documents, word_index) = reindex(documents)

    # initialize commonly used numbers
    n_docs = len(documents)
    n_words = len(word_index)
    n_topics = self.n_topics
    r = np.random.RandomState(0)

    # 1. Calculate moments (defer third till later)
    self.logger.debug("Constructing 1st and 2nd moments")
    m1 = self.moment1(n_words, documents)
    m2 = self.moment2(n_words, documents)

    # 2. Whiten
    self.logger.debug("Doing first SVD")
    pairs = self.pairs(n_words, documents, m1=m1, m2=m2)
    (A, Sigma, _) = linalg.svd(pairs)
    A = A[:,0:n_topics]                 # first k singular vectors
    Sigma = np.diag(Sigma[0:n_topics])  # first k singular values
    W = A.dot(np.sqrt(Sigma))

    # 3. SVD

    # # SVD via random projection
    # self.logger.debug("Constructing 3rd moment")
    # axis = r.randn(n_topics); axis /= linalg.norm(axis)  # random unit norm vector
    # triples = self.triples(n_words, documents, W.dot(axis), m1=m1, m2=m2)
    # self.logger.debug("Performing second SVD")
    # V = linalg.svd(W.T.dot(triples).dot(W))[0]   # columns are left singular vectors

    # SVD via power method
    self.logger.debug("Starting power iterations")
    V = r.randn(n_topics, n_topics)  # initialize an orthonormal basis
    V = linalg.orth(V)
    for iteration in range(n_sweeps):
      self.logger.debug("iteration %d" % (iteration,))
      for t in range(n_topics):
        Wv = W.dot(V[:,t])
        triples = self.triples(n_words, documents, Wv, m1=m1, m2=m2)
        V[:,t] = W.T.dot(triples).dot(Wv)
      V = linalg.orth(V)

    # 4. Reconstruct and Normalize
    self.logger.debug("Reconstructing topic-word vectors")
    W_inv = linalg.pinv(W)
    O = np.zeros((n_words, n_topics))
    for t in range(n_topics):
      O[:,t] = W_inv.T.dot(V[:,t])

      # change sign of singular vector
      i = np.argmax(np.abs(O[:,t]))
      O[:,t] = np.sign(O[i,t]) * O[:,t]

      # drop negative components and normalize
      O[O[:,t] < 0,t] = 0
      O[:,t] /= linalg.norm(O[:,t], 1)

    return {
      'topic_word': O.T,  # each row is a topic
      'word_index': word_index,
    }

  def moment1(self, n_words, documents):
    m1 = np.zeros(n_words)  # average mean over all documents
    for doc in documents:
      m1doc = np.zeros(n_words)   # mean for this document alone
      for w in doc:
        m1doc[w] += 1.0
      m1 += m1doc / len(doc)
    m1 /= len(documents)
    return m1

  def moment2(self, n_words, documents):
    m2 = np.zeros( (n_words, n_words) )   # average covariance over all docs
    for doc in documents:
      m2doc = np.zeros((n_words, n_words))  # covariance for this doc
      total = 0
      for i in range(len(doc)):
        for j in range(i+1, len(doc)):
          if i == j: continue
          m2doc[doc[i], doc[j]] += 1.0
          total += 1
      m2 += m2doc / total
    m2 /= len(documents)
    return m2

  def moment3(self, n_words, documents, axis):
    m3 = np.zeros( (n_words, n_words) )
    for doc in documents:
      m3doc = np.zeros((n_words, n_words))
      total = 0
      for i in range(len(doc)):
        for j in range(i+1, len(doc)):
          for k in range(j+1, len(doc)):
            m3doc[doc[i], doc[j]] += 1.0 * axis[doc[k]]
            total += 1
      m3 += m3doc / total
    m3 /= len(documents)
    return m3

  def pairs(self, n_words, documents, m1=None, m2=None):
    alpha = self.doc_topic_prior

    if m1 is None:
      m1 = self.moment1(n_words, documents)

    if m2 is None:
      m2 = self.moment2(n_words, documents)

    return m2 - (alpha / (1 + alpha)) * np.outer(m1, m1)

  def triples(self, n_words, documents, axis, m1=None, m2=None, m3=None):
    alpha = self.doc_topic_prior

    if m1 is None:
      m1 = self.moment1(n_words, documents)

    if m2 is None:
      m2 = self.moment2(n_words, documents)

    if m3 is None:
      m3 = self.moment3(n_words, documents, axis)

    return (
      m3
      - (alpha / (2 + alpha)) * (
        m2.dot(np.outer(axis, m1))
        + np.outer(m1, axis).dot(m2)
        + axis.dot(m1) * m2
      )
      + (2 * alpha**2 / ((alpha + 2)*(alpha + 1))) * (
        np.dot(axis, m1) * np.outer(m1, m1)
      )
    )

