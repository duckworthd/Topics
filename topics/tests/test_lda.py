import unittest

import numpy as np
from scipy.linalg import norm

import topics.lda as lda


def check_distributions(est, truth, tol):
  """Check that two topic-word sets of topic-word distributions are close"""
  assert est.shape == truth.shape
  (n_topics, n_words) = est.shape
  for t in range(n_topics):
    errs = [norm(est[t] - truth[t2], np.inf) for t2 in range(n_topics)]
    assert min(errs) < tol


def sample(
    doc_topic_prior = 1.0,
    topic_word_prior = 1.0,
    n_topics = 3,
    n_words = 10,
    n_docs = 500,
    doc_length = 10
  ):
  """Generate a sample from LDA"""
  sampler = lda.Sampler(
      doc_topic_prior=doc_topic_prior,
      topic_word_prior=topic_word_prior,
      n_topics=n_topics,
      n_words=n_words
  )
  truth = sampler.sample(
      n_docs=n_docs,
      doc_length=doc_length
  )
  return truth


def test_gibbs():
  # parameters
  doc_topic_prior = 1.0
  topic_word_prior = 2.0
  n_topics = 3
  n_words = 10
  n_docs = 500
  doc_length = 10
  n_sweeps = 30

  # generate sample
  truth = sample(
    doc_topic_prior=doc_topic_prior,
    topic_word_prior=topic_word_prior,
    n_topics=n_topics,
    n_words=n_words,
    n_docs=n_docs,
    doc_length=doc_length
  )

  # create inference class
  inf = lda.Gibbs(
      doc_topic_prior=doc_topic_prior,
      topic_word_prior=topic_word_prior,
      n_topics=n_topics
  )

  # run inference
  for (iteration, state) in enumerate(inf.infer(truth['documents'], n_sweeps=n_sweeps)):
    pass

  # make sure samples match
  check_distributions(state['topic_word'], truth['topic_word'], 0.03)


def test_spectral():
  # parameters
  doc_topic_prior = 1.0
  topic_word_prior = 0.85
  n_topics = 3
  n_words = 10
  n_docs = 20000
  doc_length = 3
  n_sweeps = 30

  # generate sample
  truth = sample(
    doc_topic_prior=doc_topic_prior,
    topic_word_prior=topic_word_prior,
    n_topics=n_topics,
    n_words=n_words,
    n_docs=n_docs,
    doc_length=doc_length
  )

  # create inference class
  inf = lda.Spectral(
      doc_topic_prior=doc_topic_prior,
      n_topics=n_topics
  )
  result = inf.infer(truth['documents'], n_sweeps=5)

  # make sure samples match
  check_distributions(result['topic_word'], truth['topic_word'], 0.065)


def test_variational():
  # parameters
  doc_topic_prior = 1.0
  topic_word_prior = 0.95
  n_topics = 3
  n_words = 5
  n_docs = 500
  doc_length = 5
  n_sweeps = 10

  # generate sample
  truth = sample(
    doc_topic_prior=doc_topic_prior,
    topic_word_prior=topic_word_prior,
    n_topics=n_topics,
    n_words=n_words,
    n_docs=n_docs,
    doc_length=doc_length
  )

  # create inference class
  inf = lda.Variational(
      doc_topic_prior=doc_topic_prior,
      n_topics=n_topics
  )
  result = inf.infer(truth['documents'], n_sweeps=n_sweeps)

  # make sure samples match
  check_distributions(result['topic_word'], truth['topic_word'], 0.41)
