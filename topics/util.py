"""
Miscellaneous functions
"""
import itertools

import numpy as np
import scipy.sparse as sp


def reindex(documents):
  """Transform lists of words into lists of indices

  Parameters
  ----------
  documents : [[str]]
      a list whose elements are lists of words

  Returns
  -------
  documents2 : [[int]]
      a list whose elements are list of word indices
  word_index : {str:int}
      a mapping from words to word indices
  """
  words = set(itertools.chain(*documents))
  word_index = dict( (w,i) for (i,w) in enumerate(sorted(words)) )
  documents2 = []
  for (d, doc) in enumerate(documents):
    doc2 = []
    for (i, word) in enumerate(doc):
      doc2.append(word_index[word])
    documents2.append(doc2)
  return (documents2, word_index)


def flatten_counts(s):
  '''Transform [1,0,3] into [0,2,2,2]'''
  result = []
  for (i,v) in enumerate(s):
      result.extend([i]*v)
  return result


def categorical(p, r=None):
  '''Sample from a categorical distribution

  Parameters
  ----------
  p : array
      discrete probability distribution. Assumed to sum to 1.
  r : RandomState or None
      if given, use this random number generator
  '''
  if r is None:
    np.random.get_state()
  counts = r.multinomial(1, p)
  return np.nonzero(counts)[0][0]


def doc_word_matrix(documents, n_words):
  """Create a sparse document-word count matrix"""
  n_docs = len(documents)
  coordinates = []
  for (row, doc) in enumerate(documents):
    for col in doc:
      coordinates.append( (row, col, 1) )
  rows, cols, vals = zip(*coordinates)
  M = sp.coo_matrix((vals, (rows, cols)), dtype=float)
  return M.tocsr()


def default_on_nan(f, default):
  """If f is nan, return a default value instead"""
  if np.isnan(f):
    return default
  else:
    return f


def normalize(arr):
  """Normalize each row of a 2D array to sum to 1"""
  s = np.sum(arr, axis=1)
  return arr / s[:, np.newaxis]


def qualified_name(func):
  """Get fully qualified name of a function or class"""
  return "%s.%s" % (func.__module__, func.__name__)
