"""
Miscellaneous functions
"""
import itertools

import numpy as np
import scipy.sparse as sp


def reindex(documents):
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
  '''Sample from a categorical distribution'''
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
  if np.isnan(f):
    return default
  else:
    return f
