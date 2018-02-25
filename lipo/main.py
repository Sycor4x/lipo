#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (year-month-day)

"""
An implementation of

Cédric Malherbe and Nicolas Vayatis, "Global optimization of Lipschitz functions"
https://arxiv.org/abs/1703.02628

with some experimental modifications also implemented.
"""

from __future__ import division

import argparse
import quadprog as qp

import numpy as np
from scipy.spatial.distance import euclidean, cdist, sqeuclidean

import optimization_test_functions as otf


def verify_int(a, minimum=None, maximum=None):
  if not isinstance(a, int):
    raise ValueError("niter must be int")
  if minimum is not None:
    if a < minimum:
      raise ValueError("niter must be greater than %s, but supplied value is %d" % (minimum, a))
  if maximum is not None:
    if a > maximum:
      raise ValueError("niter must be less than %s, but supplied value is %d" % (maximum, a))


class SimpleLIPO(object):
  def __init__(self, objective_function, bounding_box, minimize=False, x_star=None):
    # TODO - make x_star do something
    """
    IMPORTANT NOTE: this procedure is maximization; can implicitly do minimization of some function f by providing -f.

    :param objective_function: function object - the function under maximization; must return a float
    :param bounding_box: iterable containing the max and min for each dimension. Order of max and min is irrelevant,
     but all outputs depend on the order in which the dimensions are supplied.
     For example, suppose you want a bounding box on (Z x Y) = [2, -2] x [-5, 5]. You could supply
       [(2,-2),(-5,5)] or
       [[-2,2],[5,-5]] or
       np.array([[2,-2],[5,-5])
     or similar as each of these will be iterated in the order of first [-2,2], and second [-5,5].
    """
    self._minimize = minimize
    self._obj_fn = objective_function
    self._niter = 0
    self._t = 0
    box = []
    for i, (x1, x2) in enumerate(bounding_box):
      if np.isclose(x1, x2):
        raise ValueError("The interval for dimension %d is too short to be plausible: [%s, %s]." %
                         (i, min(x1, x2), max(x1, x2)))
      box.append([min(x1, x2), max(x1, x2)])
    box = np.array(box)
    self._d = len(box)
    self.box = box

    self.x = np.array([]).reshape((-1, self.d))
    self.y = np.array([])
    self.k = 0.0
    self.sigma = None
    self.distance_lookup = dict()

  @property
  def t(self):
    return self._t

  @property
  def minimize(self):
    return self._minimize

  @property
  def niter(self):
    return self._niter

  @property
  def d(self):
    return self._d

  @property
  def obj_fn(self):
    if self.minimize:
      return lambda x: -self._obj_fn(x)
    else:
      return self._obj_fn

  def update_x_y(self, new_x, new_y):
    self.x = np.vstack((self.x, new_x))
    self.y = np.append(self.y, new_y)

  def sample_next(self, size=1):
    """
    samples a new point uniformly from within the bounding box
    :return:
    """
    out = np.zeros((0, self.d))
    for j in range(size):
      new_x = np.zeros(self.d)
      for i in range(self.d):
        new_x[i] = np.random.uniform(low=self.box[i, 0], high=self.box[i, 1], size=1)
      out = np.vstack((out, new_x))
    return np.matrix(out).reshape((size, self.d))

  def update_distances(self):
    for i in range(len(self.x) - 1):
      for j in range(i + 1, len(self.x)):
        try:
          self.distance_lookup[(i, j)]
        except KeyError:
          w = euclidean(self.x[i, :], self.x[j, :])
          self.distance_lookup.update({(i, j): w})

  def get_constraints(self, n):
    self.update_distances()
    C = [[1.0] + (n - 1) * [0.0]]
    # b = [self.k]
    b = [0.0]
    # constraints on k, sigma[i]
    for k in range(1, n):
      C_new = n * [0.0]
      C_new[k] = 1.0
      C.append(C_new)
      b.append(0.0)
    # constraints U(x_i) >= f(x_i)
    # Only need to do all combinations i,j instead of all pairs because the constraints are symmetric wrt i, j.
    for i in range(len(self.x) - 1):
      for j in range(i + 1, len(self.x)):
        C_new = n * [0.0]
        C_new[0] = self.distance_lookup[(i, j)] ** 2
        C_new[j + 1] = 1.0
        C.append(C_new)
        b.append((self.y[i] - self.y[j]) ** 2)
    C_out = np.matrix(C).T
    b_out = np.array(b)
    return C_out, b_out

  def set_k_sigma(self):
    n = 1 + len(self.x)
    G = 1e6 * np.eye(n)
    G[0, 0] = 1.0
    C, b = self.get_constraints(n)
    a = np.zeros(n)
    qp_soln = qp.solve_qp(G=G, a=a, C=C, b=b, meq=0)
    x = qp_soln[0]
    self.k = float(x[0])
    self.sigma = x[1:]

  def lipschitz_surrogate_fn(self, z):
    Uz_proto = self.y + np.sqrt(self.sigma + self.k * cdist(z, self.x, sqeuclidean))
    Uz = Uz_proto.min(axis=1)
    ndz = Uz.argmax()
    return z[ndz]

  def explore(self, n_explore):
    z = self.sample_next(n_explore)
    new_x = self.lipschitz_surrogate_fn(z)
    new_y = self.obj_fn(new_x)
    self.update_x_y(new_x=new_x, new_y=new_y)

  def exploit(self):
    return

  def fit(self, niter=55, nstart=3, explore_batch=100):
    for i in range(nstart):
      new_x = self.sample_next()
      new_y = self.obj_fn(new_x)
      self.update_x_y(new_x=new_x, new_y=new_y)

    for j in range(niter):
      self.set_k_sigma()
      self.explore(explore_batch)


def parse_args():
  """Parses arguments"""
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--function_name", type=str, default="goldstein_price", required=True,
                      choices=["goldstein_price",
                               "branin",
                               "gramacy_lee"],
                      help="string naming which optimization test function to use")
  parser.add_argument("-n", "--niter", type=int, default=100,
                      help="budget for function evaluations")
  args_out = parser.parse_args()
  return args_out


function_options = {
  "goldstein_price": {
    "objective_function": lambda x: -otf.goldstein_price(x),
    "bounding_box": [[-2.0, 2.0], [-2.0, 2.0]],
    "x_star": np.array([[0.0, -1.0]])
  },
  "branin": {
    "objective_function": lambda x: -otf.branin(x),
    "bounding_box": [[-5, 10], [0, 15]],
    "x_star": np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
  },
  "gramacy_lee": {
    "objective_function": lambda x: -otf.gramacy_lee(x),
    "bounding_box": [[0.5, 2.5]],
    "x_star": np.array([0.548563444114526])
  }
}

if __name__ == "__main__":
  args = parse_args()

  active_fn = function_options[args.function_name]
  optimizer = SimpleLIPO(**active_fn, minimize=True)

  optimizer.fit(niter=10, explore_batch=5)
