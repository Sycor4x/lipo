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

from scipy.spatial.distance import euclidean, cdist
import numpy as np
import optimization_test_functions as otf

import argparse


def verify_positive_int(a):
  if not isinstance(a, int):
    raise ValueError("niter must be positive int")
  if a < 1:
    raise ValueError("niter must be positive int")


class LIPO(object):
  def __init__(self, objective_function, bounding_box, x_star=None):
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

    self.x = np.zeros((0, self.d))
    self.y = np.zeros((0, 1))
    self.k = -1
    self.distance_lookup = dict()

  @property
  def t(self):
    return self._t

  @property
  def niter(self):
    return self._niter

  @property
  def d(self):
    return self._d

  @property
  def obj_fn(self):
    return self._obj_fn

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
    return out.reshape((size, self.d))

  def decision_function(self, new_x):
    self.set_k()
    upper_bound = float("inf")
    for i in range(len(self.x)):
      ubi = self.y[i, 0] + self.k * euclidean(new_x, self.x[i, :])
      upper_bound = min(upper_bound, ubi)
    condition_test = upper_bound >= np.max(self.y)
    return condition_test

  def set_k(self):
    """
    Estimates the Lipschitz constant.
    """
    # TODO - replace this with an actual estimator
    k_new = self.k

    for i in range(len(self.x)):
      for j in range(i + 1, len(self.x)):
        try:
          w = self.distance_lookup[(i, j)]
        except KeyError:
          xi, xj = self.x[i, :], self.x[j, :]
          w = euclidean(xi, xj)
          self.distance_lookup.update({(i, j): w})
        # print("number of distances: %d" % len(self.distance_lookup))
        yi, yj = self.y[i, 0], self.y[j, 0]
        k_tilde = np.abs(yi - yj) / w
        k_new = max(k_tilde, k_new)
    self.k = k_new * 1.5

  def upper_bound(self, z):
    # upper_bound_function = self.y + self.k * cdist(z, self.x)
    print(self.y)
    print(self.k * cdist(z, self.x))
    upper_bound_function = self.k * cdist(z, self.x) + self.y

    print(upper_bound_function.shape)
    print(upper_bound_function)
    print(upper_bound_function.min(axis=1))
    asdf
    return upper_bound_function.min()

  def fit(self, niter=60, nstart=6):
    verify_positive_int(niter)
    verify_positive_int(nstart)
    self._niter += niter

    i = 0
    for i in range(nstart):
      new_x = self.sample_next()
      new_y = self.obj_fn(new_x)
      self.x = np.vstack((self.x, new_x))
      self.y = np.vstack((self.y, new_y))

    for k in range(i, self.niter + nstart):
      new_x = self.sample_next()

      z = optimizer.sample_next()
      fz = optimizer.upper_bound(self.x)
      print(z, fz)

      if self.decision_function(new_x):
        self._t += 1
        self.x = np.vstack((self.x, new_x))
        new_y = self.obj_fn(new_x)
        self.y = np.vstack((self.y, new_y))

    best_i = np.argmax(self.y)
    best_x = self.x[best_i, :]
    best_y = self.y[best_i, 0]
    print(self.t)
    return best_x, best_y


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
  optimizer = LIPO(**active_fn)
  x, y = optimizer.fit()

  z = optimizer.sample_next()
  fz = optimizer.upper_bound(z)
  print(z, fz)

  x = x.reshape((1, -1))
  y *= -1.0
  x_star = active_fn["x_star"]
  miss = np.min(cdist(x, x_star))
  print(u"||x_best - x*||\u2082 = %.4f" % miss)
  print(r"f(x_best) - f(x*) = %.4f" % (y - active_fn["objective_function"](x_star)))
