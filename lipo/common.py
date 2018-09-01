#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-26 (year-month-day)

"""
Functions, etc. common to the rest of the package.
"""

import argparse
import warnings

import numpy as np


def pure_random_search_sample_size(q=0.95, p=0.95):
  """
  Computes the minimum sample size required to obtain function value in the smallest q quantile with probability p
  via pure random search. The most common usage assigns q = 0.05 and p = 0.95, whence one finds the rule-of-thumb of
  n = 60 random hyperparameter tuples.
  :param q: float in (0.0, 1.0) - the target quantile
  :param p: float in (0.0, 1.0) - the target probability
  :return: int
  """
  if not (0.0 < q < 1.0) or not (0.0 < p < 1.0):
    raise ValueError("Both p and q must be probabilities.")
  out = np.log(1.0 - p) // np.log(q)
  out += 1  # operation a//b is floor division
  return int(out)


class AbstractGlobalOptimizer:
  def __init__(self,
               objective_function,
               bounding_box,
               initialization_data,
               argmin=None,
               fmin=None,
               verbose=False,
               *args,
               **kwargs):
    # TODO - make argmin do something
    """
    :param objective_function: function object - the function under minimization; must return a scalar real number
    :param bounding_box: iterable - contains the max and min for each dimension. Order of max and min is irrelevant,
      but all outputs depend on the order in which the _dimensions_ are supplied.
      For example, suppose you want a bounding box on (Z x Y) = [2, -2] x [-5, 5]. You could supply
        [(2,-2),(-5,5)] or
        [[-2,2],[5,-5]] or
        np.array([[2,-2],[5,-5])
      or similar as each of these will be iterated in the order of first [-2,2], and second [-5,5].
      Note that all computations are done on the scale of the supplied parameters. If you want to do something like fit
      a parameter that varies on the log scale, supply log-scaled coordinates and then do the appropriate
      back-transformation in the call to the objective_function.
    :param initialization_data: dictionary of np arrays with keys "x" and "y".
      array "x" must have shape (N, d).
      array "y" must have N entries.
    :param argmin: np.array - where the objective function is minimized
    :param fmin: np.array - the minimum objective function value
    :param verbose: bool - print informational messages
    """
    self._verbose = verbose
    self._obj_fn = objective_function
    self._argmin = argmin
    self._fmin = fmin
    box = []
    for i, (x1, x2) in enumerate(bounding_box):
      if np.isclose(x1, x2):
        warnings.warn("The interval for dimension %d is too short to be plausible: [%s, %s]." %
                      (i, min(x1, x2), max(x1, x2)))
      box.append([min(x1, x2), max(x1, x2)])
    box = np.array(box)
    self._d = len(box)
    self.box = box

    # load and validate initial objective function values
    self.x = initialization_data["x"]
    self.y = initialization_data["y"]
    if self.x.shape[1] != self.d:
      raise ValueError("x has the wrong shape - x.shape[1] must equal len(bounding_box)=%d" % self.d)
    if self.y.size != len(self.x):
      raise ValueError("y must have len(x)=%d entries." % len(self.x))
    self.y.reshape((-1, 1))  # 2-dimensional array with 1 column

  @property
  def verbose(self):
    return self._verbose

  @property
  def fmin(self):
    return self._fmin

  @property
  def argmin(self):
    return self._argmin

  @property
  def d(self):
    return self._d

  @property
  def obj_fn(self):
    return self._obj_fn

  def random_sample(self, size=1):
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

  def update_x_y(self, new_x, new_y):
    self.x = np.vstack((self.x, new_x))
    self.y = np.append(self.y, new_y)

  def report_results(self):
    best_ndx = self.y.argmin()
    return self.x[best_ndx, :], self.y[best_ndx]

  def _validate_fit_inputs(self, niter, iter_start):
    # validate inputs
    if not all([isinstance(par, int) for par in [niter, iter_start]]):
      raise ValueError("arguments niter, batch size and iter_start must all be positive int.")
    if not (niter >= 1 or iter_start >= 0):
      raise ValueError("Arguments niter and batch size must be positive and argument iter_start must be non-negative.")

  def fit(self, niter=60, iter_start=0):
    raise NotImplementedError("this is implemented in child classes")

  def fit_benchmark(self, target, max_obj_fn_evals=100):
    best_x, best_y = self.report_results()
    start_counter = self.y.size + 1

    for i in range(start_counter, max_obj_fn_evals + 1):
      if best_y <= target:
        break
      self.fit(niter=1, iter_start=i)
      best_x, best_y = self.report_results()
    if self.verbose:
      print("Target achieved in in %d iterations." % self.y.size)
    return best_x, best_y, self.y.size


class PureRandomSearch(AbstractGlobalOptimizer):
  def __init__(self,
               objective_function,
               bounding_box,
               initialization_data=None,
               argmin=None,
               fmin=None,
               verbose=False,
               *args,
               **kwargs):
    box = []
    for i, (x1, x2) in enumerate(bounding_box):
      if np.isclose(x1, x2):
        warnings.warn("The interval for dimension %d is too short to be plausible: [%s, %s]." %
                      (i, min(x1, x2), max(x1, x2)))
      box.append([min(x1, x2), max(x1, x2)])
    box = np.array(box)
    self._d = len(box)
    self.box = box

    # load and validate initial objective function values
    if isinstance(initialization_data, int):
      x = self.random_sample(initialization_data)
      y = objective_function(x)
      init_data = {
        "x": x,
        "y": y
      }
    else:
      init_data = initialization_data
    super().__init__(objective_function,
                     bounding_box,
                     initialization_data=init_data,
                     verbose=verbose,
                     fmin=fmin,
                     argmin=argmin)

  @property
  def method_name(self):
    return "PureRandomSearch"

  def fit(self, niter=59, iter_start=0):
    """

    :param niter: int - How many random iterations to take.
    :param iter_start: int - ignored here
    :return: results
    """
    random_x = self.random_sample(niter)
    for new_x in random_x:
      new_y = self.obj_fn(new_x)
      self.update_x_y(new_x=new_x, new_y=new_y)
    return self.report_results()


def parse_args():
  """Parses arguments"""
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--function_name", type=str, default="goldstein_price", required=True,
                      choices=["goldstein_price",
                               "branin",
                               "gramacy_lee",
                               "modified_branin",
                               "bukin_6",
                               "sphere",
                               "camel_hump_6",
                               "hartmann_6",
                               "forrester",
                               "colville",
                               "holder_table",
                               "rosenbrock"],
                      help="string naming which optimization test function to use")
  parser.add_argument("-n", "--niter", type=int, default=100,
                      help="budget for function evaluations")
  args_out = parser.parse_args()
  return args_out
