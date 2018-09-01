#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-26 (year-month-day)

"""
LIPO and related procedures

Inspired by
Ćedric Malherbe and Nicolas Vayatis, "Global Optimization of Lipschitz Functions",
arXiv:1703.02628v3 [stat.ML] 15 Jun 2017

and discussion at
Davis King, "A Global Optimization Algorithm Worth Using"
http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html
"""

import quadprog as qp

import numpy as np
import lipo.common as common
from scipy.spatial.distance import cdist, euclidean, sqeuclidean


class AdaLIPO(common.AbstractGlobalOptimizer):
  def __init__(self,
               objective_function,
               bounding_box,
               initialization_data,
               argmin=None,
               fmin=None,
               minimize=True,
               batch_size=512,
               verbose=False,
               *args,
               **kwargs):
    """
    :param objective_function: function object - the function under maximization; must return a float
    :param bounding_box: iterable containing the max and min for each dimension. Order of max and min is irrelevant,
     but all outputs depend on the order in which the dimensions are supplied.
     For example, suppose you want a bounding box on (Z x Y) = [2, -2] x [-5, 5]. You could supply
       [(2,-2),(-5,5)] or
       [[-2,2],[5,-5]] or
       np.array([[2,-2],[5,-5])
     or similar as each of these will be iterated in the order of first [-2,2], and second [-5,5].
    :param initialization_data: locations at which that you have already evaluated objective_function
    :param argmin: np.array - optional - location of the minimum; useful for benchmarking
    :param fmin: float - optional - the value at the minimum; useful for benchmarking
    :param minimize: bool - if False then the procedure is maximization; default True
    :param batch_size: positive int - instead of directly optimizing the Lipschitz constraints, random samples from the
      bounding box are taken until the constraint is satisfied. The samples are of size `batch_size` at each iteration.
    :param verbose: bool - True prints informational messages
    :param args:
    :param kwargs:
    """
    self._minimize = minimize
    # in minimization mode, flip all the y values
    if self.minimize:
      y = -1.0 * initialization_data["y"]
      initialization_data.update({"y": y})
    super().__init__(objective_function=objective_function,
                     bounding_box=bounding_box,
                     initialization_data=initialization_data,
                     argmin=argmin,
                     verbose=verbose,
                     fmin=fmin)
    self.k = 0.0
    self.sigma = None
    self.distance_lookup = dict()
    self._batch_size = batch_size

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def method_name(self):
    return "AdaLIPO"

  @property
  def minimize(self):
    return self._minimize

  @property
  def obj_fn(self):
    if self.minimize:
      return lambda x: -self._obj_fn(x)
    else:
      return self._obj_fn

  def report_results(self):
    if self.minimize:
      y = -self.y
      best_ndx = y.argmin()
    else:
      y = self.y
      best_ndx = y.argmax()
    return self.x[best_ndx, :], y[best_ndx]

  def update_distances(self):
    for i in range(len(self.x)):
      u = self.x[i, :]
      for j in range(i, len(self.x)):
        v = self.x[j, :]
        try:
          self.distance_lookup[(i, j)]
        except KeyError:
          w = euclidean(u, v)
          self.distance_lookup.update({(i, j): w})
          self.distance_lookup.update({(j, i): w})

  def get_constraints(self):
    self.update_distances()
    n = 1 + len(self.x)
    C = [[1.0] + (n - 1) * [0.0]]
    b = [self.k]
    # constraints on k, sigma[i]
    for k in range(1, n):
      C_new = n * [0.0]
      C_new[k] = 1.0
      C.append(C_new)
      b.append(0.0)
    # constraints U(x_i) >= f(x_i)
    # NB - constraints are not symmetric wrt i,j due to sigma[j]
    for i in range(len(self.x)):
      for j in [jj for jj in range(len(self.x)) if jj != i]:
        C_new = n * [0.0]
        C_new[0] = self.distance_lookup[(i, j)] ** 2
        C_new[j + 1] = 1.0
        C.append(C_new)
        b.append((self.y[i] - self.y[j]) ** 2)
    C_out = np.matrix(C).T
    b_out = np.array(b)
    return C_out, b_out

  def set_k_sigma(self, sigma_penalty=1e7):
    n = 1 + len(self.x)
    G = sigma_penalty * np.eye(n)
    G[0, 0] = 1.0
    C, b = self.get_constraints()
    a = np.zeros(n)
    qp_soln = qp.solve_qp(G=G, a=a, C=C, b=b, meq=0)
    x = qp_soln[0]
    self.k = float(x[0])
    self.sigma = x[1:]

  def lipschitz_surrogate_fn(self, z):
    Uz_proto = self.y + np.sqrt(self.sigma + self.k * cdist(z, self.x, sqeuclidean))
    Uz = Uz_proto.min(axis=1)
    return Uz

  def exploit(self, size=1):
    max_y = self.y.max()
    while_counter = -1
    z = np.zeros((0, self.d))
    while len(z) < size and while_counter < 1000:
      while_counter += 1
      new_z = self.random_sample(self.batch_size)
      Uz_candidates = self.lipschitz_surrogate_fn(new_z)
      subset = np.argwhere(Uz_candidates > max_y).flatten()
      z = np.append(z, new_z[subset, :], axis=0)
    if len(z) >= size:
      new_x = z[:size, :]  # .reshape((size, self.d))
      new_y = self.obj_fn(new_x)
    else:
      new_x, new_y = self.explore()
    return new_x, new_y

  def explore(self):
    new_x = self.random_sample(1)
    new_y = self.obj_fn(new_x)
    return new_x, new_y

  def fit(self, niter=60, iter_start=0):
    self._validate_fit_inputs(niter=niter, iter_start=iter_start)
    # increment end iterations
    niter += iter_start
    for j in range(iter_start, niter):
      msg = ""
      self.set_k_sigma()
      if j % 2 == 1:
        new_x, new_y = self.exploit()
      else:
        new_x, new_y = self.explore()
      self.update_x_y(new_x=new_x, new_y=new_y)
      if self.verbose:
        if self.minimize:
          msg += "Iteration %d - New f(x): %.6f; best f(x) so far: %.6f" % (j, -new_y, (-self.y).min())
        else:
          msg += "Iteration %d - New f(x): %.6f; best f(x) so far: %.6f" % (j, new_y, self.y.max())
        print(msg)
    # do this here so that other functions will work correctly...
    self.set_k_sigma()
    return self.report_results()


class RandomAdaLIPO(AdaLIPO):
  def __init__(self,
               objective_function,
               bounding_box,
               initialization_data,
               pr_explore=0.5,
               argmin=None,
               fmin=None,
               minimize=True,
               batch_size=256,
               verbose=False,
               *args,
               **kwargs):
    """
    :param objective_function: function object - the function under maximization; must return a float
    :param bounding_box: iterable containing the max and min for each dimension. Order of max and min is irrelevant,
     but all outputs depend on the order in which the dimensions are supplied.
     For example, suppose you want a bounding box on (Z x Y) = [2, -2] x [-5, 5]. You could supply
       [(2,-2),(-5,5)] or
       [[-2,2],[5,-5]] or
       np.array([[2,-2],[5,-5])
     or similar as each of these will be iterated in the order of first [-2,2], and second [-5,5].
    :param initialization_data: locations at which that you have already evaluated objective_function
    :param argmin: np.array - optional - location of the minimum; useful for benchmarking
    :param fmin: float - optional - the value at the minimum; useful for benchmarking
    :param minimize: bool - if False then the procedure is maximization; default True
    :param batch_size: positive int - instead of directly optimizing the Lipschitz constraints, random samples from the
      bounding box are taken until the constraint is satisfied. The samples are of size `batch_size` at each iteration.
    :param verbose: bool - True prints informational messages
    :param args:
    :param kwargs:
    """
    self._pr_explore = pr_explore
    super().__init__(objective_function=objective_function,
                     bounding_box=bounding_box,
                     initialization_data=initialization_data,
                     argmin=argmin,
                     fmin=fmin,
                     minimize=minimize,
                     batch_size=batch_size,
                     verbose=verbose, )
    self.explore_hist = []

  @property
  def pr_explore(self):
    return self._pr_explore

  def fit(self, niter=60, iter_start=0):
    self._validate_fit_inputs(niter=niter, iter_start=iter_start)

    # increment end iterations
    niter += iter_start
    for j in range(iter_start, niter):
      msg = ""
      self.set_k_sigma()
      explore_indicator = np.random.binomial(n=1, p=self.pr_explore, size=1)[0]
      self.explore_hist.append(explore_indicator)
      if explore_indicator:
        new_x, new_y = self.explore()
      else:
        new_x, new_y = self.exploit()
      self.update_x_y(new_x=new_x, new_y=new_y)

      if self.verbose:
        if self.minimize:
          msg += "Iteration %d - New f(x): %.6f; best f(x) so far: %.6f" % (j, -new_y, (-self.y).min())
        else:
          msg += "Iteration %d - New f(x): %.6f; best f(x) so far: %.6f" % (j, new_y, self.y.max())
        print(msg)
    # do this here so that other functions will work correctly...
    self.set_k_sigma()
    return self.report_results()
