#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (YYYY-MM-DD)

"""
An implementation of

Cédric Malherbe and Nicolas Vayatis, "Global optimization of Lipschitz functions"
https://arxiv.org/abs/1703.02628

with some experimental modifications also implemented.
"""

from __future__ import division

import numpy as np
import os


class LIPO(object):
  def __init__(self, objective_functon, bounding_box, acquisition_function, stan_surrogate_model_path):
    """

    :param objective_functon: function object - the function under minimization; must return a float
    :param bounding_box: iterable containing the max and min for each dimension. Order of max and min is irrelevant,
     but all outputs depend on the order in which the dimensions are supplied.
     For example, suppose you want a bounding box on (Z x Y) = [2, -2] x [-5, 5]. You could supply
       [(2,-2),(-5,5)] or
       [[-2,2],[5,-5]] or
       np.array([[2,-2],[5,-5])
     or similar as each of these will be iterated in the order of first [-2,2], and second [-5,5].
    :param acquisition_function: string - string must be in {"PI", "EQI", "EI", "UCB"}
      PI - probability of improvement
      EQI - expected quantile improvement
      EI - expected improvement
      UCB - upper confidence bound
    :param stan_surrogate_model_path:
    """
    self._obj_fn = objective_functon
    self.bounding_box = bounding_box
    self.acquisition_function = acquisition_function

    try:
      with open(stan_surrogate_model_path) as f:
        self._model_str = f.read()
    except FileNotFoundError:
      load_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), stan_surrogate_model_path)
      with open(load_path) as f:
        self._model_str = f.read()

  @property
  def obj_fn(self):
    return self._obj_fn

  @property
  def model_str(self):
    return self._model_str

  def fit(self):
    return


if __name__ == "__main__":
  def gramacy_lee_2012(x):
    if not (0.5 < x < 2.5):
      raise ValueError("provided value of x not in [0.5, 2.5].")
    y = np.sin(10.0 * np.pi * x)
    y /= 2 * x
    y += np.power(x - 1.0, 4.0)
    return y


  pass
