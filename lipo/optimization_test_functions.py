#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (year-month-day)

"""
Python implementations of common optimization test functions, as well as dictionaries storing their properties.
"""

import numpy as np


class ObjectiveFunction:
  def __init__(self, function_token):
    self.bounding_box_key = "bounding_box"
    self._function_token = function_token.lower()
    if self.function_token not in function_info.keys():
      raise ValueError("Function not found in the dictionary containing function information; provided value: %s")

  def __call__(self, x):
    # globals()["myfunction"]() Maybe need this
    locals()[self.function_token](x)

  @property
  def function_token(self):
    return self._function_token

  @property
  def bounding_box(self):
    return function_info.get(self.function_token).get(self.bounding_box_key)


def gramacy_lee(x):
  """
  A standard 1-dimensional global optimization test function with many local minima.
  f(x) = sin(10 pi x) / (2 x) + (x - 1) ^ 4

  There is a global minimum at

  :param x: a float in [0.5, 2.5]
  :return: f(x)
  """
  try:
    x = x.A.flatten()
  except AttributeError:
    x = x.flatten()
  if not all([0.5 <= xx <= 2.5 for xx in x]):
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")
  y = np.sin(10.0 * np.pi * x)
  y /= 2 * x
  y += np.power(x - 1.0, 4.0)
  return y


def modified_branin(x):
  fx = branin(x)
  try:
    x = x.A
  except AttributeError:
    pass
  x1 = x[:, 0]
  fx += 5 * x1 + 16.64099
  return fx


def branin(x):
  """
  A standard 2-dimensional global optimization test function with a long, shallow valley.
  f(x) = (x2 - 4 * 5.1 (x1 / pi)^2 + 5 x1 / pi - 6)^2 + 10 (1 - (8 pi)^-1 ) cos (x1) + 10
  (The general form doesn't specify coefficients.)

  There is a global minimum at [[-pi, 12.275], [pi, 2.275], [9.42478, 2.475]]

  :param x: tuple such that
    x[0] in [-5, 10]
    x[1] in [0, 15]
  :return: f(x)
  """
  try:
    x = x.A
  except AttributeError:
    pass
  x1 = x[:, 0]
  x2 = x[:, 1]

  x1_test = np.logical_or(x1 < -5.0, x1 > 10.0).any()
  x2_test = np.logical_or(x2 < 0.0, x2 > 15.0).any()
  if x1_test or x2_test:
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")

  fx = 10
  fx += np.power(x2 - 5.1 / 4 * np.power(x1 / np.pi, 2.0) + 5 / np.pi * x1 - 6, 2.0)
  fx += 10 * (1 - np.reciprocal(8 * np.pi)) * np.cos(x1)
  return fx


def goldstein_price(x):
  """
  A standard 2-dimensional global optimization test function with a large, nearly-flat region.

  There is a global minimum at (0, -1).
  :param x:
  :return:
  """
  try:
    x = x.A
  except AttributeError:
    pass
  test = np.logical_and(x <= 2.0, -2.0 <= x)
  if not test.all():
    raise ValueError("Provided value of x not in bounds for this objective function. See documentation.")
  x1 = x[:, 0]
  x2 = x[:, 1]
  factor1a = (x1 + x2 + 1) ** 2
  factor1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
  factor1 = 1 + factor1a * factor1b
  factor2a = np.power(2 * x1 - 3 * x2, 2.0)
  factor2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
  factor2 = 30 + factor2a * factor2b
  out = factor1 * factor2
  return np.array(out).reshape((-1,))


def bukin_6(x):
  try:
    x = x.A
  except AttributeError:
    pass
  x1 = x[:, 0]
  x2 = x[:, 1]
  x1_test = np.logical_or(x1 < -15.0, x1 > -5.0).any()
  x2_test = np.logical_or(x2 < -3.0, x2 > 3.0).any()
  if x1_test or x2_test:
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")

  summand = 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2))
  summand += 0.01 * np.abs(x1 - 10)
  return summand


def sphere(x):
  try:
    x = x.A
  except AttributeError:
    pass
  x_test = np.logical_or(x < -5.12, x > 5.12)
  if x_test.any():
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")
  z = np.square(x)
  fx = z.sum(axis=1)
  return fx


def camel_hump_6(x):
  try:
    x = x.A
  except AttributeError:
    pass
  x1 = x[:, 0]
  x2 = x[:, 1]
  term1 = (4.0 - 2.1 * x1 ** 2 + x1 ** 4 / 3.0)
  term2 = x1 * x2
  term3 = (-4.0 + 4.0 * x2 ** 2) * x2 ** 2
  return term1 + term2 + term3


def hartmann_6(x):
  test = np.logical_or(x < 0.0, x > 1.0)
  if test.any():
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")
  alpha = np.array([1.0, 1.2, 3.0, 3.2]).reshape((4,))
  A = np.array([[10.0, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14]])
  P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091, 381]])
  out = []
  for i in range(len(x)):
    row_mat = np.array([x[i, :] for _ in range(4)]).reshape((4, 6))
    inner = (A * (row_mat - P) ** 2.0).sum(axis=1)
    # print(inner)
    outer = alpha * np.exp(-inner)
    # print(outer)
    y = - outer.sum()
    out.append((float(y)))
  return np.array(out)


def michalewicz_10(x):
  m = 10
  out = []
  for i in range(len(x)):
    prod1 = np.sin(x[i, :])
    z = np.array(range(m))
    prod2a = np.sin(z * x[i, :] ** 2 / np.pi)
    prod2 = np.power(prod2a, 2 * m)
    y = -sum(prod1 * prod2)
    out.append(y)
  return np.array(out)


def forrester(x):
  try:
    x = x.A
  except AttributeError:
    pass
  if np.logical_or(0.0 > x, x > 1.0).any():
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")
  y = np.square(6 * x - 2) * np.sin(12 * x - 4)
  return y.flatten()


def colville(x):
  try:
    x = x.A
  except AttributeError:
    pass
  if np.logical_or(x < -10, x > 10).any():
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")
  out = []
  for i in range(len(x)):
    x1 = x[i, 0]
    x2 = x[i, 1]
    x3 = x[i, 2]
    x4 = x[i, 3]
    y = 100 * (x1 ** 2 - x2) ** 2
    y += (x1 - 1.0) ** 2
    y += (x3 - 1.0) ** 2
    y += 90 * (x3 ** 2 - x4) ** 2
    y += 10.1 * ((x2 - 1.0) ** 2 + (x4 - 1.0) ** 2)
    y += 19.8 * (x2 - 1.0) * (x4 - 1.0)
    out.append(float(y))
  return np.array(out)


def holder_table(x):
  try:
    x = x.A
  except AttributeError:
    pass
  if np.logical_or(x < -10, x > 10).any():
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")
  x1 = x[:, 0]
  x2 = x[:, 1]
  inner = np.sin(x1)
  inner *= np.cos(x2)
  inner *= np.exp(np.abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi))
  out = - np.abs(inner)
  return out


def rosenbrock(x):
  try:
    x = x.A
  except AttributeError:
    pass
  if np.logical_or(x < -10, x > 10).any():
    raise ValueError("Provided values of x not in bounds for this objective function. See documentation.")
  x1 = x[:, 0]
  x2 = x[:, 1]
  out = np.square(1 - x1)
  out += 100 * np.square(x2 - np.square(x1))
  return out.flatten()


function_info = {
  "goldstein_price": {
    "objective_function": goldstein_price,
    "bounding_box": [[-2.0, 2.0], [-2.0, 2.0]],
    "argmin": np.array([[0.0, -1.0]]),
    "fmin": 3.0,
    "percentile_5": 100.153513
  },
  "branin": {
    "objective_function": branin,
    "bounding_box": [[-5, 10], [0, 15]],
    "argmin": np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]),
    "fmin": 0.397887,
    "percentile_5": 3.064375
  },
  "gramacy_lee": {
    "objective_function": gramacy_lee,
    "bounding_box": [[0.5, 2.5]],
    "argmin": np.array([0.548563444114526]),
    "fmin": -0.869011134989500,
    "percentile_5": -0.526385
  },
  "modified_branin": {
    "objective_function": modified_branin,
    "bounding_box": [[-5, 10], [0, 15]],
    "argmin": np.array([[-3.68928444, 13.62998588]]),
    "fmin": 0.0,
    "percentile_5": 11.254351
  },
  "bukin_6": {
    "objective_function": bukin_6,
    "bounding_box": [[-15, -5], [-3, 3]],
    "argmin": np.array([-10, 1]),
    "fmin": 0.0,
    "percentile_5": 39.130709
  },
  "sphere": {
    # The sphere function can have arbitrary dimension; this is for dimension 6
    "objective_function": sphere,
    "bounding_box": [[-5.12, 5.12] for _ in range(6)],
    "argmin": [[0.0] for _ in range(6)],
    "fmin": 0.0,
    "percentile_5": 22.401199
  },
  "camel_hump_6": {
    "objective_function": camel_hump_6,
    "bounding_box": [[-3, 3], [-2, 2]],
    "argmin": [[0.0898, -0.7126], [-0.0898, 0.7126]],
    "fmin": -1.0316,
    "percentile_5": -0.395632
  },
  "hartmann_6": {
    "objective_function": hartmann_6,
    "bounding_box": [[0, 1] for _ in range(6)],
    "argmin": np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
    "fmin": -3.32237,
    "percentile_5": -1.071243
  },
  "forrester": {
    "objective_function": forrester,
    "bounding_box": [[0, 1]],
    "argmin": np.array([0.757248757841856]),
    "fmin": -6.02074,
    "percentile_5": -5.684173
  },
  "colville": {
    "objective_function": colville,
    "bounding_box": [[-10.0, 10.0] for _ in range(4)],
    "argmin": np.array(4 * [1]).reshape((1, 4)),
    "fmin": 0.0,
    "percentile_5": 6902.285759
  },
  "holder_table": {
    "objective_function": holder_table,
    "bounding_box": [[-10, 10] for _ in range(2)],
    "argmin": np.array([[8.05502, 9.66459], [8.05502, -9.66459], [-8.05502, 9.66459], [-8.05502, -9.66459]]),
    "fmin": -19.2085,
    "percentile_5": -8.552125
  },
  "rosenbrock": {
    "objective_function": rosenbrock,
    "bounding_box": [[-10.0, 10.0] for _ in range(2)],
    "argmin": np.array([[1, 1]]),
    "fmin": 0.0,
    "percentile_5": 245.991066
  }
}
