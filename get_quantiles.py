#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-03-26 (year-month-day)

"""
"""

from __future__ import division

import argparse

import numpy as np

from seidhr.common import PureRandomSearch
from seidhr.optimization_test_functions import function_info


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
  parser.add_argument("-n", "--niter", type=int, default=100000,
                      help="budget for function evaluations")
  parser.add_argument("-p", "--percentile", type=float, default=5.0,
                      help="float in (0.0, 100.0) - the target percentile of the function")
  args_out = parser.parse_args()
  return args_out


if __name__ == "__main__":
  args = parse_args()
  active_function = function_info[args.function_name]
  prs = PureRandomSearch(**active_function, initialization_data=1)
  prs.fit(niter=args.niter)

  target = np.percentile(prs.y, q=args.percentile)
  print("Function `%s` has %.2f percentile equal to %.6f" % (args.function_name, args.percentile, target))
