#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-03-24 (year-month-day)

"""
"""

from __future__ import division

import seidhr.common as common
from seidhr.optimization_test_functions import function_info
from seidhr.lipo import AdaLIPO

if __name__ == "__main__":
  args = common.parse_args()
  active_fn = function_info[args.function_name]

  # grab initial values at random
  prs = common.PureRandomSearch(**active_fn, initialization_data=4)
  x_init = prs.x
  y_init = prs.y

  optimizer = AdaLIPO(**active_fn,
                      minimize=True,
                      initialization_data={"x": x_init, "y": y_init})
  x_best, f_x_best = optimizer.fit(niter=60)
  x_star = active_fn["argmin"]
  f_x_star = active_fn["fmin"]

  print("x* = %s, y* = %.6f" % (x_star, f_x_star))
  print(u"x\u0302  = %s, y\u0302  = %.6f" % (x_best, f_x_best))
