#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (year-month-day)

"""
"""

from __future__ import division

from numpy import isclose

from ..lipo.optimization_test_functions import goldstein_price, gramacy_lee_2012

foo = gramacy_lee_2012(1.0)


def test_gramacy_lee_2012_2():
  assert isclose(gramacy_lee_2012(2.0), 1.0)


def test_gramacy_lee_2012_1p5():
  assert isclose(gramacy_lee_2012(1.5), 0.0625)


def test_goldstein_price():
  assert isclose(goldstein_price((-1, 1)), 87100)
