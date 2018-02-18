#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (year-month-day)

"""
"""

from __future__ import division

from numpy import isclose

from lipo.optimization_test_functions import goldstein_price, gramacy_lee_2012, branin

foo = gramacy_lee_2012(1.0)


def test_gramacy_lee_2012_1():
  assert isclose(gramacy_lee_2012(2.0), 1.0)


def test_gramacy_lee_2012_2():
  assert isclose(gramacy_lee_2012(1.5), 0.0625)


def test_goldstein_price_1():
  assert isclose(goldstein_price((-1, 1)), 87100)


def test_goldstein_price_2():
  assert isclose(goldstein_price((1, 1)), 1876)


def test_goldstein_price_3():
  assert isclose(goldstein_price((1, -1)), 7100)


def test_branin_1():
  assert isclose(branin((1, 1)), 27.70291)


def test_branin_2():
  assert isclose(branin((-3, 5)), 48.62023)


def test_branin_3():
  assert isclose(branin((-2.5, 7.5)), 13.10694)
