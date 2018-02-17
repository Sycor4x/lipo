#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (year-month-day)

"""
"""

from __future__ import division

import pytest

from ..optimization_test_functions import gramacy_lee_2012, goldstein_price, branin


gramacy_lee_2012()


class NumericTest(unittest.TestCase):
  pass


class GramacyLee2012Test(NumericTest):
  pass


class GoldsteinPriceTest(NumericTest):
  pass


class BraninTest(NumericTest):
  pass


