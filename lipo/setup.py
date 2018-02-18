#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-02-17 (year-month-day)

from setuptools import setup

with open("README.md") as f:
  long_description = f.read()

setup(
  name='lipo',
  version='0.1',
  description='Global optimization of Lipschitz functions',
  license="BSD 3-clause",
  long_description=long_description,
  author='David J. Elkind',
  author_email='djelkind@gmail.com',
  url="http://www.github.com/sycor4x/lipo",
  packages=['foo'],  # same as name
  install_requires=['numpy==1.14.0'],
)
