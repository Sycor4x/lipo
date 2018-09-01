#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2018-03-07 (year-month-day)

"""
Benchmarking of optimization functions
"""

import argparse
import time

import numpy as np

import seidhr.common as common
from seidhr.lipo import AdaLIPO
from seidhr.optimization_test_functions import function_info

P_EXPLORE_LOW = 0.025
P_EXPLORE_HIGH = 0.975


class OptimizerBenchmark:
  def __init__(self,
               dict_of_test_fns,
               dict_of_optimizers,
               target_value="percentile_5",
               n_experiments=256,
               max_objective_fn_calls=100,
               monte_carlo_size=10000,
               random_pr_explore=False,
               verbose=False):
    # validate inputs
    if len(dict_of_test_fns) < 1 or len(dict_of_optimizers) < 1:
      raise ValueError("supplied dictionaries must be non-empty")

    if not isinstance(monte_carlo_size, int):
      raise ValueError("monte_carlo_size must be positive int")
    if monte_carlo_size < 1:
      raise ValueError("monte_carlo_size must be positive int")

    if not isinstance(max_objective_fn_calls, int):
      raise ValueError("max_objective_fn_calls must be positive int")
    if max_objective_fn_calls < 1:
      raise ValueError("max_objective_fn_calls must be positive int")

    if not isinstance(n_experiments, int):
      raise ValueError("n_experiments must be positive int")
    if n_experiments < 1:
      raise ValueError("n_experiments must be positive int")

    # store args
    self._dict_of_test_fns = dict_of_test_fns
    self._dict_of_optimizers = dict_of_optimizers
    self._max_objective_fn_calls = max_objective_fn_calls
    self._n_experiments = n_experiments
    self._target_value = target_value
    self._monte_carlo_size = monte_carlo_size
    self._random_pr_explore = random_pr_explore
    self._verbose = verbose

  @property
  def random_pr_explore(self):
    return self._random_pr_explore

  @property
  def dict_of_optimizers(self):
    return self._dict_of_optimizers

  @property
  def dict_of_test_fns(self):
    return self._dict_of_test_fns

  @property
  def max_objective_fn_calls(self):
    return self._max_objective_fn_calls

  @property
  def monte_carlo_size(self):
    return self._monte_carlo_size

  @property
  def n_experiments(self):
    return self._n_experiments

  @property
  def target_value(self):
    return self._target_value

  @property
  def verbose(self):
    return self._verbose

  def get_target_y(self, objective_function_name):
    y_target = self.dict_of_test_fns[objective_function_name][self.target_value]
    return float(y_target)

  def get_init_data(self, objective_function_name, size):
    active_fn = self.dict_of_test_fns[objective_function_name]
    prs = common.PureRandomSearch(**active_fn, initialization_data=size)
    y_target = self.get_target_y(objective_function_name=objective_function_name)
    if self.verbose:
      msg = "This is the target value: %.2f; the algorithm is initialized with %d (x, f(x)) tuples." % (y_target, size)
      print(msg)
    init_data = {
      "x": prs.x,
      "y": prs.y
    }
    return y_target, init_data

  def single_test(self, initial_data, objective_function_name, optimizer_name):
    active_fn = self.dict_of_test_fns[objective_function_name]
    x_star = active_fn["argmin"]

    y_target, init_xy = initial_data
    if self.random_pr_explore:
      pr_explore = np.random.uniform(low=P_EXPLORE_LOW, high=P_EXPLORE_HIGH, size=1)[0]
    else:
      pr_explore = 0.5
    optim = self.dict_of_optimizers[optimizer_name](**active_fn,
                                                    pr_explore=pr_explore,
                                                    initialization_data=init_xy,
                                                    verbose=self.verbose)
    x_best, y_best, n_objective_calls = optim.fit_benchmark(target=y_target,
                                                            max_obj_fn_evals=self.max_objective_fn_calls)
    x_star_norm = np.linalg.norm(x_star - x_best, 2)
    n_objective_calls = optim.y.size
    return x_star_norm, y_best, n_objective_calls, pr_explore

  def test_all(self):
    results = dict()
    for fn_name in self.dict_of_test_fns.keys():
      fn_results = results.get(fn_name, dict())
      for optim_name in self.dict_of_optimizers.keys():
        if self.verbose:
          print(optim_name)
        optim_results = fn_results.get(optim_name, dict())
        for _ in range(self.n_experiments):
          initial_data = self.get_init_data(objective_function_name=fn_name, size=1)
          x_star_norm, y_best, n_objective_calls, pr_explore = self.single_test(objective_function_name=fn_name,
                                                                                optimizer_name=optim_name,
                                                                                initial_data=initial_data)

          n_elements = 4
          old_row = optim_results.get("result_array", np.zeros((0, n_elements)))
          new_row = np.array([pr_explore, n_objective_calls, x_star_norm, y_best]).reshape((1, n_elements))
          all_row = np.vstack((old_row, new_row))
          optim_results.update({"result_array": all_row})
          print("%d objective calls" % n_objective_calls)
        fn_results.update({optim_name: optim_results})
      results.update({fn_name: fn_results})
    return results


class Range(object):
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def __eq__(self, other):
    return self.start <= other <= self.end


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
  parser.add_argument("-n", "--n_experiments",
                      type=int,
                      default=128,
                      help="How many times to run each optimizer for each objective function")
  parser.add_argument("-p", "--percentile",
                      type=float,
                      default=5.0,
                      choices=[Range(0.0, 50.0)],
                      help="the target quantile of the objective function; capped at 50.0.")
  args_out = parser.parse_args()
  return args_out


def subset_fns(list_of_str):
  out = dict()
  for fn_name in list_of_str:
    out.update({fn_name: function_info[fn_name]})
  return out


def flatten_dict(nested_dict):
  out = ["obj_fn\toptimizer\tpr_explore\tn_obj_fn_calls\tx_star_norm\ty_best\n"]
  for k1, v1 in nested_dict.items():
    for k2, v2 in v1.items():
      row = "%s\t%s\t" % (k1, k2)
      for v3 in v2.values():
        for v in v3:
          row_data = "\t".join("%.6f" % fl for fl in v.tolist())
          new_row = row + row_data + "\n"
          out.append(new_row)
  return out


if __name__ == "__main__":
  start = time.time()
  args = parse_args()
  dict_of_optim = {
    "PRS": common.PureRandomSearch,
    # "AdaLIPO": RandomAdaLIPO,
    "AdaLIPO": AdaLIPO
  }
  dict_of_fn = subset_fns([args.function_name])
  prs_sample_size = common.pure_random_search_sample_size(q=1.0 - args.percentile / 100.0)
  max_obj_fn_calls = 4 * prs_sample_size
  print("This is the maximum number of objective function calls for each experiment: %d" % max_obj_fn_calls)
  bench = OptimizerBenchmark(dict_of_optimizers=dict_of_optim,
                             dict_of_test_fns=dict_of_fn,
                             n_experiments=args.n_experiments,
                             max_objective_fn_calls=max_obj_fn_calls,
                             verbose=False)
  benchmark_results = bench.test_all()

  flattened_results = flatten_dict(benchmark_results)
  for item in flattened_results:
    print(item)

  # with open("%s_n=%d_results.json" % (args.function_name, args.n_experiments), "w") as out_file:
  #   json.dump(obj=benchmark_results, fp=out_file)
  with open("%s_n=%d_results_random_pr_explore.tsv" % (args.function_name, args.n_experiments), "w") as out_file:
    out_file.writelines(flattened_results)

  stop = time.time()
  print("Elapsed time: %.2f hours" % ((stop - start) / 60 / 60))
