# Databricks notebook source
# MAGIC %pip install ray==2.22.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

setup_ray_cluster(
  num_worker_nodes=2,
  num_cpus_worker_node=4,
  num_gpus_per_node=1,
  collect_log_to_path="/dbfs/Users/sriharsha.jana@databricks.com/ray_collected_logs"
)

# COMMAND ----------

import ray
ray.init()
ray.cluster_resources()

# COMMAND ----------

import random
import time
from fractions import Fraction

@ray.remote
def pi4_sample(sample_count):
    """pi4_sample runs sample_count experiments, and returns the
    fraction of time it was inside the circle.
    """
    in_count = 0
    for i in range(sample_count):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1:
            in_count += 1
    return Fraction(in_count, sample_count)

SAMPLE_COUNT = 1000 * 1000
start = time.time()
future = pi4_sample.remote(sample_count=SAMPLE_COUNT)
pi4 = ray.get(future)
end = time.time()
dur = end - start
print(f'Running {SAMPLE_COUNT} tests took {dur} seconds')

pi = pi4 * 4
print(float(pi))

# COMMAND ----------

FULL_SAMPLE_COUNT = 2000 * 1000 * 1000
BATCHES = int(FULL_SAMPLE_COUNT / SAMPLE_COUNT)
print(f'Doing {BATCHES} batches')
results = []
for _ in range(BATCHES):
    results.append(pi4_sample.remote(sample_count = SAMPLE_COUNT))
output = ray.get(results)
 
pi = sum(output)*4/len(output)
print(float(pi))

# COMMAND ----------

shutdown_ray_cluster()
