# Databricks notebook source
# MAGIC %pip install vllm==0.4.2 ray==2.22.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

setup_ray_cluster(
  max_worker_nodes=2,
  num_cpus_per_node=1,
  num_gpus_worker_node=1,
  collect_log_to_path="/dbfs/Users/sriharsha.jana@databricks.com/ray_collected_logs"
)

# COMMAND ----------

import ray
ray.init()
ray.cluster_resources()

# COMMAND ----------

# DBTITLE 1,Trying to run Vllm on multiple GPU
from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m", dtype="half", tensor_parallel_size=1)

# COMMAND ----------

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# COMMAND ----------

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shutdown the ray cluster

# COMMAND ----------

# To shutdown ray cluster after the test
shutdown_ray_cluster()
