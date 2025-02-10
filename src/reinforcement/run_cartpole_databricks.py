# Databricks notebook source
# MAGIC %md
# MAGIC This notebook contains a minimal example of training a Deep Q-Network (DQN) agent on the CartPole environment using OpenAI's Gymnasium. The implementation works both locally and on Databricks.

# COMMAND ----------

# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

# MAGIC %sh
# MAGIC python cartpole_dqn.py
