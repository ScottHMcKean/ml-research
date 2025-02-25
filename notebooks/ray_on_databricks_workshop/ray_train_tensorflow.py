# Databricks notebook source
# MAGIC %pip install -U pyarrow "ray[default]>=2.3.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_name = f"/Users/{username}/mnist-tf-experiment"

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

exp_obj = mlflow.set_experiment(experiment_name)

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES
setup_ray_cluster(
  num_worker_nodes=2,
  num_cpus_per_node=4,
  collect_log_to_path="/dbfs/Users/scott.mckean@databricks.com/ray_collected_logs"
)

# COMMAND ----------

import ray
ray.init()
ray.cluster_resources()

# COMMAND ----------

import json
import os
import mlflow
import tempfile
import numpy as np
import tensorflow as tf

from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train import Result, RunConfig, ScalingConfig, Checkpoint
from ray.train.tensorflow import TensorflowTrainer
from ray import train

# COMMAND ----------

def mnist_dataset(batch_size: int) -> tf.data.Dataset:
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    # The `x` arrays are in uint8 and have values in the [0, 255] range.
    # You need to convert them to float32 with values in the [0, 1] range.
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(60000)
        .repeat()
        .batch(batch_size)
    )
    return train_dataset

# COMMAND ----------

def build_cnn_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28)),
            tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    return model

# COMMAND ----------

def train_func(config: dict):
    per_worker_batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 3)
    steps_per_epoch = config.get("steps_per_epoch", 70)

    tf_config = json.loads(os.environ["TF_CONFIG"])
    num_workers = len(tf_config["cluster"]["worker"])

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_cnn_model()
        learning_rate = config.get("lr", 0.001)
        multi_worker_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=["accuracy"],
        )
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=config["checkpoint_path"],
                                                    save_weights_only=True,
                                                    verbose=1)
    history = multi_worker_model.fit(
        multi_worker_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[cp_callback]
    )
    
    # temp_checkpoint_dir = config["checkpoint_path"]
    # multi_worker_model.save(os.path.join(temp_checkpoint_dir, "model_keras/"))
    results = history.history
    return results

# COMMAND ----------

num_workers :int=2
epochs :int=1
use_gpu :bool=False
storage_path :str= "/Volumes/uc_demos_sriharsha_jana/test_db/shjdata/tf_ray_mnist"

config = {"lr": 1e-3, 
          "batch_size": 64, 
          "epochs": epochs, 
          "checkpoint_path": storage_path}

trainer = TensorflowTrainer(
    train_loop_per_worker=train_func,
    train_loop_config=config,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    run_config=RunConfig(storage_path=storage_path)
)

# COMMAND ----------

results = trainer.fit()

# COMMAND ----------

checkpoint_details = os.path.join(storage_path, "model.keras")
model = tf.keras.models.load_model(checkpoint_details)
model.summary()

# COMMAND ----------

shutdown_ray_cluster()
