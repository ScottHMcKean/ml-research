# Databricks notebook source
# MAGIC %pip install lightning deltalake ray[default]==2.22.0

# COMMAND ----------

# MAGIC %pip install  git+https://github.com/delta-incubator/deltatorch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import lightning.pytorch as pl
from lightning import Trainer

from lightning.pytorch import loggers
from lightning.pytorch.callbacks import DeviceStatsMonitor, RichModelSummary, ModelCheckpoint

from torchvision import transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from deltalake import DeltaTable
import pyarrow.dataset as ds
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import mlflow
import os

# COMMAND ----------

import ray

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_name = f"/Users/{username}/flowers-embedding"
log_path = f"/dbfs/Users/{username}/flowers_pl_train_logger"
tensroboard_path = f"/dbfs/Users/{username}/flowers_pl_tensorboard"

db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

exp_obj = mlflow.set_experiment(experiment_name)

# COMMAND ----------

flower_dataset = spark.read.format('delta').load('/databricks-datasets/flowers/delta/')
print(flower_dataset.count())

# COMMAND ----------

display(flower_dataset)

# COMMAND ----------

label_dict = dict()
for i, img_label in enumerate(flower_dataset.select('label').distinct().collect()):
  label_dict[img_label['label']] = i
  
label_dict, len(label_dict)

# COMMAND ----------

class FlowerItrDataSet(IterableDataset):

  def __init__(self, label_dict: dict, delta_path:str, batch_size:int=128):
    super().__init__()
    self.label_dict = label_dict
    dt = DeltaTable(delta_path)
    self.delta_dataset = dt.to_pyarrow_dataset()
    self.batch_size = batch_size

  def __iter__(self):
    for batch in self.delta_dataset.to_batches(columns=['content', 'label'], batch_size=self.batch_size):
      batch_df = batch.to_pandas()
      for i, each_item in enumerate(batch_df['content'].values):
        img = Image.open(BytesIO(each_item))
        img = img.convert('RGB')
        img_resized = img.resize((128, 128))
        x_input = torch.tensor(np.moveaxis(np.array(img_resized, dtype=np.uint8), -1, 0))
        lbl = self.label_dict[batch_df.label.values[i]]
        y_input = F.one_hot(torch.tensor(lbl), num_classes=len(self.label_dict))
        yield {'x_input':x_input, 'y_label': y_input}

# COMMAND ----------

ds = FlowerItrDataSet(label_dict, delta_path="/dbfs/databricks-datasets/flowers/delta/", batch_size=4)
sample_data = list(DataLoader(ds, num_workers=2))
print(sample_data[0]['y_label'].view(-1).to(torch.long))
print(sample_data[0]['x_input'].shape)

# COMMAND ----------

class FlowerDatasetPl(pl.LightningDataModule):
    def __init__(self, label_dict:dict, data_dir: str, batch_size: int):
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=['label_dict'])
        self.label_dict = label_dict
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
       if stage=="fit" or stage is None:
            self.train_set = FlowerItrDataSet(self.label_dict, self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

# COMMAND ----------

class FlowerModel(nn.Module):
    def __init__(self):
        super(FlowerModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(246016, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.conv1(x.to(torch.float))
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output
      
class FlowerModelPL(pl.LightningModule):
    def __init__(self, learning_rate=1e-2):
        super().__init__()
        self.lr = learning_rate
        self.save_hyperparameters(logger=True)
        self.model = FlowerModel()
        self.criterion = nn.CrossEntropyLoss()
    
    def training_step(self, batch, batch_idx):
        x = self.model(batch['x_input'])
        labels = batch['y_label'].float()
        loss = self.criterion(x, labels)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# COMMAND ----------

def main_training_loop():
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import lightning.pytorch as pl
  from lightning import Trainer
  from deltalake import DeltaTable
  import pyarrow.dataset as ds
  from PIL import Image
  from io import BytesIO
  import numpy as np
  import pandas as pd
  import mlflow
  import os
  import ray
  from ray.train.lightning import (
    RayDDPStrategy, RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
  )

  # We need to do this so that different processes that will be able to find mlflow
  os.environ['DATABRICKS_HOST'] = db_host
  os.environ['DATABRICKS_TOKEN'] = db_token

  mlf_logger = loggers.MLFlowLogger(experiment_name=experiment_name, log_model=True)

  device_stats = DeviceStatsMonitor(cpu_stats=False)
  # rich_model_summary = RichModelSummary()

  epochs = 1
  train_batch_size = 32

  flower_module = FlowerModelPL()

  data_module = FlowerDatasetPl(label_dict=label_dict, 
                                data_dir="/dbfs/databricks-datasets/flowers/delta/",
                                batch_size=train_batch_size)

  trainer = Trainer(accelerator="auto",
                    devices="auto",
                    num_nodes=1, 
                    strategy=RayDDPStrategy(),
                    plugins=[RayLightningEnvironment()],
                    default_root_dir=log_path,
                    logger=[mlf_logger],
                    # callbacks=[device_stats, rich_model_summary],
                    max_epochs=epochs,
                    log_every_n_steps=1)
  
  trainer = prepare_trainer(trainer)
  trainer.fit(model=flower_module, datamodule=data_module)

  mlf_logger.experiment.log_artifact(
    run_id=mlf_logger.run_id,
    local_path=trainer.checkpoint_callback.best_model_path)
  
  return trainer.checkpoint_callback.best_model_path

# COMMAND ----------

from ray.train import ScalingConfig

scaling_config = ScalingConfig(num_workers=2, use_gpu=False)

# COMMAND ----------

from ray.train import RunConfig

# Local path (/some/local/path/unique_run_name)
run_config = RunConfig(storage_path="/dbfs/Users/sriharsha.jana@databricks.com/ray-torch-lightning", 
                       name="flower_train_run_two")

# COMMAND ----------

from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    main_training_loop,
    scaling_config=scaling_config,
    run_config=run_config,
)

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster, MAX_NUM_WORKER_NODES

setup_ray_cluster(
  num_worker_nodes=2,
  num_cpus_per_node=4,
  num_gpu_per_node=1,
  collect_log_to_path="/dbfs/Users/sriharsha.jana@databricks.com/ray_collected_logs"
)
ray.init()
ray.cluster_resources()

# COMMAND ----------

result = trainer.fit()

# COMMAND ----------

result = trainer.fit()

# COMMAND ----------

result

# COMMAND ----------

shutdown_ray_cluster()
