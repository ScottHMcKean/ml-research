# Databricks notebook source
# MAGIC %pip install kedro[spark.SparkDataSet]~=0.1

# COMMAND ----------

import yaml
with open('./config.yaml') as file:
    ingest_dict = yaml.load(file, Loader=yaml.FullLoader)

# COMMAND ----------

# Import necessary libraries
from kedro.pipeline import node, pipeline
from kedro.io import MemoryDataset
from pyspark.sql.functions import col
from kedro.io import DataCatalog
from kedro.runner import ThreadRunner

# Define nodes
def create_spark_df():
    table_name = ingest_dict["parts"]["filepath"]
    return spark.table(table_name)
    
def process_df(df):
    return df.withColumn("height_plus_10", col("height") + 10)

def show_results(df):
    df.show()
    return None

# Create pipeline
def create_pipeline():
    return pipeline([
        node(func=create_spark_df, inputs=None, outputs="initial_df", name="create_df"),
        node(func=process_df, inputs="initial_df", outputs="processed_df", name="process_df"),
        node(func=show_results, inputs="processed_df", outputs=None, name="show_results")
    ])

# Set up the catalog
catalog = DataCatalog({
    "initial_df": MemoryDataset(copy_mode="assign"),
    "processed_df": MemoryDataset(copy_mode="assign")
})

# Run the pipeline
runner = ThreadRunner()
runner.run(create_pipeline(), catalog)

# COMMAND ----------

# Set up the catalog
catalog = DataCatalog({
    "initial_df": df,
    "processed_df": MemoryDataset(copy_mode="assign")
})

# Run the pipeline
runner = ThreadRunner()
runner.run(create_pipeline(), catalog)
