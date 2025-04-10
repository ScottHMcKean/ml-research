# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from graphframes import GraphFrame
from graphframes.lib import AggregateMessages as AM

from warnings import filterwarnings

filterwarnings('ignore', 'DataFrame.sql_ctx is an internal property')
filterwarnings('ignore', 'DataFrame constructor is internal')

# COMMAND ----------

vertices_list = [
    {"id": 1, "type": "physical", "reading": 2},
    {"id": 2, "type": "physical", "reading": 3},
    {"id": 3, "type": "physical", "reading": 10},
    {"id": 4, "type": "physical", "reading": 10},
    {"id": 5, "type": "virtual", "reading": None},
    {"id": 6, "type": "virtual", "reading": None},
    {"id": 7, "type": "virtual", "reading": None},
    {"id": 8, "type": "virtual", "reading": None},
]

edges_list = [
    {"src": 1, "dst": 5},
    {"src": 2, "dst": 6},
    {"src": 3, "dst": 6},
    {"src": 4, "dst": 7},
    {"src": 6, "dst": 8},
    {"src": 7, "dst": 8},
]

vertices = spark.createDataFrame(vertices_list)
edges = spark.createDataFrame(edges_list)
graph = GraphFrame(vertices, edges).cache()

# COMMAND ----------

def compute_sums(graph: GraphFrame, max_iter: int) -> DataFrame:
    
    # initialize working vertices
    v = (
        graph.vertices
        .withColumn('total', F.coalesce(F.col('reading'), F.lit(0)))
        .withColumn('increment', F.col('total')))
    g = GraphFrame(AM.getCachedDataFrame(v), graph.edges)
    
    # expand the increment fringe one layer at a time
    for i in range(max_iter):
        # send and aggregate messages telling each parent of the increments
        agg = (
            g.aggregateMessages(
                F.sum(AM.msg).alias('increment'), 
                sendToDst=AM.src['increment'])
            .filter(F.col('increment') > 0))
        
        # exit early if there are no more increments to communicate
        if agg.isEmpty():
            break

        # incorporate the aggregate increments for this layer and 
        # propogate the received increment to the next layer
        v = AM.getCachedDataFrame(
            g.vertices.join(agg, on='id', how='left_outer')
            .withColumn('total', F.when(
                agg.increment.isNull(), 
                g.vertices.total).otherwise(
                    agg.increment + g.vertices.total))
            .withColumn('new_increment', agg.increment)
            .drop('increment').withColumnRenamed('new_increment', 'increment'))
        g = GraphFrame(v, g.edges)
    
    # return the final vertices and their totals
    return g.vertices.drop('increment')


# COMMAND ----------

display(compute_sums(graph, 5))
