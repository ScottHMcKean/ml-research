# Serving Endpoint Orchestrator

A minimal, DAB-deployable Databricks App that runs a **config-driven compute graph**
over Databricks model serving endpoints. Serving endpoints are one node type; the
graph is designed to hold other node types too (pure-python transforms, constants,
and whatever you add). Wiring a new backend or a new step is a `graph.yaml` edit, not
a code change.

## Contents

```
01_train_deploy_pyfunc.ipynb    RandomForest → custom mlflow.pyfunc → CPU-SMALL endpoint
02_train_deploy_xgboost.ipynb   XGBClassifier → mlflow.xgboost flavor → CPU-SMALL endpoint
app/
  app.py         FastAPI compute-graph engine (topological execution)
  graph.yaml     the graph spec: typed nodes + edges + named graphs
  app.yaml       Databricks App config
  requirements.txt
```

## The two models

Both train on `sklearn`'s breast-cancer dataset and deploy to a **CPU / SMALL**,
scale-to-zero serving endpoint on the shm-skunkworks FEVM. Dependencies are pinned to
the exact training-time versions (`pip_requirements`) so the serving container
reproduces training.

| Notebook            | Flavor          | UC model                            | Endpoint                     |
| ------------------- | --------------- | ----------------------------------- | ---------------------------- |
| `01_..._pyfunc`     | `mlflow.pyfunc` | `shm_catalog.shared.tree_pyfunc_model`  | `shm_tree_pyfunc_endpoint`   |
| `02_..._xgboost`    | `mlflow.xgboost`| `shm_catalog.shared.tree_xgboost_model` | `shm_tree_xgboost_endpoint`  |

> The pyfunc endpoint returns class-1 **probability**; the xgboost flavor's
> `/invocations` returns class **labels**. The `mean_predictions` ensemble averages
> whatever numeric predictions come back — it demonstrates graph wiring, not a
> calibrated ensemble. Swap in your own transform for real blending.

## The compute graph

`graph.yaml` declares **nodes** (each typed) and **graphs** (a named output node; the
app runs that node's ancestors in topological order).

```yaml
nodes:
  - {id: tree_pyfunc,  type: serving_endpoint, endpoint: shm_tree_pyfunc_endpoint,  path: /invocations}
  - {id: tree_xgboost, type: serving_endpoint, endpoint: shm_tree_xgboost_endpoint, path: /invocations}
  - {id: ensemble,     type: transform, fn: mean_predictions, inputs: [tree_pyfunc, tree_xgboost]}
graphs:
  ensemble_score: {output: ensemble}
```

- **Source nodes** (no `inputs`) receive the graph request body.
- A node with a single upstream forwards that upstream's output as its payload.
- A `transform` node receives `{upstream_id: output}` and runs a registered function.

**Node types** live in `NODE_TYPES` (`serving_endpoint`, `transform`, `constant`) and
**transforms** in `TRANSFORMS` (`mean_predictions`, `merge`) — add your own "other
nodes" by registering a handler.

## Deploy

The DAB resources (`resources/serving_orchestrator_*.yml`) are the source of truth. On
a CLI where `bundle deploy` is all-or-nothing (it deploys *every* resource in this
repo), deploy the pieces in isolation instead — this is exactly the path used to bring
up the live app:

```bash
cd ~/Repos/ml-research
P=fe-vm-shm-skunkworks
B=/Workspace/Users/$(whoami)@databricks.com/serving_orchestrator

# 1. Train + deploy both CPU-SMALL endpoints (one-off serverless runs).
databricks workspace import --file notebooks/serving_orchestrator/01_train_deploy_pyfunc.ipynb  "$B/01_train_deploy_pyfunc"  --format JUPYTER --overwrite -p $P
databricks workspace import --file notebooks/serving_orchestrator/02_train_deploy_xgboost.ipynb "$B/02_train_deploy_xgboost" --format JUPYTER --overwrite -p $P
databricks jobs submit --json '{"run_name":"orch-pyfunc","tasks":[{"task_key":"t","notebook_task":{"notebook_path":"'$B'/01_train_deploy_pyfunc"}}]}' -p $P
databricks jobs submit --json '{"run_name":"orch-xgboost","tasks":[{"task_key":"t","notebook_task":{"notebook_path":"'$B'/02_train_deploy_xgboost"}}]}' -p $P

# 2. Create the app, sync source, grant its service principal CAN_QUERY on both
#    endpoints (by endpoint ID, not name), then deploy.
databricks apps create serving-orchestrator -p $P
databricks sync notebooks/serving_orchestrator/app "$B/app" --full -p $P
databricks apps deploy serving-orchestrator --source-code-path "$B/app" -p $P
```

Or, on a workspace where a repo-wide `databricks bundle deploy -t <target>` is fine, the
bundle wires all of it (jobs + app + CAN_QUERY resources) in one shot.

## Live app

`https://serving-orchestrator-7474644262257186.aws.databricksapps.com`

## Try it

```bash
TOKEN=$(databricks auth token -p fe-vm-shm-skunkworks | jq -r .access_token)
curl -s "$APP_URL/graph/ensemble_score" -H "Authorization: Bearer $TOKEN" \
     -H 'content-type: application/json' \
     -d '{"dataframe_split":{"columns":["mean radius", ...30 cols...],"data":[[...]]}}'
# -> {"result":{"predictions":[0.323]},"nodes":{...per-node outputs...}}
```
