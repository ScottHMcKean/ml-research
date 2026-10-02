# NVIDIA cuOpt diet-optimization LP on Serverless GPU

A faithful port of NVIDIA's [`diet_optimization_lp.ipynb`](https://github.com/NVIDIA/cuopt-examples/blob/main/diet_optimization/diet_optimization_lp.ipynb)
to a Databricks notebook, used as the reproducible **cuOpt test vehicle** for the Kinaxis
"AI Runtime" POC. The goal is to prove that an NVIDIA cuOpt (PDLP, GPU-accelerated linear
programming) workload runs cleanly on Databricks **Serverless GPU**, and to capture spin-up +
solve timing. The LP model itself (USDA nutrition guidelines, food costs, solve, solution
print, sensitivity analysis) is unchanged from the source — the point is the runtime/plumbing,
not the optimization.

## What it is

The classic diet problem: pick servings of 9 foods to meet USDA nutritional bounds (calories,
protein, fat, sodium) at minimum cost. Built with the cuOpt Python API (`Problem`,
`addVariable`, `LinearExpression`, `setObjective`, `addConstraint`, `SolverSettings`,
`problem.solve()`), then reads duals / reduced costs for a sensitivity read.

## How to run it on Serverless GPU

1. Open `diet_optimization_lp.py` as a notebook in the Databricks workspace.
2. Attach it to **Serverless GPU** compute. An **A10** is more than enough for this toy LP.
3. In the notebook's **Environment** panel (right sidebar), select serverless
   **environment version 5 or 6** — the AI-runtime line that ships the CUDA 12 / torch 2.9
   stack cuOpt builds against.
4. Run all cells top to bottom. The first code cell `%pip install`s cuOpt from NVIDIA's index
   and the next cell `%restart_python` to pick up the fresh wheels.

### CUDA 12 vs CUDA 13

The notebook defaults to the **CUDA 12** build (`cuopt-cu12`), which matches the Kinaxis stack
(CUDA 12.7 wheel; the Serverless GPU AI runtime ships torch 2.9 / CUDA 12.9). To target a
CUDA 13 runtime, swap the install line for its `-cu13` equivalents:

```
%pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu13 nvidia-nvjitlink-cu13 rapids-logger==0.1.19
```

## Expected output

- A GPU name/memory line from `nvidia-smi` (confirms you're on a GPU).
- The nutritional-values DataFrame, then the model build (9 variables, 7 constraints).
- An **optimal** solution — total cost around **$3** with a handful of foods at nonzero
  servings — followed by the per-category nutritional-intake check.
- A sensitivity table: constraint **dual values** + **slack**, and per-food **reduced costs**.

## Timing notes

Kinaxis cares about **cold-start vs. solve** as separate numbers, so the notebook prints both:

- `Notebook start -> GPU ready` — wall clock from the first post-restart cell to the
  `nvidia-smi` check completing.
- `Solve` — the cuOpt `problem.solve()` call alone (the source already timed this).
- `Notebook start -> solve done` — the two combined.

Caveat: `%restart_python` runs the rest of the notebook in a fresh kernel, so the
`%pip install` time is a **separate kernel session** and is *not* included in the
`NOTEBOOK_START` checkpoint. The reported "cold start" is notebook-attach → GPU-ready →
library import, which is the meaningful serverless-GPU acquisition window; the pip-install
duration should be read from the install cell's own output if you need it.

## Source fidelity

Model logic, data, solve, `print_solution()`, and the sensitivity-analysis section are
preserved verbatim from the NVIDIA source. The only changes are Databricks/serverless-GPU
plumbing: the Colab/Docker HTML `check_gpu()` is replaced with the lean `nvidia-smi` subprocess
check, the commented Colab/Docker pip block becomes an active `%pip install` + `%restart_python`,
and a wall-clock timing split was added. The NVIDIA SPDX/Apache-2.0 copyright cell is kept at
the end.
