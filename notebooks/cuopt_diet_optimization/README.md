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
4. Run all cells top to bottom. For an **interactive** run that is *not* using the job's
   pre-baked environment, add cuOpt yourself first — paste the two commented fallback lines
   (near the top of the notebook) each into their own cell: a `%pip install` from NVIDIA's
   index, then `%restart_python`.

The faster, repeatable path is the **sweep job** below, which pre-bakes cuOpt into the
serverless environment so nothing installs per run.

## Parameterized price-shock sweep (the job)

`resources/cuopt_diet_jobs.yml` declares the `cuopt_diet_optimization` job, which fans this
notebook out over price-shock scenarios with a `for_each` task — all on the **same Serverless
GPU environment**, run in parallel (`concurrency: 3`).

- **Pre-baked environment.** cuOpt is declared in the job's `environments` block
  (`environment_version: "5"`, `cuopt-cu12` + friends from `pypi.nvidia.com`), not
  `%pip install`ed in the notebook. Serverless **caches and reuses** that environment across
  runs and iterations, so the ~630 MB install happens once (on the first cold env build)
  instead of every run — this is what removes the install-dominated cold start.
- **Per-iteration compute.** Each nested task carries
  `compute.hardware_accelerator: GPU_1xA10` + `environment_key: gpu_env`.
- **Parameters.** The notebook reads two widgets, set per iteration by the job:
  - `scenario_name` — a named scenario from `PRICE_SCENARIOS` (`baseline`, `dairy_shock`,
    `meat_inflation`, `carb_discount`, `broad_shock`).
  - `price_multipliers` — optional JSON `{food: multiplier}` that overrides the named scenario.
- **Result.** Each iteration ends with `dbutils.notebook.exit(...)` returning a compact JSON
  (`scenario`, `status`, `objective`, `plan`, and the timing split), so every scenario's
  outcome is visible in the run output.

Only the per-serving **costs** change between scenarios; the LP model (variables, constraints,
objective) is identical, so cuOpt simply re-solves for the cheapest diet under each cost regime.

### CUDA 12 vs CUDA 13

The job defaults to the **CUDA 12** build (`cuopt-cu12`), which matches the Kinaxis stack
(CUDA 12.7 wheel; the Serverless GPU AI runtime ships torch 2.9 / CUDA 12.9). To target a
CUDA 13 runtime, swap the environment dependencies for their `-cu13` equivalents
(`cuopt-cu13`, `nvidia-nvjitlink-cu13`).

## Expected output

- A GPU name/memory line from `nvidia-smi` (confirms you're on a GPU).
- The nutritional-values DataFrame, then the model build (9 variables, 7 constraints).
- An **optimal** solution — total cost around **$3** with a handful of foods at nonzero
  servings — followed by the per-category nutritional-intake check.
- A sensitivity table: constraint **dual values** + **slack**, and per-food **reduced costs**.

## Timing notes

Kinaxis cares about **cold-start vs. solve** as separate numbers, so the notebook prints both:

- `Notebook start -> GPU ready` — wall clock from the first executable cell to the
  `nvidia-smi` check completing.
- `Solve` — the cuOpt `problem.solve()` call alone (the source already timed this).
- `Notebook start -> solve done` — the two combined.

With the **pre-baked environment**, there is no in-notebook `%pip install`, so `NOTEBOOK_START`
reflects true notebook start. Dependency install happens during serverless **environment init**
(before the notebook runs) and is paid once per cold environment build, then amortized across
cached reuse — so the per-run numbers above isolate GPU-ready + solve, which is the metric that
matters once the environment is warm.

### Verified result (fevm-shm-skunkworks)

Single-A10 run (`GPU_1xA10`, env v5), baseline scenario: GPU `NVIDIA A10G, 23028 MiB`,
CUDA 12.9, cuOpt 25.10.1. Solved to **Optimal**, objective **$11.83** (hamburger 0.605, milk
6.970, ice cream 2.591 servings), all four nutrition limits satisfied. Timing: GPU-ready
0.03 s, cuOpt solve 0.16 s. The *first* run paid ~3 min of wheel install (~630 MB) as a plain
`%pip install`; pre-baking into the cached env is what this sweep job removes from per-run cost.

## Source fidelity

The LP model (variables, constraints, objective), nutrition data, solve, `print_solution()`,
and the sensitivity-analysis section are preserved from the NVIDIA source. The changes are
Databricks/serverless-GPU plumbing plus the price-shock parameterization: the Colab/Docker HTML
`check_gpu()` is replaced with the lean `nvidia-smi` subprocess check; cuOpt moves from an
in-notebook pip install to the **pre-baked job environment** (with the pip install kept as an
inactive, commented fallback for interactive runs); `scenario_name` / `price_multipliers`
widgets scale the per-serving **costs only**; a wall-clock timing split and a
`dbutils.notebook.exit(...)` JSON result were added. The NVIDIA SPDX/Apache-2.0 copyright cell
is kept at the end.
