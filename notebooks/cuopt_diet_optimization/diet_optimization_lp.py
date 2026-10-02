# Databricks notebook source

# MAGIC %md
# MAGIC # Diet Optimization with cuOpt Python API
# MAGIC
# MAGIC **Run this notebook on Databricks Serverless GPU (A10 is sufficient for this toy LP).**
# MAGIC Use serverless **environment version 5 or 6** (set it in the notebook's Environment panel,
# MAGIC right sidebar) — this is the AI-runtime line that ships the CUDA 12 / torch 2.9 stack cuOpt
# MAGIC needs.
# MAGIC
# MAGIC This notebook demonstrates how to solve the classic diet optimization problem using the
# MAGIC NVIDIA cuOpt Python API (PDLP, GPU-accelerated linear programming). The problem involves
# MAGIC selecting foods to meet nutritional requirements while minimizing cost.
# MAGIC
# MAGIC It is **parameterized for a price-shock sweep**: the `cuopt_diet_optimization` job
# MAGIC (`resources/cuopt_diet_jobs.yml`) fans this notebook out over several cost scenarios with a
# MAGIC `for_each` task, all against the same pre-baked Serverless GPU environment.
# MAGIC
# MAGIC ## Problem Description
# MAGIC
# MAGIC We need to select quantities of different foods to:
# MAGIC - Meet minimum and maximum nutritional requirements
# MAGIC - Minimize total cost
# MAGIC - Satisfy additional constraints (like limiting dairy servings)
# MAGIC
# MAGIC The nutrition guidelines are based on USDA Dietary Guidelines for Americans, 2005.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters & environment
# MAGIC
# MAGIC cuOpt is **pre-baked into the Serverless GPU environment** by the job
# MAGIC (`environment_version: "5"` + `cuopt-cu12` from `pypi.nvidia.com` in the job's
# MAGIC `environments` block), so there is **no `%pip install` here** — repeated runs reuse the
# MAGIC cached environment instead of reinstalling ~630 MB of wheels each time. To run this
# MAGIC notebook interactively *without* the job environment, uncomment the fallback install below.
# MAGIC
# MAGIC Parameters (widgets, also set per `for_each` iteration by the job):
# MAGIC - `scenario_name` — a named price-shock scenario (see `PRICE_SCENARIOS` below).
# MAGIC - `price_multipliers` — optional JSON `{food: multiplier}` that **overrides** the named
# MAGIC   scenario, e.g. `{"milk": 1.5}`. Leave as `{}` to use `scenario_name`.
# MAGIC
# MAGIC To target a CUDA 13 runtime instead, use the `-cu13` packages in the job environment
# MAGIC (`cuopt-cu13`, `nvidia-nvjitlink-cu13`).

# COMMAND ----------

# Fallback for interactive runs that are NOT using the pre-baked job environment.
# These are intentionally inactive (plain comments) so the pre-baked env is used by default.
# To install manually, paste each into its OWN cell as a magic and run them:
#   %pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12 nvidia-nvjitlink-cu12 rapids-logger==0.1.19
#   %restart_python

# COMMAND ----------

import json
import time
import subprocess

# First executable cell = true notebook start (no %restart_python when the env is pre-baked).
NOTEBOOK_START = time.time()

dbutils.widgets.text("scenario_name", "baseline")
dbutils.widgets.text("price_multipliers", "{}")  # optional JSON override, e.g. {"milk": 1.5}
SCENARIO_NAME = dbutils.widgets.get("scenario_name")
PRICE_MULTIPLIERS_OVERRIDE = dbutils.widgets.get("price_multipliers")

print(f"scenario_name     = {SCENARIO_NAME!r}")
print(f"price_multipliers = {PRICE_MULTIPLIERS_OVERRIDE!r}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm we are actually on a GPU
# MAGIC
# MAGIC Lean `nvidia-smi` sanity check so the run is unambiguous. We also record wall-clock from
# MAGIC `NOTEBOOK_START` so the job can report cold-start (notebook attach → GPU ready) versus the
# MAGIC solve itself — the metric Kinaxis cares about. With the pre-baked environment there is no
# MAGIC in-notebook install, so this checkpoint reflects true notebook start.

# COMMAND ----------

try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout)
except Exception as e:
    print("nvidia-smi not available:", e)

gpu_ready_elapsed = time.time() - NOTEBOOK_START
print(f"GPU sanity check completed {gpu_ready_elapsed:.3f}s after notebook start")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Required Libraries
# MAGIC
# MAGIC `cuopt` is provided by the pre-baked Serverless GPU environment (or the fallback install above).

# COMMAND ----------

import numpy as np
import pandas as pd
from cuopt.linear_programming.problem import Problem, VType, sense, LinearExpression
from cuopt.linear_programming.solver_settings import SolverSettings
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem Data Setup
# MAGIC
# MAGIC Define the nutrition guidelines, food costs, and nutritional values for each food item.

# COMMAND ----------

# Nutrition guidelines based on USDA Dietary Guidelines for Americans, 2005
# http://www.health.gov/DietaryGuidelines/dga2005/

# minimum and maximum values for each category
categories = {
    "calories": {
        "min": 1800,
        "max": 2200
    },
    "protein": {
        "min": 91,
        "max": float('inf')
    },
    "fat": {
        "min": 0,
        "max": 65
    },
    "sodium": {
        "min": 0,
        "max": 1779
    }
}

# COMMAND ----------

# Food costs per serving
food_costs = {
    "hamburger": 2.49,
    "chicken": 2.89,
    "hot dog": 1.50,
    "fries": 1.89,
    "macaroni": 2.09,
    "pizza": 1.99,
    "salad": 2.49,
    "milk": 0.89,
    "ice cream": 1.59
}

# Nutrition values for each food (per serving)
nutrition_data = {
    "hamburger": [410, 24, 26, 730],
    "chicken": [420, 32, 10, 1190],
    "hot dog": [560, 20, 32, 1800],
    "fries": [380, 4, 19, 270],
    "macaroni": [320, 12, 10, 930],
    "pizza": [320, 15, 12, 820],
    "salad": [320, 31, 12, 1230],
    "milk": [100, 8, 2.5, 125],
    "ice cream": [330, 8, 10, 180]
}

# COMMAND ----------

# Apply the price-shock scenario. The LP model itself is unchanged — only the per-serving
# costs are perturbed, so cuOpt re-solves for the cheapest diet under each cost regime.
PRICE_SCENARIOS = {
    "baseline": {},                                            # unchanged costs
    "dairy_shock": {"milk": 1.5, "ice cream": 1.5},            # milk dominates the baseline plan
    "meat_inflation": {"hamburger": 1.4, "chicken": 1.4, "hot dog": 1.4},
    "carb_discount": {"fries": 0.5, "macaroni": 0.5, "pizza": 0.5},
    "broad_shock": {"__all__": 1.25},                          # 25% across-the-board inflation
}

_override = PRICE_MULTIPLIERS_OVERRIDE.strip()
if _override and _override != "{}":
    multipliers = json.loads(_override)
else:
    multipliers = PRICE_SCENARIOS.get(SCENARIO_NAME, {})

base_costs = dict(food_costs)
if "__all__" in multipliers:
    m = multipliers["__all__"]
    food_costs = {k: round(v * m, 4) for k, v in food_costs.items()}
else:
    for food, mult in multipliers.items():
        if food in food_costs:
            food_costs[food] = round(food_costs[food] * mult, 4)

print(f"Scenario: {SCENARIO_NAME!r}  multipliers={multipliers}")
print("Food costs ($/serving):")
for k in food_costs:
    tag = "" if food_costs[k] == base_costs[k] else f"   (was ${base_costs[k]:.2f})"
    print(f"  {k:10s} ${food_costs[k]:.2f}{tag}")

# COMMAND ----------

# Create a DataFrame for better visualization
nutrition_df = pd.DataFrame(nutrition_data, index=categories.keys()).T
nutrition_df.columns = [f"{cat} (per serving)" for cat in categories.keys()]
print("Nutritional Values per Serving:")
print(nutrition_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem Formulation
# MAGIC
# MAGIC Now we'll create the optimization problem using the cuOpt Python API as LP. The problem has:
# MAGIC - **Variables**: Amount of each food to buy (continuous, non-negative)
# MAGIC - **Objective**: Minimize total cost
# MAGIC - **Constraints**: Meet nutritional requirements (minimum and maximum bounds)

# COMMAND ----------

# Create the optimization problem
problem = Problem("diet_optimization")

# Add decision variables for each food (amount to buy)
buy_vars = {}
for food_name in food_costs:
    var = problem.addVariable(name=f"{food_name}", vtype=VType.CONTINUOUS, lb=0.0, ub=float('inf'))
    buy_vars[food_name] = var

print(f"Created {len(buy_vars)} decision variables for foods")
print(f"Variables: {[var.getVariableName() for var in buy_vars.values()]}")

# COMMAND ----------

# Set objective function: minimize total cost
objective_expr = LinearExpression([], [], 0.0)

for var in buy_vars.values():
    if food_costs[var.getVariableName()] != 0:  # Only include non-zero coefficients
        objective_expr += var * food_costs[var.getVariableName()]

# Set objective function: minimize total cost
problem.setObjective(objective_expr, sense.MINIMIZE)

# COMMAND ----------

# Add nutrition constraints
constraint_names = []

for i, category in enumerate(categories):
    # Calculate total nutrition from all foods for this category
    nutrition_expr = LinearExpression([], [], 0.0)

    for food_name in food_costs:
        nutrition_value = nutrition_data[food_name][i]
        if nutrition_value != 0:  # Only include non-zero coefficients
            nutrition_expr += buy_vars[food_name] * nutrition_value

    # Add constraint: min_nutrition[i] <= nutrition_expr <= max_nutrition[i]
    min_val = categories[category]["min"]
    max_val = categories[category]["max"]

    if max_val == float('inf'):
        # Only lower bound constraint
        constraint = problem.addConstraint(nutrition_expr >= min_val, name=f"min_{category}")
        constraint_names.append(f"min_{category}")
    else:
        # Range constraint (both lower and upper bounds)
        constraint = problem.addConstraint(nutrition_expr >= min_val, name=f"min_{category}")
        constraint_names.append(f"min_{category}")
        constraint = problem.addConstraint(nutrition_expr <= max_val, name=f"max_{category}")
        constraint_names.append(f"max_{category}")

print(f"Added {len(constraint_names)} nutrition constraints")
print(f"Constraints: {constraint_names}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Solver Configuration and Solution
# MAGIC
# MAGIC Configure the solver settings and solve the optimization problem.

# COMMAND ----------

# Configure solver settings
settings = SolverSettings()
settings.set_parameter("time_limit", 60.0)  # 60 second time limit
settings.set_parameter("log_to_console", True)  # Enable solver logging
settings.set_parameter("method", 0)  # Use default method

print("Solver configured with 60-second time limit")

# COMMAND ----------

# Solve the problem
print("Solving diet optimization problem...")
print(f"Problem type: {'MIP' if problem.IsMIP else 'LP'}")

start_time = time.time()
problem.solve(settings)
solve_time = time.time() - start_time

print(f"\nSolve completed in {solve_time:.3f} seconds")
print(f"Solver status: {problem.Status.name}")
print(f"Objective value: ${problem.ObjValue:.2f}")

# Wall-clock split for the Serverless GPU runtime report: cold-start vs. solve.
print(f"\n--- Serverless GPU timing ---")
print(f"Notebook start -> GPU ready : {gpu_ready_elapsed:.3f}s")
print(f"Solve                       : {solve_time:.3f}s")
print(f"Notebook start -> solve done: {time.time() - NOTEBOOK_START:.3f}s")

# COMMAND ----------

def print_solution():
    """Print the optimal solution in a readable format"""
    if problem.Status.name == "Optimal":
        print(f"\nOptimal Solution Found!")
        print(f"Total Cost: ${problem.ObjValue:.2f}")
        print("\nFood Purchases:")

        total_cost = 0
        for var in buy_vars.values():
            amount = var.getValue()
            if amount > 0.0001:  # Only show foods with significant amounts
                food_cost = amount * food_costs[var.getVariableName()]
                total_cost += food_cost
                print(f"  {var.getVariableName()}: {amount:.3f} servings (${food_cost:.2f})")

        print(f"\nTotal Cost: ${total_cost:.2f}")

        # Check nutritional intake
        print("\nNutritional Intake:")
        for i, category in enumerate(categories):
            total_nutrition = 0
            for var in buy_vars.values():
                amount = var.getValue()
                nutrition_value = nutrition_data[var.getVariableName()][i]
                total_nutrition += amount * nutrition_value

            min_req = categories[category]["min"]
            max_req = categories[category]["max"]

            # Check constraints with tolerance for floating point precision
            tolerance = 1e-6
            min_satisfied = total_nutrition >= (min_req - tolerance)
            max_satisfied = (max_req == float('inf')) or (total_nutrition <= (max_req + tolerance))
            status = "✓" if (min_satisfied and max_satisfied) else "✗"

            if max_req == float('inf'):
                print(f"  {category}: {total_nutrition:.1f} (min: {min_req}) {status}")
            else:
                print(f"  {category}: {total_nutrition:.1f} (min: {min_req}, max: {max_req}) {status}")
    else:
        print(f"No optimal solution found. Status: {problem.Status.name}")

print_solution()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sensitivity Analysis: Dual Values and Reduced Costs
# MAGIC
# MAGIC Every variable here is continuous, so cuOpt returns dual information at the optimum — the economic read behind the plan:
# MAGIC
# MAGIC - Each nutrition limit carries a **dual value** — at a non-degenerate optimum, how much total cost moves per unit you tighten or relax that limit. The implication runs one way: a limit with slack prices to ~0, while a binding limit (slack ≈ 0) *can* carry a nonzero dual but need not (a binding limit with a zero dual is a form of degeneracy). And a dual is $ per *that constraint's own unit* — per kcal for calories, per mg for sodium — so raw magnitudes aren't comparable across limits; to find where renegotiating pays off most, compare the value of, say, a 1% relaxation of each limit rather than ranking raw duals.
# MAGIC - A food left at 0 carries a **reduced cost** — roughly how far its per-serving price must fall before it *could* enter the diet without raising total cost. Reduced costs are per serving and serving sizes are arbitrary, so compare each as a fraction of the food's own price ("needs a 5% price cut" vs. "a 40% one") rather than sorting raw values.
# MAGIC
# MAGIC Two caveats keep the read honest: under degeneracy a dual is a local, one-sided rate — confirm any rate you quote with a one-unit re-solve — and cuOpt's default first-order (PDLP) path can return duals without a simplex basis, accurate only to the convergence tolerance, so tiny near-zero values are noise, not signal.

# COMMAND ----------

# Sensitivity analysis — read the LP duals at the optimum
if problem.Status.name == "Optimal":
    print("Constraint duals — local marginal cost per unit of each limit (units differ per constraint):")
    for c in problem.getConstraints():
        print(f"  {c.ConstraintName:14s} dual={c.DualValue:+.4f}  slack={c.Slack:.4f}")

    print("\nReduced costs (variable duals) — for foods at 0, ~price drop before it could enter the diet:")
    for v in problem.getVariables():
        print(f"  {v.VariableName:12s} amount={v.getValue():7.3f}  reduced_cost={v.ReducedCost:+.4f}")
else:
    print(f"No duals available — solver status is {problem.Status.name}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC This notebook demonstrated how to:
# MAGIC
# MAGIC 1. **Formulate a diet optimization problem** using the cuOpt Python API
# MAGIC 2. **Set up decision variables** for food quantities
# MAGIC 3. **Define an objective function** to minimize total cost
# MAGIC 4. **Add nutritional constraints** with both lower and upper bounds
# MAGIC 5. **Solve the optimization problem** using cuOpt's high-performance solver
# MAGIC 6. **Read dual values and reduced costs** for a local sensitivity read — which limits drive cost at the margin, and which unused foods sit closest to entering
# MAGIC
# MAGIC The cuOpt Python API provides a clean, intuitive interface for building and solving optimization problems, making it easy to model complex real-world scenarios like diet optimization.
# MAGIC
# MAGIC ### Key Benefits of cuOpt:
# MAGIC - **High Performance**: GPU-accelerated solving for large-scale problems
# MAGIC - **Easy to Use**: Intuitive Python API similar to other optimization libraries
# MAGIC - **Flexible**: Support for both LP and MIP problems
# MAGIC - **Scalable**: Handles problems with thousands of variables and constraints efficiently

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# MAGIC
# MAGIC SPDX-License-Identifier: Apache-2.0
# MAGIC
# MAGIC Licensed under the Apache License, Version 2.0 (the "License");
# MAGIC you may not use this file except in compliance with the License.
# MAGIC You may obtain a copy of the License at
# MAGIC
# MAGIC http://www.apache.org/licenses/LICENSE-2.0
# MAGIC
# MAGIC Unless required by applicable law or agreed to in writing, software
# MAGIC distributed under the License is distributed on an "AS IS" BASIS,
# MAGIC WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# MAGIC See the License for the specific language governing permissions and
# MAGIC limitations under the License.

# COMMAND ----------

# Return a compact result so each for_each iteration's outcome is visible in the run output.
result = {
    "scenario": SCENARIO_NAME,
    "multipliers": multipliers,
    "status": problem.Status.name,
    "objective": round(problem.ObjValue, 4) if problem.Status.name == "Optimal" else None,
    "plan": {v.getVariableName(): round(v.getValue(), 3)
             for v in buy_vars.values() if v.getValue() > 1e-4},
    "solve_time_s": round(solve_time, 4),
    "gpu_ready_s": round(gpu_ready_elapsed, 4),
    "wall_to_solve_s": round(time.time() - NOTEBOOK_START, 4),
}
print(json.dumps(result, indent=2))
dbutils.notebook.exit(json.dumps(result))
