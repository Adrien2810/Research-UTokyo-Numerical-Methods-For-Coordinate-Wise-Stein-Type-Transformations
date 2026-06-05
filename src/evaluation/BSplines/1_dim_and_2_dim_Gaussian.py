from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from scipy.optimize import LinearConstraint, minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
TOY_FILE = ROOT / "Toy 1 Gaussian.py"
BSPLINE_FILE = ROOT / "B-Spline approach.py"
OUTPUT_DIR = ROOT / "1_dim_and_2_dim_Gaussian_output"

# load the modules from the other files
def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

toy_module = load_module("toy_1_gaussian_module", TOY_FILE)
bspline_module = load_module("bspline_module", BSPLINE_FILE)

# load the specified classes and functions from the modules
Gaussian1DToyProblem = toy_module.Gaussian1DToyProblem
GaussianBiUnitToyProblem = toy_module.GaussianBiUnitToyProblem

BSplineConfig = bspline_module.BSplineConfig
MonteCarloSamples = bspline_module.MonteCarloSamples
BSplineApproach = bspline_module.BSplineApproach

# Define the number of transforms for 1D and 2D cases
number_transforms_1D = 1
number_transforms_2D = 2

# Now define run configurations for both cases
# You can play around with the parameters
