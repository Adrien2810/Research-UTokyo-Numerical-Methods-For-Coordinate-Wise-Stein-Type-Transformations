from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(
    r"C:\Users\adrie\.anaconda\UTokyo Research\Research-UTokyo-Numerical-Methods-For-Coordinate-Wise-Stein-Type-Transformations\src\evaluation\BSplines"
)
TOY_FILE = Path(
    r"C:\Users\adrie\.anaconda\UTokyo Research\Research-UTokyo-Numerical-Methods-For-Coordinate-Wise-Stein-Type-Transformations\src\toy_problems\Toy 1 Gaussian.py"
)
BSPLINE_FILE = Path(
    r"C:\Users\adrie\.anaconda\UTokyo Research\Research-UTokyo-Numerical-Methods-For-Coordinate-Wise-Stein-Type-Transformations\src\methods\B-Spline approach.py"
)


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

Gaussian1DToyProblem = toy_module.Gaussian1DToyProblem

GaussianBiUnitToyProblem = toy_module.GaussianBiUnitToyProblem
BSplineConfig = bspline_module.BSplineConfig
MonteCarloSamples = bspline_module.MonteCarloSamples
BSplineApproach = bspline_module.BSplineApproach


@dataclass
class RunConfig1D:
    num_samples: int = 10000
    random_state: int = 7
    number_transforms: int = 1
    optimizer_backend: str = "cvxpy"
    degree: int = 3
    domain: tuple[float, float] = (-5.0, 5.0)
    num_internal_knots: int = 10
    maxiter: int = 250
    minimum_derivative: float = 1e-5
    grid_size: int = 400
    histogram_bins: int = 45

@dataclass
class RunConfig2D:
    num_samples: int = 30000
    random_state: int = 7
    number_transforms: int = 2
    optimizer_backend: str = "cvxpy"
    degree: int = 3
    domain_1: tuple[float, float] = (-5.0, 5.0)
    domain_2: tuple[float, float] = (-5.0, 5.0)
    num_internal_knots: int = 10
    maxiter: int = 250
    minimum_derivative: float = 1e-5
    grid_size: int = 300
    histogram_bins: int = 45
    covariance: np.ndarray = field(
    default_factory=lambda: np.array([[0.7, 0.3], [0.3, 0.7]], dtype=float)
    )
    
def knot_average(knots: np.ndarray, degree: int) -> np.ndarray:
    """Greville abscissae for the identity map."""
    if degree <= 0:
        raise ValueError("Greville abscissae require degree >= 1.")

    n_basis = len(knots) - degree - 1
    values = np.zeros(n_basis, dtype=float)
    for j in range(n_basis):
        values[j] = np.mean(knots[j + 1 : j + degree + 1])
    return values


def open_uniform_knots(a: float, b: float, degree: int, n_internal: int) -> np.ndarray:
    """Construct an open uniform knot vector on [a, b]."""
    if b <= a:
        raise ValueError("The right endpoint must be larger than the left endpoint.")
    if degree < 0:
        raise ValueError("The spline degree must be non-negative.")
    if n_internal < 0:
        raise ValueError("The number of internal knots must be non-negative.")

    internal = np.linspace(a, b, n_internal + 2, dtype=float)[1:-1]
    left = np.repeat(a, degree + 1)
    right = np.repeat(b, degree + 1)
    return np.concatenate([left, internal, right])


def extract_history(bspline_approach, result) -> dict:
    """Read optimizer history from the class or result object."""
    candidates = [
        getattr(result, "history", None),
        getattr(result, "optimization_history", None),
        getattr(bspline_approach, "history", None),
        getattr(bspline_approach, "optimization_history", None),
        getattr(bspline_approach, "callback_history", None),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate

    return {}


def run_1d(config: RunConfig1D):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up the toy problem and draw samples.
    problem = Gaussian1DToyProblem()
    raw_points = problem.sample(config.num_samples, random_state=config.random_state)

    # Keep only samples inside the spline domain.
    in_domain_mask = (
        (raw_points[:, 0] >= config.domain[0]) &
        (raw_points[:, 0] <= config.domain[1])
    )
    filtered_points = raw_points[in_domain_mask]

    if filtered_points.shape[0] == 0:
        raise ValueError("No samples remained inside the chosen spline domain.")

    samples = MonteCarloSamples(points=filtered_points)

    # Build the spline configuration.
    knots = open_uniform_knots(
        a=config.domain[0],
        b=config.domain[1],
        degree=config.degree,
        n_internal=config.num_internal_knots,
    )
    bspline_config = BSplineConfig(
        degree=config.degree,
        knots=knots,
        domain=config.domain,
    )

    # Initialize the B-spline method.
    bspline_approach = BSplineApproach(
        problem=problem,
        config=bspline_config,
        samples=samples,
    )

    # For the Gaussian 1D toy problem, the identity is the exact solution.
    # We give perturbed knot avergae being the exact solution as an initial guess.
    initial_coefficients = 1.5* knot_average(knots, config.degree) + 0.2 * np.random.randn(bspline_config.n_basis)
    exact_coefficients =  knot_average(knots, config.degree)

    start_time = perf_counter()
    result = bspline_approach.optimize_with_backend(
        initial_guess=initial_coefficients,
        number_transforms=config.number_transforms,
        backend=config.optimizer_backend,
        maxiter=config.maxiter,
        minimum_derivative=config.minimum_derivative,
    )
    runtime_seconds = perf_counter() - start_time

    sample_values = filtered_points[:, 0]
    sample_min = float(np.min(sample_values))
    sample_max = float(np.max(sample_values))
    grid = np.linspace(sample_min, sample_max, config.grid_size)
    exact_solution = np.asarray(problem.exact_transformation(grid), dtype=float)

    # Final fitted map and derivative on the sample-supported interval.
    fitted_grid = bspline_approach.transform(
        grid,
        result.x,
        transform_index=0,
    )
    fitted_derivative_grid = bspline_approach.transform_derivative(
        grid,
        result.x,
        transform_index=0,
    )
    initial_grid = bspline_approach.transform(
        grid,
        initial_coefficients,
        transform_index=0,
    )

    l2_error = float(np.sqrt(np.mean((fitted_grid - exact_solution) ** 2)))
    linf_error = float(np.max(np.abs(fitted_grid - exact_solution)))
    derivative_linf_error = float(np.max(np.abs(fitted_derivative_grid - 1.0)))
    final_objective = float(
        bspline_approach.MonteCarloObjective(result.x, config.number_transforms)
    )

    # Recover callback history from the optimizer/class.
    history = extract_history(bspline_approach, result)
    coefficient_history = history.get("coefficients", [])
    objective_history = history.get("objective", [])

    if not coefficient_history:
        coefficient_history = [np.copy(result.x)]
    if not objective_history:
        objective_history = [final_objective]

    objective_iterations = np.arange(1, len(objective_history) + 1)

    # Compute error history afterwards, outside the optimizer.
    l2_history = []
    linf_history = []
    for coeffs in coefficient_history:
        fitted_grid_iter = bspline_approach.transform(
            grid,
            coeffs,
            transform_index=0,
        )
        l2_history.append(
            float(np.sqrt(np.mean((fitted_grid_iter - exact_solution) ** 2)))
        )
        linf_history.append(
            float(np.max(np.abs(fitted_grid_iter - exact_solution)))
        )

    convergence_plot_path = OUTPUT_DIR / "gaussian_1d_convergence.png"
    solution_plot_path = OUTPUT_DIR / "gaussian_1d_solution.png"
    distribution_plot_path = OUTPUT_DIR / "gaussian_1d_transformed_distribution.png"
    summary_path = OUTPUT_DIR / "gaussian_1d_summary.txt"

    # Convergence plots on linear and log scales.
    objective_history = np.asarray(objective_history, dtype=float)
    l2_history = np.asarray(l2_history, dtype=float)
    linf_history = np.asarray(linf_history, dtype=float)

    # Avoid log-scale issues if one curve hits zero exactly.
    log_floor = np.finfo(float).tiny
    objective_log = np.maximum(np.abs(objective_history), log_floor)
    l2_log = np.maximum(l2_history, log_floor)
    linf_log = np.maximum(linf_history, log_floor)

    # Now plot -> Codex did this
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex="col")
    iterations = np.arange(1, len(objective_history) + 1)

    axes[0, 0].plot(
        iterations,
        objective_history,
        color="#154c79",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0, 0].set_title("Objective Convergence")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Empirical objective")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        iterations,
        l2_history,
        label="L2 error",
        color="#b7410e",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0, 1].plot(
        iterations,
        linf_history,
        label="Linf error",
        color="#2d6a4f",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0, 1].set_title("Error Convergence to Identity")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Error on sample interval")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].semilogy(
        iterations,
        objective_log,
        color="#154c79",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[1, 0].set_title("Objective Convergence (log scale)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Absolute objective")
    axes[1, 0].grid(alpha=0.3, which="both")

    axes[1, 1].semilogy(
        iterations,
        l2_log,
        label="L2 error",
        color="#b7410e",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[1, 1].semilogy(
        iterations,
        linf_log,
        label="Linf error",
        color="#2d6a4f",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[1, 1].set_title("Error Convergence (log scale)")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Error on sample interval")
    axes[1, 1].grid(alpha=0.3, which="both")
    axes[1, 1].legend()

    if len(iterations) == 1:
        for axis in axes.ravel():
            axis.set_xlim(0.5, 1.5)

    fig.tight_layout()
    fig.savefig(convergence_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Solution and derivative plot.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(
        grid,
        exact_solution,
        label="Exact identity",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    axes[0].plot(
        grid,
        initial_grid,
        label="Initial spline",
        color="#c1121f",
        alpha=0.8,
        linewidth=2,
    )
    axes[0].plot(
        grid,
        fitted_grid,
        label="Optimized spline",
        color="#003049",
        linewidth=2.5,
    )
    axes[0].set_title("Solution Map")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("T(x)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        grid,
        np.ones_like(grid),
        label="Exact derivative",
        color="black",
        linestyle="--",
        linewidth=2,
    )
    axes[1].plot(
        grid,
        fitted_derivative_grid,
        label="Optimized derivative",
        color="#669bbc",
        linewidth=2.5,
    )
    axes[1].set_title("Derivative of the Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("T'(x)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(solution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Histogram before and after transformation.
    transformed_samples = bspline_approach.transform(
        sample_values,
        result.x,
        transform_index=0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(
        sample_values,
        bins=config.histogram_bins,
        density=True,
        alpha=0.8,
        color="#6c757d",
    )
    axes[0].set_title("Input Gaussian Samples")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.2)

    axes[1].hist(
        transformed_samples,
        bins=config.histogram_bins,
        density=True,
        alpha=0.8,
        color="#0a9396",
    )
    axes[1].set_title("Transformed Samples")
    axes[1].set_xlabel("T(x)")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(distribution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Pointwise values for quick inspection.
    sample_points = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    sample_solution_values = bspline_approach.transform(
        sample_points,
        result.x,
        transform_index=0,
    )
    sample_solution_derivatives = bspline_approach.transform_derivative(
        sample_points,
        result.x,
        transform_index=0,
    )

    summary_lines = [
        "Gaussian 1D B-spline evaluation",
        "",
        f"Runtime (seconds): {runtime_seconds:.6f}",
        f"Optimizer backend: {config.optimizer_backend}",
        f"Optimizer success: {result.success}",
        f"Optimizer status: {result.status}",
        f"Optimizer message: {result.message}",
        f"Minimum derivative lower bound: {config.minimum_derivative:.2e}",
        f"Iterations recorded: {len(objective_history)}",
        f"Number of in-domain Monte Carlo samples: {sample_values.size}",
        f"Spline domain: {config.domain}",
        f"Sample interval: [{sample_min:.6f}, {sample_max:.6f}]",
        f"Spline degree: {config.degree}",
        f"Number of basis functions: {bspline_config.n_basis}",
        f"Final empirical objective: {final_objective:.10f}",
        f"L2 error on sample interval: {l2_error:.10e}",
        f"Linf error on sample interval: {linf_error:.10e}",
        f"Linf derivative error on sample interval: {derivative_linf_error:.10e}",
        "",
        "Optimized B-spline coefficients:",
        np.array2string(result.x, precision=8, separator=", "),
        "",
        "Reference identity coefficients (Greville abscissae):",
        np.array2string(exact_coefficients, precision=8, separator=", "),
        "",
        "Solution values at sample points:",
    ]

    for x_value, t_value, dt_value in zip(
        sample_points,
        sample_solution_values,
        sample_solution_derivatives,
    ):
        summary_lines.append(
            f"  x={x_value:5.2f} -> T(x)={t_value: .8f}, T'(x)={dt_value: .8f}"
        )

    summary_lines.extend(
        [
            "",
            f"Convergence plot: {convergence_plot_path.name}",
            f"Solution plot: {solution_plot_path.name}",
            f"Distribution plot: {distribution_plot_path.name}",
        ]
    )

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    return {
        "result": result,
        "history": history,
        "samples": sample_values,
        "grid": grid,
        "fitted_grid": fitted_grid,
        "fitted_derivative_grid": fitted_derivative_grid,
        "output_dir": OUTPUT_DIR,
    }


def run_2d(config: RunConfig2D):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up the 2D bi-unit Gaussian toy problem.
    problem = GaussianBiUnitToyProblem(covariance=config.covariance)
    raw_points = problem.sample(config.num_samples, random_state=config.random_state)

    # Keep only samples inside the rectangular spline domain.
    in_domain_mask = (
        (raw_points[:, 0] >= config.domain_1[0]) &
        (raw_points[:, 0] <= config.domain_1[1]) &
        (raw_points[:, 1] >= config.domain_2[0]) &
        (raw_points[:, 1] <= config.domain_2[1])
    )
    filtered_points = raw_points[in_domain_mask]

    if filtered_points.shape[0] == 0:
        raise ValueError("No samples remained inside the chosen spline domains.")

    samples = MonteCarloSamples(points=filtered_points)

    # Build one spline configuration per coordinate.
    knots_1 = open_uniform_knots(
        a=config.domain_1[0],
        b=config.domain_1[1],
        degree=config.degree,
        n_internal=config.num_internal_knots,
    )
    knots_2 = open_uniform_knots(
        a=config.domain_2[0],
        b=config.domain_2[1],
        degree=config.degree,
        n_internal=config.num_internal_knots,
    )

    bspline_config_1 = BSplineConfig(
        degree=config.degree,
        knots=knots_1,
        domain=config.domain_1,
    )
    bspline_config_2 = BSplineConfig(
        degree=config.degree,
        knots=knots_2,
        domain=config.domain_2,
    )

    bspline_approach = BSplineApproach(
        problem=problem,
        config=[bspline_config_1, bspline_config_2],
        samples=samples,
    )

    # Identity is the exact map for the bi-unit Gaussian toy problem.
    initial_coefficients_1 = np.linspace(-1.5, 1.5, bspline_config_1.n_basis)
    initial_coefficients_2 = np.linspace(-1.0, 2.0, bspline_config_2.n_basis)
    initial_guess = np.concatenate([initial_coefficients_1, initial_coefficients_2])
    
    exact_coefficients_1 = knot_average(knots_1, config.degree)
    exact_coefficients_2 = knot_average(knots_2, config.degree)
    exact_coefficients = np.concatenate([exact_coefficients_1, exact_coefficients_2])

    start_time = perf_counter()
    result = bspline_approach.optimize_with_backend(
        initial_guess=initial_guess,
        number_transforms=config.number_transforms,
        backend=config.optimizer_backend,
        maxiter=config.maxiter,
        minimum_derivative=config.minimum_derivative,
    )
    runtime_seconds = perf_counter() - start_time

    sample_values_1 = filtered_points[:, 0]
    sample_values_2 = filtered_points[:, 1]

    sample_min_1 = float(np.min(sample_values_1))
    sample_max_1 = float(np.max(sample_values_1))
    sample_min_2 = float(np.min(sample_values_2))
    sample_max_2 = float(np.max(sample_values_2))

    grid_1 = np.linspace(sample_min_1, sample_max_1, config.grid_size)
    grid_2 = np.linspace(sample_min_2, sample_max_2, config.grid_size)

    exact_solution_1 = grid_1.copy()
    exact_solution_2 = grid_2.copy()

    coeff_blocks = bspline_approach._split_coefficients(result.x, config.number_transforms)

    fitted_grid_1 = bspline_approach.transform(grid_1, coeff_blocks[0], transform_index=0)
    fitted_grid_2 = bspline_approach.transform(grid_2, coeff_blocks[1], transform_index=1)

    fitted_derivative_grid_1 = bspline_approach.transform_derivative(grid_1, coeff_blocks[0], transform_index=0)
    fitted_derivative_grid_2 = bspline_approach.transform_derivative(grid_2, coeff_blocks[1], transform_index=1)

    initial_grid_1 = bspline_approach.transform(grid_1, initial_coefficients_1, transform_index=0)
    initial_grid_2 = bspline_approach.transform(grid_2, initial_coefficients_2, transform_index=1)

    l2_error_1 = float(np.sqrt(np.mean((fitted_grid_1 - exact_solution_1) ** 2)))
    l2_error_2 = float(np.sqrt(np.mean((fitted_grid_2 - exact_solution_2) ** 2)))
    linf_error_1 = float(np.max(np.abs(fitted_grid_1 - exact_solution_1)))
    linf_error_2 = float(np.max(np.abs(fitted_grid_2 - exact_solution_2)))

    derivative_linf_error_1 = float(np.max(np.abs(fitted_derivative_grid_1 - 1.0)))
    derivative_linf_error_2 = float(np.max(np.abs(fitted_derivative_grid_2 - 1.0)))

    final_objective = float(
        bspline_approach.MonteCarloObjective(result.x, config.number_transforms)
    )

    history = extract_history(bspline_approach, result)
    coefficient_history = history.get("coefficients", [])
    objective_history = history.get("objective", [])

    if not coefficient_history:
        coefficient_history = [np.copy(result.x)]
    if not objective_history:
        objective_history = [final_objective]

    objective_iterations = np.arange(1, len(objective_history) + 1)

    # Compute convergence histories after optimization.
    l2_history = []
    linf_history = []

    for coeffs in coefficient_history:
        coeff_blocks_iter = bspline_approach._split_coefficients(coeffs, config.number_transforms)

        fitted_grid_iter_1 = bspline_approach.transform(grid_1, coeff_blocks_iter[0], transform_index=0)
        fitted_grid_iter_2 = bspline_approach.transform(grid_2, coeff_blocks_iter[1], transform_index=1)

        l2_iter_1 = np.sqrt(np.mean((fitted_grid_iter_1 - exact_solution_1) ** 2))
        l2_iter_2 = np.sqrt(np.mean((fitted_grid_iter_2 - exact_solution_2) ** 2))
        linf_iter_1 = np.max(np.abs(fitted_grid_iter_1 - exact_solution_1))
        linf_iter_2 = np.max(np.abs(fitted_grid_iter_2 - exact_solution_2))

        l2_history.append(float(max(l2_iter_1, l2_iter_2)))
        linf_history.append(float(max(linf_iter_1, linf_iter_2)))

    objective_history = np.asarray(objective_history, dtype=float)
    l2_history = np.asarray(l2_history, dtype=float)
    linf_history = np.asarray(linf_history, dtype=float)
    error_iterations = np.arange(1, len(l2_history) + 1)
    error_iterations = np.arange(1, len(l2_history) + 1)

    log_floor = np.finfo(float).tiny
    objective_log = np.maximum(np.abs(objective_history), log_floor)
    l2_log = np.maximum(l2_history, log_floor)
    linf_log = np.maximum(linf_history, log_floor)

    convergence_plot_path = OUTPUT_DIR / "gaussian_2d_convergence.png"
    solution_plot_path = OUTPUT_DIR / "gaussian_2d_solution.png"
    distribution_plot_path = OUTPUT_DIR / "gaussian_2d_transformed_distribution.png"
    summary_path = OUTPUT_DIR / "gaussian_2d_summary.txt"

    # Convergence plots.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].plot(
        objective_iterations,
        objective_history,
        color="#154c79",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0, 0].set_title("Objective Convergence")
    axes[0, 0].set_xlabel("Objective evaluation")
    axes[0, 0].set_ylabel("Empirical objective")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        error_iterations,
        l2_history,
        label="max coord L2",
        color="#b7410e",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0, 1].plot(
        error_iterations,
        linf_history,
        label="max coord Linf",
        color="#2d6a4f",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[0, 1].set_title("Error Convergence to Identity")
    axes[0, 1].set_xlabel("Accepted iterate")
    axes[0, 1].set_ylabel("Error on sample intervals")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].semilogy(
        objective_iterations,
        objective_log,
        color="#154c79",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[1, 0].set_title("Objective Convergence (log scale)")
    axes[1, 0].set_xlabel("Objective evaluation")
    axes[1, 0].set_ylabel("Absolute objective")
    axes[1, 0].grid(alpha=0.3, which="both")

    axes[1, 1].semilogy(
        error_iterations,
        l2_log,
        label="max coord L2",
        color="#b7410e",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[1, 1].semilogy(
        error_iterations,
        linf_log,
        label="max coord Linf",
        color="#2d6a4f",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[1, 1].set_title("Error Convergence (log scale)")
    axes[1, 1].set_xlabel("Accepted iterate")
    axes[1, 1].set_ylabel("Error on sample intervals")
    axes[1, 1].grid(alpha=0.3, which="both")
    axes[1, 1].legend()

    if len(objective_iterations) == 1:
        axes[0, 0].set_xlim(0.5, 1.5)
        axes[1, 0].set_xlim(0.5, 1.5)

    if len(error_iterations) == 1:
        axes[0, 1].set_xlim(0.5, 1.5)
        axes[1, 1].set_xlim(0.5, 1.5)

    fig.tight_layout()
    fig.savefig(convergence_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Coordinate-wise solution and derivative plots.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].plot(grid_1, exact_solution_1, label="Exact identity", color="black", linestyle="--", linewidth=2)
    axes[0, 0].plot(grid_1, initial_grid_1, label="Initial spline", color="#c1121f", alpha=0.8, linewidth=2)
    axes[0, 0].plot(grid_1, fitted_grid_1, label="Optimized spline", color="#003049", linewidth=2.5)
    axes[0, 0].set_title("Solution Map T1")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("T1(x1)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(grid_2, exact_solution_2, label="Exact identity", color="black", linestyle="--", linewidth=2)
    axes[0, 1].plot(grid_2, initial_grid_2, label="Initial spline", color="#c1121f", alpha=0.8, linewidth=2)
    axes[0, 1].plot(grid_2, fitted_grid_2, label="Optimized spline", color="#003049", linewidth=2.5)
    axes[0, 1].set_title("Solution Map T2")
    axes[0, 1].set_xlabel("x2")
    axes[0, 1].set_ylabel("T2(x2)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(grid_1, np.ones_like(grid_1), label="Exact derivative", color="black", linestyle="--", linewidth=2)
    axes[1, 0].plot(grid_1, fitted_derivative_grid_1, label="Optimized derivative", color="#669bbc", linewidth=2.5)
    axes[1, 0].set_title("Derivative T1'")
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("T1'(x1)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(grid_2, np.ones_like(grid_2), label="Exact derivative", color="black", linestyle="--", linewidth=2)
    axes[1, 1].plot(grid_2, fitted_derivative_grid_2, label="Optimized derivative", color="#669bbc", linewidth=2.5)
    axes[1, 1].set_title("Derivative T2'")
    axes[1, 1].set_xlabel("x2")
    axes[1, 1].set_ylabel("T2'(x2)")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(solution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Scatter before and after transformation.
    transformed_1 = bspline_approach.transform(sample_values_1, coeff_blocks[0], transform_index=0)
    transformed_2 = bspline_approach.transform(sample_values_2, coeff_blocks[1], transform_index=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(sample_values_1, sample_values_2, s=8, alpha=0.35, color="#6c757d")
    axes[0].set_title("Input Gaussian Samples")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(transformed_1, transformed_2, s=8, alpha=0.35, color="#0a9396")
    axes[1].set_title("Transformed Samples")
    axes[1].set_xlabel("T1(x1)")
    axes[1].set_ylabel("T2(x2)")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(distribution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    sample_points_1 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    sample_points_2 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)

    sample_solution_values_1 = bspline_approach.transform(sample_points_1, coeff_blocks[0], transform_index=0)
    sample_solution_values_2 = bspline_approach.transform(sample_points_2, coeff_blocks[1], transform_index=1)

    sample_solution_derivatives_1 = bspline_approach.transform_derivative(sample_points_1, coeff_blocks[0], transform_index=0)
    sample_solution_derivatives_2 = bspline_approach.transform_derivative(sample_points_2, coeff_blocks[1], transform_index=1)

    summary_lines = [
        "Gaussian 2D B-spline evaluation",
        "",
        f"Runtime (seconds): {runtime_seconds:.6f}",
        f"Optimizer backend: {config.optimizer_backend}",
        f"Optimizer success: {result.success}",
        f"Optimizer status: {result.status}",
        f"Optimizer message: {result.message}",
        f"Minimum derivative lower bound: {config.minimum_derivative:.2e}",
        f"Objective evaluations recorded: {len(objective_history)}",
        f"Accepted iterates recorded: {len(coefficient_history)}",
        f"Number of in-domain Monte Carlo samples: {filtered_points.shape[0]}",
        f"Covariance matrix:\n{config.covariance}",
        f"Domain x1: {config.domain_1}",
        f"Domain x2: {config.domain_2}",
        f"Sample interval x1: [{sample_min_1:.6f}, {sample_max_1:.6f}]",
        f"Sample interval x2: [{sample_min_2:.6f}, {sample_max_2:.6f}]",
        f"Spline degree: {config.degree}",
        f"Number of basis functions T1: {bspline_config_1.n_basis}",
        f"Number of basis functions T2: {bspline_config_2.n_basis}",
        f"Final empirical objective: {final_objective:.10f}",
        f"L2 error T1 on sample interval: {l2_error_1:.10e}",
        f"L2 error T2 on sample interval: {l2_error_2:.10e}",
        f"Linf error T1 on sample interval: {linf_error_1:.10e}",
        f"Linf error T2 on sample interval: {linf_error_2:.10e}",
        f"Linf derivative error T1: {derivative_linf_error_1:.10e}",
        f"Linf derivative error T2: {derivative_linf_error_2:.10e}",
        "",
        "Optimized B-spline coefficients:",
        np.array2string(result.x, precision=8, separator=", "),
        "",
        "Reference identity coefficients (Greville abscissae):",
        np.array2string(exact_coefficients, precision=8, separator=", "),
        "",
        "Solution values at sample points for T1:",
    ]

    for x_value, t_value, dt_value in zip(sample_points_1, sample_solution_values_1, sample_solution_derivatives_1):
        summary_lines.append(f"  x1={x_value:5.2f} -> T1(x1)={t_value: .8f}, T1'(x1)={dt_value: .8f}")

    summary_lines.append("")
    summary_lines.append("Solution values at sample points for T2:")

    for x_value, t_value, dt_value in zip(sample_points_2, sample_solution_values_2, sample_solution_derivatives_2):
        summary_lines.append(f"  x2={x_value:5.2f} -> T2(x2)={t_value: .8f}, T2'(x2)={dt_value: .8f}")

    summary_lines.extend(
        [
            "",
            f"Convergence plot: {convergence_plot_path.name}",
            f"Solution plot: {solution_plot_path.name}",
            f"Distribution plot: {distribution_plot_path.name}",
        ]
    )

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    return {
        "result": result,
        "history": history,
        "samples": filtered_points,
        "grid_1": grid_1,
        "grid_2": grid_2,
        "output_dir": OUTPUT_DIR,
    }


if __name__ == "__main__":
    # config = RunConfig1D()
    # run_1d(config)
    config = RunConfig2D()
    run_2d(config)
