from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from scipy.stats import norm, uniform

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


toy_module = load_module("toy_uniform_module", TOY_FILE)
bspline_module = load_module("bspline_uniform_module", BSPLINE_FILE)

IndependentJointToyProblem = toy_module.IndependentJointToyProblem
ScipyDistribution = toy_module.ScipyDistribution
STANDARD_NORMAL = toy_module.STANDARD_NORMAL

BSplineConfig = bspline_module.BSplineConfig
MonteCarloSamples = bspline_module.MonteCarloSamples
BSplineApproach = bspline_module.BSplineApproach


@dataclass
class RunConfigUniform:
    num_samples: int = 15000
    random_state: int = 7
    number_transforms: int = 2
    optimizer_backend: str = "cvxpy"
    degree: int = 3
    support_1: tuple[float, float] = (0.0, 1.0)
    support_2: tuple[float, float] = (0.0, 1.0)
    domain_1: tuple[float, float] = (0.02, 0.98)
    domain_2: tuple[float, float] = (0.02, 0.98)
    num_internal_knots_1: int = 16
    num_internal_knots_2: int = 16
    maxiter: int = 250
    minimum_derivative: float = 1e-5
    grid_size: int = 400
    histogram_bins: int = 45
    stationary_tolerance: float = 1e-3


@dataclass
class RunConfigPiecewise:
    num_samples: int = 20000
    random_state: int = 7
    number_transforms: int = 3
    optimizer_backend: str = "cvxpy"
    degree: int = 3
    domain_1: tuple[float, float] = (-0.98, 0.98)
    domain_2: tuple[float, float] = (-0.98, 0.98)
    domain_3: tuple[float, float] = (-0.98, 0.98)
    num_internal_knots_1: int = 18
    num_internal_knots_2: int = 18
    num_internal_knots_3: int = 18
    maxiter: int = 250
    minimum_derivative: float = 1e-5
    grid_size: int = 400
    histogram_bins: int = 45
    stationary_tolerance: float = 1e-3
    constants: tuple[float, float, float] = (1.2490, 0.3445, 0.3445)


def knot_average(knots: np.ndarray, degree: int) -> np.ndarray:
    """Greville abscissae for a monotone spline start."""
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
    """Normalize the callback history keys coming from the method class."""
    candidates = [
        getattr(result, "history", None),
        getattr(result, "optimization_history", None),
        getattr(bspline_approach, "history", None),
        getattr(bspline_approach, "optimization_history", None),
        getattr(bspline_approach, "callback_history", None),
    ]

    for candidate in candidates:
        if isinstance(candidate, dict):
            coefficients = candidate.get("coefficients", [])
            if not coefficients:
                iteration_values = candidate.get("iteration", [])
                if (
                    isinstance(iteration_values, list)
                    and iteration_values
                    and hasattr(iteration_values[0], "__len__")
                ):
                    coefficients = iteration_values

            objective = candidate.get("objective", candidate.get("objective_value", []))
            return {
                "coefficients": list(coefficients),
                "objective": list(objective),
                "monotonicity_satisfied": list(candidate.get("monotonicity_satisfied", [])),
                "min_derivatives": list(candidate.get("min_derivatives", [])),
            }

    return {}


def exact_coordinate_map(bounds: tuple[float, float], x: np.ndarray) -> np.ndarray:
    """Exact map T(x) = Phi^{-1}((x-a)/(b-a)) for uniform marginals."""
    left, right = bounds
    probabilities = (np.asarray(x, dtype=float) - left) / (right - left)
    probabilities = np.clip(probabilities, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return norm.ppf(probabilities)


def exact_coordinate_derivative(bounds: tuple[float, float], x: np.ndarray) -> np.ndarray:
    """Derivative of the exact uniform-to-normal map."""
    transformed = exact_coordinate_map(bounds, x)
    density = 1.0 / (bounds[1] - bounds[0])
    return density / norm.pdf(transformed)


def affine_normal_range_initial_coefficients(
    knots: np.ndarray,
    degree: int,
    target_range: tuple[float, float] = (-1.5, 1.5),
) -> np.ndarray:
    """
    Neutral monotone initialization:
    map the Greville abscissae affinely into a moderate normal-looking range.

    This avoids starting from the exact solution while still giving the solver
    a reasonable increasing spline.
    """
    greville_points = knot_average(knots, degree)
    left = float(greville_points[0])
    right = float(greville_points[-1])
    target_left, target_right = target_range

    if right <= left:
        return np.full_like(greville_points, 0.5 * (target_left + target_right))

    scale = (target_right - target_left) / (right - left)
    shift = target_left - scale * left
    return scale * greville_points + shift


def diagnostic_points(interval: tuple[float, float], count: int = 5) -> np.ndarray:
    """Pick interior diagnostic points away from uniform endpoints."""
    left, right = interval
    padding = 0.1 * (right - left)
    return np.linspace(left + padding, right - padding, count, dtype=float)


def sample_piecewise_distribution(num_samples: int, random_state: int | None = None) -> np.ndarray:
    """
    Sample from the 3D piecewise-constant density on [-1, 1]^3:
    density 3/8 on (-,+,+) and (+,-,-), and 1/24 on the other six sign boxes.
    """
    rng = np.random.default_rng(random_state)
    orthants = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    probabilities = np.array(
        [1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0],
        dtype=float,
    )
    orthant_indices = rng.choice(orthants.shape[0], size=num_samples, p=probabilities)

    samples = np.empty((num_samples, 3), dtype=float)
    for index, signs in enumerate(orthants[orthant_indices]):
        lower = np.where(signs < 0.0, -1.0, 0.0)
        upper = np.where(signs < 0.0, 0.0, 1.0)
        samples[index] = rng.uniform(lower, upper)

    return samples


def piecewise_coordinate_map(constant: float, x: np.ndarray) -> np.ndarray:
    """
    Exact piecewise Stein-type transformation from the paper example.

    For x < 0:
        T(x) = -c + Phi^{-1}((1 + x) Phi(c))
    For x > 0:
        T(x) =  c - Phi^{-1}((1 - x) Phi(c))

    With this convention T is continuous at 0 and diverges at the endpoints.
    """
    x = np.asarray(x, dtype=float)
    cdf_c = norm.cdf(constant)
    transformed = np.empty_like(x, dtype=float)

    negative_mask = x < 0.0
    nonnegative_mask = ~negative_mask

    if np.any(negative_mask):
        probabilities_left = (1.0 + x[negative_mask]) * cdf_c
        probabilities_left = np.clip(
            probabilities_left,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
        transformed[negative_mask] = -constant + norm.ppf(probabilities_left)

    if np.any(nonnegative_mask):
        probabilities_right = (1.0 - x[nonnegative_mask]) * cdf_c
        probabilities_right = np.clip(
            probabilities_right,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
        transformed[nonnegative_mask] = constant - norm.ppf(probabilities_right)

    return transformed


def piecewise_coordinate_derivative(constant: float, x: np.ndarray) -> np.ndarray:
    """Derivative of the exact piecewise Stein-type transformation."""
    x = np.asarray(x, dtype=float)
    cdf_c = norm.cdf(constant)
    derivative = np.empty_like(x, dtype=float)

    negative_mask = x < 0.0
    nonnegative_mask = ~negative_mask

    if np.any(negative_mask):
        z_left = norm.ppf(
            np.clip(
                (1.0 + x[negative_mask]) * cdf_c,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
        )
        derivative[negative_mask] = cdf_c / norm.pdf(z_left)

    if np.any(nonnegative_mask):
        z_right = norm.ppf(
            np.clip(
                (1.0 - x[nonnegative_mask]) * cdf_c,
                np.finfo(float).eps,
                1.0 - np.finfo(float).eps,
            )
        )
        derivative[nonnegative_mask] = cdf_c / norm.pdf(z_right)

    return derivative


def compute_stationary_residuals(
    bspline_approach,
    flat_coefficients: np.ndarray,
    number_transforms: int,
    potential_function_derivative,
    test_function,
    test_function_derivative,
):
    """Get stationary residuals without raising, so we can summarize them."""
    return bspline_approach.check_stationary_condition(
        flat_coefficients,
        number_transforms,
        potential_function_derivative,
        test_function,
        test_function_derivative,
        tolerance=np.inf,
    )


def run_uniform_quadratic(config: RunConfigUniform):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    support_1 = config.support_1
    support_2 = config.support_2
    marginal_1 = ScipyDistribution(
        "uniform_margin_1",
        uniform(loc=support_1[0], scale=support_1[1] - support_1[0]),
    )
    marginal_2 = ScipyDistribution(
        "uniform_margin_2",
        uniform(loc=support_2[0], scale=support_2[1] - support_2[0]),
    )

    problem = IndependentJointToyProblem(
        marginals=(marginal_1, marginal_2),
        target_marginal=STANDARD_NORMAL,
        name="independent_quadratic_uniform",
        potential_name="quadratic_sum",
        description=(
            "Independent uniform marginals on a rectangle under the quadratic "
            "potential. Exact coordinate-wise map is Phi^{-1}(F_i)."
        ),
    )

    raw_points = problem.sample(config.num_samples, random_state=config.random_state)

    in_domain_mask = (
        (raw_points[:, 0] >= config.domain_1[0])
        & (raw_points[:, 0] <= config.domain_1[1])
        & (raw_points[:, 1] >= config.domain_2[0])
        & (raw_points[:, 1] <= config.domain_2[1])
    )
    filtered_points = raw_points[in_domain_mask]

    if filtered_points.shape[0] == 0:
        raise ValueError("No samples remained inside the chosen spline domains.")

    samples = MonteCarloSamples(points=filtered_points)

    knots_1 = open_uniform_knots(
        a=config.domain_1[0],
        b=config.domain_1[1],
        degree=config.degree,
        n_internal=config.num_internal_knots_1,
    )
    knots_2 = open_uniform_knots(
        a=config.domain_2[0],
        b=config.domain_2[1],
        degree=config.degree,
        n_internal=config.num_internal_knots_2,
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

    initial_coefficients_1 = affine_normal_range_initial_coefficients(knots_1, config.degree)
    initial_coefficients_2 = affine_normal_range_initial_coefficients(knots_2, config.degree)
    initial_guess = np.concatenate([initial_coefficients_1, initial_coefficients_2])

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

    exact_solution_1 = exact_coordinate_map(config.support_1, grid_1)
    exact_solution_2 = exact_coordinate_map(config.support_2, grid_2)
    exact_derivative_1 = exact_coordinate_derivative(config.support_1, grid_1)
    exact_derivative_2 = exact_coordinate_derivative(config.support_2, grid_2)

    coeff_blocks = bspline_approach._split_coefficients(result.x, config.number_transforms)

    fitted_grid_1 = bspline_approach.transform(grid_1, coeff_blocks[0], transform_index=0)
    fitted_grid_2 = bspline_approach.transform(grid_2, coeff_blocks[1], transform_index=1)
    fitted_derivative_grid_1 = bspline_approach.transform_derivative(
        grid_1,
        coeff_blocks[0],
        transform_index=0,
    )
    fitted_derivative_grid_2 = bspline_approach.transform_derivative(
        grid_2,
        coeff_blocks[1],
        transform_index=1,
    )

    initial_grid_1 = bspline_approach.transform(grid_1, initial_coefficients_1, transform_index=0)
    initial_grid_2 = bspline_approach.transform(grid_2, initial_coefficients_2, transform_index=1)

    l2_error_1 = float(np.sqrt(np.mean((fitted_grid_1 - exact_solution_1) ** 2)))
    l2_error_2 = float(np.sqrt(np.mean((fitted_grid_2 - exact_solution_2) ** 2)))
    linf_error_1 = float(np.max(np.abs(fitted_grid_1 - exact_solution_1)))
    linf_error_2 = float(np.max(np.abs(fitted_grid_2 - exact_solution_2)))
    derivative_linf_error_1 = float(np.max(np.abs(fitted_derivative_grid_1 - exact_derivative_1)))
    derivative_linf_error_2 = float(np.max(np.abs(fitted_derivative_grid_2 - exact_derivative_2)))

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

    log_floor = np.finfo(float).tiny
    objective_log = np.maximum(np.abs(objective_history), log_floor)
    l2_log = np.maximum(l2_history, log_floor)
    linf_log = np.maximum(linf_history, log_floor)

    convergence_plot_path = OUTPUT_DIR / "uniform_quadratic_convergence.png"
    solution_plot_path = OUTPUT_DIR / "uniform_quadratic_solution.png"
    distribution_plot_path = OUTPUT_DIR / "uniform_quadratic_transformed_distribution.png"
    summary_path = OUTPUT_DIR / "uniform_quadratic_summary.txt"

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
    axes[0, 1].set_title("Error Convergence to Exact Map")
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

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].plot(grid_1, exact_solution_1, label="Exact map", color="black", linestyle="--", linewidth=2)
    axes[0, 0].plot(grid_1, initial_grid_1, label="Initial spline", color="#c1121f", alpha=0.8, linewidth=2)
    axes[0, 0].plot(grid_1, fitted_grid_1, label="Optimized spline", color="#003049", linewidth=2.5)
    axes[0, 0].set_title("Solution Map T1")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("T1(x1)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(grid_2, exact_solution_2, label="Exact map", color="black", linestyle="--", linewidth=2)
    axes[0, 1].plot(grid_2, initial_grid_2, label="Initial spline", color="#c1121f", alpha=0.8, linewidth=2)
    axes[0, 1].plot(grid_2, fitted_grid_2, label="Optimized spline", color="#003049", linewidth=2.5)
    axes[0, 1].set_title("Solution Map T2")
    axes[0, 1].set_xlabel("x2")
    axes[0, 1].set_ylabel("T2(x2)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(grid_1, exact_derivative_1, label="Exact derivative", color="black", linestyle="--", linewidth=2)
    axes[1, 0].plot(grid_1, fitted_derivative_grid_1, label="Optimized derivative", color="#669bbc", linewidth=2.5)
    axes[1, 0].set_title("Derivative T1'")
    axes[1, 0].set_xlabel("x1")
    axes[1, 0].set_ylabel("T1'(x1)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(grid_2, exact_derivative_2, label="Exact derivative", color="black", linestyle="--", linewidth=2)
    axes[1, 1].plot(grid_2, fitted_derivative_grid_2, label="Optimized derivative", color="#669bbc", linewidth=2.5)
    axes[1, 1].set_title("Derivative T2'")
    axes[1, 1].set_xlabel("x2")
    axes[1, 1].set_ylabel("T2'(x2)")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(solution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    transformed_1 = bspline_approach.transform(sample_values_1, coeff_blocks[0], transform_index=0)
    transformed_2 = bspline_approach.transform(sample_values_2, coeff_blocks[1], transform_index=1)

    exact_transformed = np.column_stack(
        [
            exact_coordinate_map(config.support_1, sample_values_1),
            exact_coordinate_map(config.support_2, sample_values_2),
        ]
    )

    all_x = np.concatenate([sample_values_1, transformed_1, exact_transformed[:, 0]])
    all_y = np.concatenate([sample_values_2, transformed_2, exact_transformed[:, 1]])

    x_pad = 0.05 * (np.max(all_x) - np.min(all_x))
    y_pad = 0.05 * (np.max(all_y) - np.min(all_y))
    x_limits = (np.min(all_x) - x_pad, np.max(all_x) + x_pad)
    y_limits = (np.min(all_y) - y_pad, np.max(all_y) + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].scatter(sample_values_1, sample_values_2, s=10, alpha=0.35, color="#6c757d")
    axes[0].set_title("Input Uniform Samples")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_xlim(x_limits)
    axes[0].set_ylim(y_limits)
    axes[0].grid(alpha=0.2)
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(transformed_1, transformed_2, s=10, alpha=0.35, color="#0a9396")
    axes[1].set_title("Optimized Transformed Samples")
    axes[1].set_xlabel("T1(x1)")
    axes[1].set_ylabel("T2(x2)")
    axes[1].set_xlim(x_limits)
    axes[1].set_ylim(y_limits)
    axes[1].grid(alpha=0.2)
    axes[1].set_aspect("equal", adjustable="box")

    axes[2].scatter(exact_transformed[:, 0], exact_transformed[:, 1], s=10, alpha=0.35, color="#9c6644")
    axes[2].set_title("Exact Transformed Samples")
    axes[2].set_xlabel("T1*(x1)")
    axes[2].set_ylabel("T2*(x2)")
    axes[2].set_xlim(x_limits)
    axes[2].set_ylim(y_limits)
    axes[2].grid(alpha=0.2)
    axes[2].set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(distribution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    potential_derivative = lambda s: s
    stationary_tests = [
        ("x", lambda y: y, lambda y: np.ones_like(y)),
        ("x^2", lambda y: y ** 2, lambda y: 2.0 * y),
        ("x^3", lambda y: y ** 3, lambda y: 3.0 * y ** 2),
        ("exp(x)", lambda y: np.exp(y), lambda y: np.exp(y)),
    ]

    stationary_results = []
    for name, test_function, test_function_derivative in stationary_tests:
        residuals = compute_stationary_residuals(
            bspline_approach,
            result.x,
            config.number_transforms,
            potential_derivative,
            test_function,
            test_function_derivative,
        )
        passed = bool(np.all(np.abs(residuals) <= config.stationary_tolerance))
        stationary_results.append((name, residuals, passed))

    sample_points_1 = diagnostic_points(config.domain_1)
    sample_points_2 = diagnostic_points(config.domain_2)

    sample_solution_values_1 = bspline_approach.transform(sample_points_1, coeff_blocks[0], transform_index=0)
    sample_solution_values_2 = bspline_approach.transform(sample_points_2, coeff_blocks[1], transform_index=1)
    sample_solution_derivatives_1 = bspline_approach.transform_derivative(
        sample_points_1,
        coeff_blocks[0],
        transform_index=0,
    )
    sample_solution_derivatives_2 = bspline_approach.transform_derivative(
        sample_points_2,
        coeff_blocks[1],
        transform_index=1,
    )

    exact_sample_values_1 = exact_coordinate_map(config.support_1, sample_points_1)
    exact_sample_values_2 = exact_coordinate_map(config.support_2, sample_points_2)

    summary_lines = [
        "Independent quadratic uniform B-spline evaluation",
        "",
        "Exact transformations:",
        "  T1(x) = Phi^{-1}((x-a1)/(b1-a1))",
        "  T2(x) = Phi^{-1}((x-a2)/(b2-a2))",
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
        f"Uniform support x1: {config.support_1}",
        f"Uniform support x2: {config.support_2}",
        f"Spline domain x1: {config.domain_1}",
        f"Spline domain x2: {config.domain_2}",
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
        "Stationary-condition checks:",
    ]

    for name, residuals, passed in stationary_results:
        summary_lines.append(
            f"  {name}: residuals={np.array2string(residuals, precision=6, separator=', ')}, "
            f"passed={passed}"
        )

    summary_lines.extend(
        [
            "",
            "Solution values at sample points for T1:",
        ]
    )
    for x_value, t_value, t_exact, dt_value in zip(
        sample_points_1,
        sample_solution_values_1,
        exact_sample_values_1,
        sample_solution_derivatives_1,
    ):
        summary_lines.append(
            f"  x1={x_value: .4f} -> T1(x1)={t_value: .8f}, exact={t_exact: .8f}, T1'(x1)={dt_value: .8f}"
        )

    summary_lines.append("")
    summary_lines.append("Solution values at sample points for T2:")
    for x_value, t_value, t_exact, dt_value in zip(
        sample_points_2,
        sample_solution_values_2,
        exact_sample_values_2,
        sample_solution_derivatives_2,
    ):
        summary_lines.append(
            f"  x2={x_value: .4f} -> T2(x2)={t_value: .8f}, exact={t_exact: .8f}, T2'(x2)={dt_value: .8f}"
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
        "samples": filtered_points,
        "grid_1": grid_1,
        "grid_2": grid_2,
        "output_dir": OUTPUT_DIR,
    }


def run_piecewise_transformation(config: RunConfigPiecewise):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_points = sample_piecewise_distribution(
        config.num_samples,
        random_state=config.random_state,
    )

    in_domain_mask = (
        (raw_points[:, 0] >= config.domain_1[0])
        & (raw_points[:, 0] <= config.domain_1[1])
        & (raw_points[:, 1] >= config.domain_2[0])
        & (raw_points[:, 1] <= config.domain_2[1])
        & (raw_points[:, 2] >= config.domain_3[0])
        & (raw_points[:, 2] <= config.domain_3[1])
    )
    filtered_points = raw_points[in_domain_mask]

    if filtered_points.shape[0] == 0:
        raise ValueError("No samples remained inside the chosen spline domains.")

    problem = None
    samples = MonteCarloSamples(points=filtered_points)

    knots_1 = open_uniform_knots(
        a=config.domain_1[0],
        b=config.domain_1[1],
        degree=config.degree,
        n_internal=config.num_internal_knots_1,
    )
    knots_2 = open_uniform_knots(
        a=config.domain_2[0],
        b=config.domain_2[1],
        degree=config.degree,
        n_internal=config.num_internal_knots_2,
    )
    knots_3 = open_uniform_knots(
        a=config.domain_3[0],
        b=config.domain_3[1],
        degree=config.degree,
        n_internal=config.num_internal_knots_3,
    )

    spline_configs = [
        BSplineConfig(degree=config.degree, knots=knots_1, domain=config.domain_1),
        BSplineConfig(degree=config.degree, knots=knots_2, domain=config.domain_2),
        BSplineConfig(degree=config.degree, knots=knots_3, domain=config.domain_3),
    ]

    bspline_approach = BSplineApproach(
        problem=problem,
        config=spline_configs,
        samples=samples,
    )

    initial_coefficients_1 = affine_normal_range_initial_coefficients(knots_1, config.degree)
    initial_coefficients_2 = affine_normal_range_initial_coefficients(knots_2, config.degree)
    initial_coefficients_3 = affine_normal_range_initial_coefficients(knots_3, config.degree)
    initial_guess = np.concatenate(
        [initial_coefficients_1, initial_coefficients_2, initial_coefficients_3]
    )

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
    sample_values_3 = filtered_points[:, 2]

    sample_min_1 = float(np.min(sample_values_1))
    sample_max_1 = float(np.max(sample_values_1))
    sample_min_2 = float(np.min(sample_values_2))
    sample_max_2 = float(np.max(sample_values_2))
    sample_min_3 = float(np.min(sample_values_3))
    sample_max_3 = float(np.max(sample_values_3))

    grid_1 = np.linspace(sample_min_1, sample_max_1, config.grid_size)
    grid_2 = np.linspace(sample_min_2, sample_max_2, config.grid_size)
    grid_3 = np.linspace(sample_min_3, sample_max_3, config.grid_size)

    c1, c2, c3 = config.constants
    exact_solution_1 = piecewise_coordinate_map(c1, grid_1)
    exact_solution_2 = piecewise_coordinate_map(c2, grid_2)
    exact_solution_3 = piecewise_coordinate_map(c3, grid_3)

    exact_derivative_1 = piecewise_coordinate_derivative(c1, grid_1)
    exact_derivative_2 = piecewise_coordinate_derivative(c2, grid_2)
    exact_derivative_3 = piecewise_coordinate_derivative(c3, grid_3)

    coeff_blocks = bspline_approach._split_coefficients(result.x, config.number_transforms)

    fitted_grid_1 = bspline_approach.transform(grid_1, coeff_blocks[0], transform_index=0)
    fitted_grid_2 = bspline_approach.transform(grid_2, coeff_blocks[1], transform_index=1)
    fitted_grid_3 = bspline_approach.transform(grid_3, coeff_blocks[2], transform_index=2)

    fitted_derivative_grid_1 = bspline_approach.transform_derivative(grid_1, coeff_blocks[0], transform_index=0)
    fitted_derivative_grid_2 = bspline_approach.transform_derivative(grid_2, coeff_blocks[1], transform_index=1)
    fitted_derivative_grid_3 = bspline_approach.transform_derivative(grid_3, coeff_blocks[2], transform_index=2)

    initial_grid_1 = bspline_approach.transform(grid_1, initial_coefficients_1, transform_index=0)
    initial_grid_2 = bspline_approach.transform(grid_2, initial_coefficients_2, transform_index=1)
    initial_grid_3 = bspline_approach.transform(grid_3, initial_coefficients_3, transform_index=2)

    l2_error_1 = float(np.sqrt(np.mean((fitted_grid_1 - exact_solution_1) ** 2)))
    l2_error_2 = float(np.sqrt(np.mean((fitted_grid_2 - exact_solution_2) ** 2)))
    l2_error_3 = float(np.sqrt(np.mean((fitted_grid_3 - exact_solution_3) ** 2)))
    linf_error_1 = float(np.max(np.abs(fitted_grid_1 - exact_solution_1)))
    linf_error_2 = float(np.max(np.abs(fitted_grid_2 - exact_solution_2)))
    linf_error_3 = float(np.max(np.abs(fitted_grid_3 - exact_solution_3)))

    derivative_linf_error_1 = float(np.max(np.abs(fitted_derivative_grid_1 - exact_derivative_1)))
    derivative_linf_error_2 = float(np.max(np.abs(fitted_derivative_grid_2 - exact_derivative_2)))
    derivative_linf_error_3 = float(np.max(np.abs(fitted_derivative_grid_3 - exact_derivative_3)))

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

    l2_history = []
    linf_history = []
    for coeffs in coefficient_history:
        coeff_blocks_iter = bspline_approach._split_coefficients(coeffs, config.number_transforms)
        fitted_grid_iter_1 = bspline_approach.transform(grid_1, coeff_blocks_iter[0], transform_index=0)
        fitted_grid_iter_2 = bspline_approach.transform(grid_2, coeff_blocks_iter[1], transform_index=1)
        fitted_grid_iter_3 = bspline_approach.transform(grid_3, coeff_blocks_iter[2], transform_index=2)

        l2_iter_1 = np.sqrt(np.mean((fitted_grid_iter_1 - exact_solution_1) ** 2))
        l2_iter_2 = np.sqrt(np.mean((fitted_grid_iter_2 - exact_solution_2) ** 2))
        l2_iter_3 = np.sqrt(np.mean((fitted_grid_iter_3 - exact_solution_3) ** 2))

        linf_iter_1 = np.max(np.abs(fitted_grid_iter_1 - exact_solution_1))
        linf_iter_2 = np.max(np.abs(fitted_grid_iter_2 - exact_solution_2))
        linf_iter_3 = np.max(np.abs(fitted_grid_iter_3 - exact_solution_3))

        l2_history.append(float(max(l2_iter_1, l2_iter_2, l2_iter_3)))
        linf_history.append(float(max(linf_iter_1, linf_iter_2, linf_iter_3)))

    objective_history = np.asarray(objective_history, dtype=float)
    l2_history = np.asarray(l2_history, dtype=float)
    linf_history = np.asarray(linf_history, dtype=float)
    error_iterations = np.arange(1, len(l2_history) + 1)

    log_floor = np.finfo(float).tiny
    objective_log = np.maximum(np.abs(objective_history), log_floor)
    l2_log = np.maximum(l2_history, log_floor)
    linf_log = np.maximum(linf_history, log_floor)

    convergence_plot_path = OUTPUT_DIR / "piecewise_transformation_convergence.png"
    solution_plot_path = OUTPUT_DIR / "piecewise_transformation_solution.png"
    distribution_plot_path = OUTPUT_DIR / "piecewise_transformation_distribution.png"
    summary_path = OUTPUT_DIR / "piecewise_transformation_summary.txt"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    axes[0, 0].plot(objective_iterations, objective_history, color="#154c79", linewidth=2, marker="o", markersize=4)
    axes[0, 0].set_title("Objective Convergence")
    axes[0, 0].set_xlabel("Objective evaluation")
    axes[0, 0].set_ylabel("Empirical objective")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(error_iterations, l2_history, label="max coord L2", color="#b7410e", linewidth=2, marker="o", markersize=4)
    axes[0, 1].plot(error_iterations, linf_history, label="max coord Linf", color="#2d6a4f", linewidth=2, marker="o", markersize=4)
    axes[0, 1].set_title("Error Convergence to Exact Piecewise Map")
    axes[0, 1].set_xlabel("Accepted iterate")
    axes[0, 1].set_ylabel("Error on sample intervals")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].semilogy(objective_iterations, objective_log, color="#154c79", linewidth=2, marker="o", markersize=4)
    axes[1, 0].set_title("Objective Convergence (log scale)")
    axes[1, 0].set_xlabel("Objective evaluation")
    axes[1, 0].set_ylabel("Absolute objective")
    axes[1, 0].grid(alpha=0.3, which="both")

    axes[1, 1].semilogy(error_iterations, l2_log, label="max coord L2", color="#b7410e", linewidth=2, marker="o", markersize=4)
    axes[1, 1].semilogy(error_iterations, linf_log, label="max coord Linf", color="#2d6a4f", linewidth=2, marker="o", markersize=4)
    axes[1, 1].set_title("Error Convergence (log scale)")
    axes[1, 1].set_xlabel("Accepted iterate")
    axes[1, 1].set_ylabel("Error on sample intervals")
    axes[1, 1].grid(alpha=0.3, which="both")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(convergence_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    plot_triplets = [
        (grid_1, exact_solution_1, initial_grid_1, fitted_grid_1, exact_derivative_1, fitted_derivative_grid_1, "T1", "x1"),
        (grid_2, exact_solution_2, initial_grid_2, fitted_grid_2, exact_derivative_2, fitted_derivative_grid_2, "T2", "x2"),
        (grid_3, exact_solution_3, initial_grid_3, fitted_grid_3, exact_derivative_3, fitted_derivative_grid_3, "T3", "x3"),
    ]

    for row, (grid, exact_solution, initial_grid, fitted_grid, exact_derivative, fitted_derivative, label, x_label) in enumerate(plot_triplets):
        axes[row, 0].plot(grid, exact_solution, label="Exact map", color="black", linestyle="--", linewidth=2)
        axes[row, 0].plot(grid, initial_grid, label="Initial spline", color="#c1121f", alpha=0.8, linewidth=2)
        axes[row, 0].plot(grid, fitted_grid, label="Optimized spline", color="#003049", linewidth=2.5)
        axes[row, 0].set_title(f"Solution Map {label}")
        axes[row, 0].set_xlabel(x_label)
        axes[row, 0].set_ylabel(f"{label}({x_label})")
        axes[row, 0].grid(alpha=0.3)
        axes[row, 0].legend()

        axes[row, 1].plot(grid, exact_derivative, label="Exact derivative", color="black", linestyle="--", linewidth=2)
        axes[row, 1].plot(grid, fitted_derivative, label="Optimized derivative", color="#669bbc", linewidth=2.5)
        axes[row, 1].set_title(f"Derivative {label}'")
        axes[row, 1].set_xlabel(x_label)
        axes[row, 1].set_ylabel(f"{label}'({x_label})")
        axes[row, 1].grid(alpha=0.3)
        axes[row, 1].legend()

    fig.tight_layout()
    fig.savefig(solution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    transformed_1 = bspline_approach.transform(sample_values_1, coeff_blocks[0], transform_index=0)
    transformed_2 = bspline_approach.transform(sample_values_2, coeff_blocks[1], transform_index=1)
    transformed_3 = bspline_approach.transform(sample_values_3, coeff_blocks[2], transform_index=2)

    exact_transformed = np.column_stack(
        [
            piecewise_coordinate_map(c1, sample_values_1),
            piecewise_coordinate_map(c2, sample_values_2),
            piecewise_coordinate_map(c3, sample_values_3),
        ]
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    scatter_pairs = [
        (sample_values_1, sample_values_2, transformed_1, transformed_2, exact_transformed[:, 0], exact_transformed[:, 1], "12"),
        (sample_values_1, sample_values_3, transformed_1, transformed_3, exact_transformed[:, 0], exact_transformed[:, 2], "13"),
        (sample_values_2, sample_values_3, transformed_2, transformed_3, exact_transformed[:, 1], exact_transformed[:, 2], "23"),
    ]

    for col, (raw_x, raw_y, fit_x, fit_y, exact_x, exact_y, label) in enumerate(scatter_pairs):
        axes[0, col].scatter(raw_x, raw_y, s=8, alpha=0.3, color="#6c757d")
        axes[0, col].set_title(f"Input Samples ({label})")
        axes[0, col].grid(alpha=0.2)

        axes[1, col].scatter(fit_x, fit_y, s=8, alpha=0.3, color="#0a9396", label="Optimized")
        axes[1, col].scatter(exact_x, exact_y, s=8, alpha=0.2, color="#9c6644", label="Exact")
        axes[1, col].set_title(f"Transformed Samples ({label})")
        axes[1, col].grid(alpha=0.2)
        axes[1, col].legend()

    fig.tight_layout()
    fig.savefig(distribution_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    potential_derivative = lambda s: s
    stationary_tests = [
        ("x", lambda y: y, lambda y: np.ones_like(y)),
        ("x^2", lambda y: y ** 2, lambda y: 2.0 * y),
        ("x^3", lambda y: y ** 3, lambda y: 3.0 * y ** 2),
    ]

    stationary_results = []
    for name, test_function, test_function_derivative in stationary_tests:
        residuals = compute_stationary_residuals(
            bspline_approach,
            result.x,
            config.number_transforms,
            potential_derivative,
            test_function,
            test_function_derivative,
        )
        passed = bool(np.all(np.abs(residuals) <= config.stationary_tolerance))
        stationary_results.append((name, residuals, passed))

    sample_points = diagnostic_points(config.domain_1)
    sample_solution_values = [
        bspline_approach.transform(sample_points, coeff_blocks[j], transform_index=j)
        for j in range(3)
    ]
    sample_solution_derivatives = [
        bspline_approach.transform_derivative(sample_points, coeff_blocks[j], transform_index=j)
        for j in range(3)
    ]
    exact_sample_values = [
        piecewise_coordinate_map(config.constants[j], sample_points)
        for j in range(3)
    ]

    summary_lines = [
        "Piecewise-transformation B-spline evaluation",
        "",
        "Exact transformations:",
        "  Ti(x) = -ci + Phi^{-1}((1 + x) Phi(ci)) for x < 0",
        "  Ti(x) =  ci - Phi^{-1}((1 - x) Phi(ci)) for x >= 0",
        "",
        f"Constants: c1={c1:.4f}, c2={c2:.4f}, c3={c3:.4f}",
        f"Runtime (seconds): {runtime_seconds:.6f}",
        f"Optimizer backend: {config.optimizer_backend}",
        f"Optimizer success: {result.success}",
        f"Optimizer status: {result.status}",
        f"Optimizer message: {result.message}",
        f"Minimum derivative lower bound: {config.minimum_derivative:.2e}",
        f"Objective evaluations recorded: {len(objective_history)}",
        f"Accepted iterates recorded: {len(coefficient_history)}",
        f"Number of in-domain Monte Carlo samples: {filtered_points.shape[0]}",
        f"Domain x1: {config.domain_1}",
        f"Domain x2: {config.domain_2}",
        f"Domain x3: {config.domain_3}",
        f"Final empirical objective: {final_objective:.10f}",
        f"Max L2 error on sample intervals: {max(l2_error_1, l2_error_2, l2_error_3):.10e}",
        f"Max Linf error on sample intervals: {max(linf_error_1, linf_error_2, linf_error_3):.10e}",
        f"Max Linf derivative error: {max(derivative_linf_error_1, derivative_linf_error_2, derivative_linf_error_3):.10e}",
        "",
        "Stationary-condition checks:",
    ]

    for name, residuals, passed in stationary_results:
        summary_lines.append(
            f"  {name}: residuals={np.array2string(residuals, precision=6, separator=', ')}, "
            f"passed={passed}"
        )

    for j in range(3):
        summary_lines.append("")
        summary_lines.append(f"Solution values at sample points for T{j + 1}:")
        for x_value, t_value, t_exact, dt_value in zip(
            sample_points,
            sample_solution_values[j],
            exact_sample_values[j],
            sample_solution_derivatives[j],
        ):
            summary_lines.append(
                f"  x={x_value: .4f} -> T{j + 1}(x)={t_value: .8f}, "
                f"exact={t_exact: .8f}, T{j + 1}'(x)={dt_value: .8f}"
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
        "samples": filtered_points,
        "grid_1": grid_1,
        "grid_2": grid_2,
        "grid_3": grid_3,
        "output_dir": OUTPUT_DIR,
    }


if __name__ == "__main__":
    # config = RunConfigUniform()
    # run_uniform_quadratic(config)

    config = RunConfigPiecewise()
    run_piecewise_transformation(config)
