from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
from scipy.stats import expon, norm

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


toy_module = load_module("toy_independent_module", TOY_FILE)
bspline_module = load_module("bspline_module", BSPLINE_FILE)

IndependentJointToyProblem = toy_module.IndependentJointToyProblem
AbsoluteValuePotentialIndependentToyProblem = toy_module.AbsoluteValuePotentialIndependentToyProblem
ScipyDistribution = toy_module.ScipyDistribution
STANDARD_NORMAL = toy_module.STANDARD_NORMAL
STANDARD_LOGISTIC = toy_module.STANDARD_LOGISTIC

BSplineConfig = bspline_module.BSplineConfig
MonteCarloSamples = bspline_module.MonteCarloSamples
BSplineApproach = bspline_module.BSplineApproach


@dataclass
class RunConfigIndependent:
    num_samples: int = 15000
    random_state: int = 7
    number_transforms: int = 2
    optimizer_backend: str = "cvxpy"
    potential_name: str = "quadratic_sum"
    degree: int = 3
    domain_1: tuple[float, float] = (0.0, 3.5)
    domain_2: tuple[float, float] = (0.0, 10.0)
    num_internal_knots_1: int = 16
    num_internal_knots_2: int = 20
    maxiter: int = 250
    minimum_derivative: float = 1e-5
    grid_size: int = 400
    histogram_bins: int = 45
    stationary_tolerance: float = 1e-3


def knot_average(knots: np.ndarray, degree: int) -> np.ndarray:
    """Greville abscissae for a simple monotone initial guess."""
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
                    # SciPy callback history stores coefficient vectors under
                    # `iteration`, but SCS stores plain iteration numbers there.
                    coefficients = iteration_values

            objective = candidate.get("objective", candidate.get("objective_value", []))
            return {
                "coefficients": list(coefficients),
                "objective": list(objective),
                "monotonicity_satisfied": list(candidate.get("monotonicity_satisfied", [])),
                "min_derivatives": list(candidate.get("min_derivatives", [])),
            }

    return {}


def exact_coordinate_map_with_target(marginal, target_marginal, x: np.ndarray) -> np.ndarray:
    """Exact map T(x) = G^{-1}(F(x)) for the chosen target marginal G."""
    probabilities = marginal.cdf(np.asarray(x, dtype=float))
    probabilities = np.clip(probabilities, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    return target_marginal.ppf(probabilities)


def exact_coordinate_derivative_with_target(marginal, target_marginal, x: np.ndarray) -> np.ndarray:
    """Derivative of the generic coordinate map G^{-1}(F(x))."""
    x = np.asarray(x, dtype=float)
    transformed = exact_coordinate_map_with_target(marginal, target_marginal, x)
    numerator = marginal.pdf(x)
    denominator = target_marginal.pdf(transformed)
    return numerator / denominator


def exact_map_initial_coefficients(marginal, target_marginal, knots: np.ndarray, degree: int) -> np.ndarray:
    """
    Build a strong monotone warm start by evaluating the known exact map at the
    Greville abscissae of the spline basis.
    """
    greville_points = knot_average(knots, degree)
    return exact_coordinate_map_with_target(marginal, target_marginal, greville_points)


def diagnostic_points(interval: tuple[float, float], count: int = 5) -> np.ndarray:
    """Pick readable interior points away from the boundary."""
    left, right = interval
    padding = 0.1 * (right - left)
    return np.linspace(left + padding, right - padding, count, dtype=float)


def compute_stationary_residuals(
    bspline_approach,
    flat_coefficients: np.ndarray,
    number_transforms: int,
    potential_function_derivative,
    test_function,
    test_function_derivative,
):
    """Get residuals without raising, so we can summarize them cleanly."""
    return bspline_approach.check_stationary_condition(
        flat_coefficients,
        number_transforms,
        potential_function_derivative,
        test_function,
        test_function_derivative,
        tolerance=np.inf,
    )


def build_problem_and_target(config: RunConfigIndependent, marginal_1, marginal_2):
    """Choose the toy problem class and the corresponding exact target marginal."""
    if config.potential_name == "quadratic_sum":
        problem = IndependentJointToyProblem(
            marginals=(marginal_1, marginal_2),
            target_marginal=STANDARD_NORMAL,
            name="independent_quadratic_exponential",
            potential_name="quadratic_sum",
            description=(
                "Independent exponential marginals with rates 2 and 0.5 under the "
                "quadratic potential. Exact coordinate-wise map is Phi^{-1}(F_i)."
            ),
        )
        target_marginal = STANDARD_NORMAL
        exact_map_label = "Phi^{-1}(F_i)"
    elif config.potential_name == "absolute_sum":
        problem = AbsoluteValuePotentialIndependentToyProblem(
            marginals=(marginal_1, marginal_2),
        )
        target_marginal = STANDARD_LOGISTIC
        exact_map_label = "G^{-1}(F_i)"
    else:
        raise ValueError(
            "Unsupported potential for this evaluator. "
            "Use `quadratic_sum` or `absolute_sum`."
        )

    return problem, target_marginal, exact_map_label


def potential_derivative_function(potential_name: str):
    """
    Return the one-dimensional derivative/subgradient of psi(s) where the
    potential has the form V(y) = psi(y_1 + ... + y_n).
    """
    if potential_name == "quadratic_sum":
        return lambda s: s
    if potential_name == "absolute_sum":
        # We use np.sign as the natural subgradient away from zero.
        return lambda s: np.sign(s)
    raise ValueError("Unsupported potential. Use `quadratic_sum` or `absolute_sum`.")


def run_independent_case(config: RunConfigIndependent):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    marginal_1 = ScipyDistribution("exponential_rate_2", expon(scale=0.5))
    marginal_2 = ScipyDistribution("exponential_rate_0.5", expon(scale=2.0))
    problem, target_marginal, exact_map_label = build_problem_and_target(
        config,
        marginal_1,
        marginal_2,
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
        potential_name=config.potential_name,
    )

    initial_coefficients_1 = exact_map_initial_coefficients(
        marginal_1,
        target_marginal,
        knots_1,
        config.degree,
    )
    initial_coefficients_2 = exact_map_initial_coefficients(
        marginal_2,
        target_marginal,
        knots_2,
        config.degree,
    )
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

    exact_solution_1 = exact_coordinate_map_with_target(marginal_1, target_marginal, grid_1)
    exact_solution_2 = exact_coordinate_map_with_target(marginal_2, target_marginal, grid_2)
    exact_derivative_1 = exact_coordinate_derivative_with_target(marginal_1, target_marginal, grid_1)
    exact_derivative_2 = exact_coordinate_derivative_with_target(marginal_2, target_marginal, grid_2)

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

    output_stem = f"independent_{config.potential_name}"
    convergence_plot_path = OUTPUT_DIR / f"{output_stem}_convergence.png"
    solution_plot_path = OUTPUT_DIR / f"{output_stem}_solution.png"
    distribution_plot_path = OUTPUT_DIR / f"{output_stem}_transformed_distribution.png"
    summary_path = OUTPUT_DIR / f"{output_stem}_summary.txt"

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

    exact_transformed = np.asarray(
        problem.exact_transformation(filtered_points),
        dtype=float,
    )

    all_x = np.concatenate([sample_values_1, transformed_1, exact_transformed[:, 0]])
    all_y = np.concatenate([sample_values_2, transformed_2, exact_transformed[:, 1]])

    x_pad = 0.05 * (np.max(all_x) - np.min(all_x))
    y_pad = 0.05 * (np.max(all_y) - np.min(all_y))
    x_limits = (np.min(all_x) - x_pad, np.max(all_x) + x_pad)
    y_limits = (np.min(all_y) - y_pad, np.max(all_y) + y_pad)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].scatter(sample_values_1, sample_values_2, s=10, alpha=0.35, color="#6c757d")
    axes[0].set_title("Input Independent Samples")
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

    potential_derivative = potential_derivative_function(config.potential_name)
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

    exact_sample_values_1 = exact_coordinate_map_with_target(marginal_1, target_marginal, sample_points_1)
    exact_sample_values_2 = exact_coordinate_map_with_target(marginal_2, target_marginal, sample_points_2)

    summary_lines = [
        "Independent exponential B-spline evaluation",
        "",
        f"Potential: {config.potential_name}",
        f"Exact coordinate map: {exact_map_label}",
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


def run_independent_quadratic(config: RunConfigIndependent):
    """Backward-compatible entry point."""
    return run_independent_case(config)


if __name__ == "__main__":
    config = RunConfigIndependent()
    run_independent_case(config)
