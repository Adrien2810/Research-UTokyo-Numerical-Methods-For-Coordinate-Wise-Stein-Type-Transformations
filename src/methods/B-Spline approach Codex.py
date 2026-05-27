from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Literal

import numpy as np
from scipy.interpolate import BSpline


ArrayLike = np.ndarray


@dataclass
class BSplineConfig:
    """Configuration of a one-dimensional B-spline ansatz."""

    degree: int 
    knots: ArrayLike
    domain: tuple[float, float]

    @property
    def n_basis(self) -> int:
        return len(self.knots) - self.degree - 1

    def contains(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        a, b = self.domain
        return (x >= a) & (x <= b)


@dataclass
class MonteCarloSamples:
    """Fixed samples drawn from the input density f."""

    points: ArrayLike

    @property
    def n_samples(self) -> int:
        return int(self.points.shape[0])


class BSplineBasis:
    """Helper for evaluating a scalar B-spline basis and its derivative."""

    def __init__(self, config: BSplineConfig):
        self.config = config
        if self.config.n_basis <= 0:
            raise ValueError("Invalid knot vector: no basis functions available.")
        self._basis_splines = []
        self._derivative_splines = []

        for j in range(self.config.n_basis):
            coeffs = np.zeros(self.config.n_basis, dtype=float)
            coeffs[j] = 1.0
            spline = BSpline(self.config.knots, coeffs, self.config.degree, extrapolate=False)
            self._basis_splines.append(spline)
            self._derivative_splines.append(spline.derivative())

    def basis_matrix(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        matrix = np.zeros((x.size, self.config.n_basis), dtype=float)

        for j in range(self.config.n_basis):
            matrix[:, j] = self._basis_splines[j](x)

        return matrix

    def derivative_matrix(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        matrix = np.zeros((x.size, self.config.n_basis), dtype=float)

        for j in range(self.config.n_basis):
            matrix[:, j] = self._derivative_splines[j](x)

        return matrix


class CoordinateMap:
    """Coordinate-wise map T_i(x) = sum_j c_j B_j(x)."""

    def __init__(self, basis: BSplineBasis, coefficients: Optional[ArrayLike] = None):
        self.basis = basis
        n_basis = self.basis.config.n_basis
        self.coefficients = np.zeros(n_basis, dtype=float) if coefficients is None else np.asarray(coefficients, dtype=float)

        if self.coefficients.shape != (n_basis,):
            raise ValueError(f"Expected coefficient vector of shape ({n_basis},), got {self.coefficients.shape}.")

    def evaluate(self, x: ArrayLike) -> ArrayLike:
        return self.basis.basis_matrix(x) @ self.coefficients

    def derivative(self, x: ArrayLike) -> ArrayLike:
        return self.basis.derivative_matrix(x) @ self.coefficients


class BSplineMonteCarloApproach:
    """
    Monte-Carlo scaffold for the variational problem.

    The idea is to replace integrals under f by empirical averages over a fixed
    sample x^(m) ~ f, then optimize over the spline coefficients.

    The bounded-box spline approximation is explicit in this class: by default
    all Monte Carlo samples must lie inside the spline domains. If you want to
    work with a truncated empirical measure instead, choose `domain_mode="restrict"`.
    """

    def __init__(
        self,
        problem,
        basis_1: BSplineBasis,
        basis_2: BSplineBasis,
        samples: MonteCarloSamples,
    ):
        self.problem = problem
        self.basis_1 = basis_1
        self.basis_2 = basis_2
        self.samples = samples

        if self.samples.points.ndim != 2 or self.samples.points.shape[1] != 2:
            raise ValueError("This scaffold currently expects samples of shape (n_samples, 2).")

    def sample_domain_report(self) -> dict:
        x_1 = self.samples.points[:, 0]
        x_2 = self.samples.points[:, 1]
        in_domain_1 = self.basis_1.config.contains(x_1)
        in_domain_2 = self.basis_2.config.contains(x_2)
        in_domain = in_domain_1 & in_domain_2

        return {
            "n_samples": int(self.samples.n_samples),
            "n_in_domain": int(np.sum(in_domain)),
            "n_out_of_domain": int(np.sum(~in_domain)),
            "fraction_in_domain": float(np.mean(in_domain)),
            "domain_1": tuple(float(v) for v in self.basis_1.config.domain),
            "domain_2": tuple(float(v) for v in self.basis_2.config.domain),
        }

    def _active_sample_mask(self) -> ArrayLike:
        x_1 = self.samples.points[:, 0]
        x_2 = self.samples.points[:, 1]
        return self.basis_1.config.contains(x_1) & self.basis_2.config.contains(x_2)

    def _validated_sample_points(self, domain_mode: Literal["raise", "restrict"]) -> ArrayLike:
        mask = self._active_sample_mask()
        if np.all(mask):
            return self.samples.points
        if domain_mode == "restrict":
            points = self.samples.points[mask]
            if points.size == 0:
                raise ValueError("No Monte Carlo samples remain after restricting to the spline domains.")
            return points

        report = self.sample_domain_report()
        raise ValueError(
            "Monte Carlo samples fall outside the spline domains. "
            f"Out-of-domain samples: {report['n_out_of_domain']} / {report['n_samples']}. "
            "Increase the domains or use domain_mode='restrict' for a truncated empirical objective."
        )

    @staticmethod
    def _potential_value(problem, transformed_points: ArrayLike) -> ArrayLike:
        if hasattr(problem, "potential"):
            return np.asarray(problem.potential(transformed_points), dtype=float)

        total = np.sum(np.asarray(transformed_points, dtype=float), axis=1)
        return 0.5 * total**2

    @staticmethod
    def _potential_sum_gradient(problem, total: ArrayLike) -> ArrayLike:
        potential_name = getattr(problem, "potential_name", "quadratic_sum")
        total = np.asarray(total, dtype=float)

        if potential_name == "quadratic_sum":
            return total
        if potential_name == "absolute_sum":
            return np.sign(total)
        raise NotImplementedError(
            "Stationary residual currently supports the 'quadratic_sum' and 'absolute_sum' potentials."
        )

    def split_coefficients(self, coefficients: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        coefficients = np.asarray(coefficients, dtype=float)
        n_1 = self.basis_1.config.n_basis
        n_2 = self.basis_2.config.n_basis

        if coefficients.shape != (n_1 + n_2,):
            raise ValueError(
                f"Expected coefficient vector of shape ({n_1 + n_2},), got {coefficients.shape}."
            )

        return coefficients[:n_1], coefficients[n_1:]

    def build_maps(self, coefficients: ArrayLike) -> tuple[CoordinateMap, CoordinateMap]:
        coeffs_1, coeffs_2 = self.split_coefficients(coefficients)
        return CoordinateMap(self.basis_1, coeffs_1), CoordinateMap(self.basis_2, coeffs_2)

    def evaluate_on_samples(self, coefficients: ArrayLike, domain_mode: Literal["raise", "restrict"] = "raise") -> dict:
        T_1, T_2 = self.build_maps(coefficients)
        points = self._validated_sample_points(domain_mode=domain_mode)
        x_1 = points[:, 0]
        x_2 = points[:, 1]

        values = {
            "points": points,
            "T1": T_1.evaluate(x_1),
            "T2": T_2.evaluate(x_2),
            "dT1": T_1.derivative(x_1),
            "dT2": T_2.derivative(x_2),
        }
        return values

    def empirical_objective(
        self,
        coefficients: ArrayLike,
        derivative_floor: float = 1e-8,
        domain_mode: Literal["raise", "restrict"] = "raise",
    ) -> float:
        """
        Empirical version of equation (19) based on sample averages.

        The additive constant E_f[log f(X)] is omitted because it does not affect
        the minimizer. Strict positivity of the derivative is enforced through the
        `derivative_floor`: if T_i' drops below that threshold, the objective
        returns +inf.
        """
        if derivative_floor <= 0.0:
            raise ValueError("derivative_floor must be strictly positive.")

        values = self.evaluate_on_samples(coefficients, domain_mode=domain_mode)
        transformed_points = np.column_stack([values["T1"], values["T2"]])

        if not np.all(np.isfinite(transformed_points)):
            return float(np.inf)

        min_derivative = min(np.min(values["dT1"]), np.min(values["dT2"]))
        if not np.isfinite(min_derivative) or min_derivative <= derivative_floor:
            return float(np.inf)

        jacobian_term = -np.mean(np.log(values["dT1"])) - np.mean(np.log(values["dT2"]))
        potential_term = np.mean(self._potential_value(self.problem, transformed_points))
        return float(jacobian_term + potential_term)

    def coefficient_differences(self, coefficients: ArrayLike) -> dict:
        coeffs_1, coeffs_2 = self.split_coefficients(coefficients)
        return {
            "T1_differences": np.diff(coeffs_1),
            "T2_differences": np.diff(coeffs_2),
        }

    def monotonicity_residual(self, coefficients: ArrayLike, eps: float = 1e-10) -> dict:
        values = self.evaluate_on_samples(coefficients, domain_mode="restrict")
        coefficient_gaps = self.coefficient_differences(coefficients)
        min_gap_1 = np.min(coefficient_gaps["T1_differences"]) if coefficient_gaps["T1_differences"].size else np.inf
        min_gap_2 = np.min(coefficient_gaps["T2_differences"]) if coefficient_gaps["T2_differences"].size else np.inf
        return {
            "min_dT1": float(np.min(values["dT1"])),
            "min_dT2": float(np.min(values["dT2"])),
            "T1_monotone_on_samples": bool(np.all(values["dT1"] > eps)),
            "T2_monotone_on_samples": bool(np.all(values["dT2"] > eps)),
            "min_T1_coefficient_gap": float(min_gap_1),
            "min_T2_coefficient_gap": float(min_gap_2),
            "T1_coefficients_non_decreasing": bool(np.all(coefficient_gaps["T1_differences"] >= 0.0)),
            "T2_coefficients_non_decreasing": bool(np.all(coefficient_gaps["T2_differences"] >= 0.0)),
        }

    def stationary_condition_residual(
        self,
        coefficients: ArrayLike,
        test_function: Callable[[ArrayLike], ArrayLike],
        test_function_derivative: Callable[[ArrayLike], ArrayLike],
        domain_mode: Literal["raise", "restrict"] = "raise",
    ) -> dict:
        """
        Empirical residual of the stationary identity.

        For potentials of the form V(y) = psi(y_1 + y_2), stationarity implies
        E[u(T_i(X_i)) psi'(T_1(X_1) + T_2(X_2))] = E[u'(T_i(X_i))]
        for i = 1, 2 and smooth test functions u.
        """
        values = self.evaluate_on_samples(coefficients, domain_mode=domain_mode)
        total = values["T1"] + values["T2"]
        grad_sum = self._potential_sum_gradient(self.problem, total)

        u_T1 = np.asarray(test_function(values["T1"]), dtype=float)
        u_T2 = np.asarray(test_function(values["T2"]), dtype=float)
        du_T1 = np.asarray(test_function_derivative(values["T1"]), dtype=float)
        du_T2 = np.asarray(test_function_derivative(values["T2"]), dtype=float)

        residual_1 = np.mean(u_T1 * grad_sum) - np.mean(du_T1)
        residual_2 = np.mean(u_T2 * grad_sum) - np.mean(du_T2)

        return {
            "lhs_1": float(np.mean(u_T1 * grad_sum)),
            "rhs_1": float(np.mean(du_T1)),
            "residual_1": float(residual_1),
            "lhs_2": float(np.mean(u_T2 * grad_sum)),
            "rhs_2": float(np.mean(du_T2)),
            "residual_2": float(residual_2),
            "max_abs_residual": float(max(abs(residual_1), abs(residual_2))),
        }


def open_uniform_knots(a: float, b: float, degree: int, n_internal: int) -> ArrayLike:
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


if __name__ == "__main__":
    degree = 3
    domain = (-4.0, 4.0)
    knots = open_uniform_knots(domain[0], domain[1], degree=degree, n_internal=6)
    config = BSplineConfig(degree=degree, knots=knots, domain=domain)
    basis = BSplineBasis(config)

    rng = np.random.default_rng(0)
    raw_samples = rng.normal(size=(256, 2))
    in_domain = np.all((raw_samples >= domain[0]) & (raw_samples <= domain[1]), axis=1)
    samples = MonteCarloSamples(points=raw_samples[in_domain][:128])

    class _QuadraticProblem:
        potential_name = "quadratic_sum"

        @staticmethod
        def potential(y: ArrayLike) -> ArrayLike:
            y = np.asarray(y, dtype=float)
            return 0.5 * np.sum(y, axis=1) ** 2

    approach = BSplineMonteCarloApproach(problem=_QuadraticProblem(), basis_1=basis, basis_2=basis, samples=samples)
    identity_coefficients = np.linspace(domain[0], domain[1], config.n_basis)

    print("Monte Carlo B-spline scaffold created.")
    print(f"Degree: {degree}")
    print(f"Number of basis functions: {config.n_basis}")
    print(f"Sample size: {samples.n_samples}")
    print(f"Domain report: {approach.sample_domain_report()}")
    initial_coefficients = np.concatenate([identity_coefficients, identity_coefficients])
    print(f"Empirical objective at a simple increasing coefficient guess: {approach.empirical_objective(initial_coefficients):.6f}")
