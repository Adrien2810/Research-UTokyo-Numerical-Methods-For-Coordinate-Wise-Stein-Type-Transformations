from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

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

    def basis_matrix(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        matrix = np.zeros((x.size, self.config.n_basis), dtype=float)

        for j in range(self.config.n_basis):
            coeffs = np.zeros(self.config.n_basis, dtype=float)
            coeffs[j] = 1.0
            spline = BSpline(self.config.knots, coeffs, self.config.degree, extrapolate=False)
            matrix[:, j] = spline(x)

        return matrix

    def derivative_matrix(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        matrix = np.zeros((x.size, self.config.n_basis), dtype=float)

        for j in range(self.config.n_basis):
            coeffs = np.zeros(self.config.n_basis, dtype=float)
            coeffs[j] = 1.0
            spline = BSpline(self.config.knots, coeffs, self.config.degree, extrapolate=False)
            matrix[:, j] = spline.derivative()(x)

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

    def evaluate_on_samples(self, coefficients: ArrayLike) -> dict:
        T_1, T_2 = self.build_maps(coefficients)
        x_1 = self.samples.points[:, 0]
        x_2 = self.samples.points[:, 1]

        values = {
            "T1": T_1.evaluate(x_1),
            "T2": T_2.evaluate(x_2),
            "dT1": T_1.derivative(x_1),
            "dT2": T_2.derivative(x_2),
        }
        return values

    def empirical_objective(self, coefficients: ArrayLike) -> float:
        """
        Placeholder for the empirical version of equation (19).

        Intended structure:
        J_M(c) = const
               - mean(log T_1'(X_1))
               - mean(log T_2'(X_2))
               + 0.5 * mean((T_1(X_1) + T_2(X_2))^2)
        """

        raise NotImplementedError("Implement the Monte Carlo empirical objective here.")

    def monotonicity_residual(self, coefficients: ArrayLike, eps: float = 1e-10) -> dict:
        values = self.evaluate_on_samples(coefficients)
        return {
            "min_dT1": float(np.min(values["dT1"])),
            "min_dT2": float(np.min(values["dT2"])),
            "T1_monotone_on_samples": bool(np.all(values["dT1"] > eps)),
            "T2_monotone_on_samples": bool(np.all(values["dT2"] > eps)),
        }

    def stationary_condition_residual(
        self,
        coefficients: ArrayLike,
        test_function: Callable[[ArrayLike], ArrayLike],
        test_function_derivative: Callable[[ArrayLike], ArrayLike],
    ) -> dict:
        """
        Placeholder for the empirical version of the stationary identity.

        For i = 1 you would compare sample averages of
        u(T_1(X_1)) (T_1(X_1) + T_2(X_2))
        and
        u'(T_1(X_1)).
        """

        raise NotImplementedError("Implement the empirical stationary-condition check here.")


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
    samples = MonteCarloSamples(points=rng.normal(size=(128, 2)))

    print("Monte Carlo B-spline scaffold created.")
    print(f"Degree: {degree}")
    print(f"Number of basis functions: {config.n_basis}")
    print(f"Sample size: {samples.n_samples}")
