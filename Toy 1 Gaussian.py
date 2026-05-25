from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np
from scipy.stats import laplace, logistic, multivariate_normal, norm, t as student_t


def _ensure_2d_points(x: np.ndarray, dim: int) -> np.ndarray:
    points = np.asarray(x, dtype=float)
    if points.ndim == 1:
        if dim == 1:
            return points.reshape(-1, 1)
        if points.shape[0] != dim:
            raise ValueError(f"Expected shape ({dim},) for a single point, got {points.shape}.")
        return points.reshape(1, dim)
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"Expected array of shape (n, {dim}), got {points.shape}.")
    return points


def _restore_input_shape(values: np.ndarray, original: np.ndarray) -> np.ndarray:
    if np.asarray(original).ndim == 1:
        if values.shape[0] == 1:
            return values[0]
        return values[:, 0]
    return values


class OneDimensionalDistribution(ABC):
    """Minimal interface for 1D marginals used in the paper's toy problems."""

    name: str

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def ppf(self, q: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def rvs(self, size: int, random_state: Optional[int] = None) -> np.ndarray:
        pass


class ScipyDistribution(OneDimensionalDistribution):
    """Wrapper around a SciPy frozen distribution."""

    def __init__(self, name: str, distribution):
        self.name = name
        self.distribution = distribution

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self.distribution.pdf(x)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return self.distribution.logpdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self.distribution.cdf(x)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self.distribution.ppf(q)

    def rvs(self, size: int, random_state: Optional[int] = None) -> np.ndarray:
        return self.distribution.rvs(size=size, random_state=random_state)


STANDARD_NORMAL = ScipyDistribution("standard_normal", norm())
STANDARD_LOGISTIC = ScipyDistribution("standard_logistic", logistic())
LAPLACE = ScipyDistribution("laplace", laplace())
STUDENT_T5 = ScipyDistribution("student_t_df5", student_t(df=5))


class ToyProblem(ABC):
    """
    Base class for the toy examples listed in Section 6.1 of the manuscript.

    The class focuses on the mathematical model:
    - the input density f,
    - the potential V used in the variational problem,
    - the exact coordinate-wise Stein-type transformation when known.

    Numerical solvers should live outside this file and consume these objects.
    """

    def __init__(self, name: str, dim: int, potential_name: str, description: str):
        self.name = name
        self.dim = dim
        self.potential_name = potential_name
        self.description = description

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        pass

    @abstractmethod
    def exact_transformation(self, x: np.ndarray) -> np.ndarray:
        pass

    def potential(self, y: np.ndarray) -> np.ndarray:
        transformed = _ensure_2d_points(y, self.dim)
        if self.potential_name == "quadratic_sum":
            return 0.5 * np.sum(transformed, axis=1) ** 2
        if self.potential_name == "absolute_sum":
            return np.abs(np.sum(transformed, axis=1))
        raise NotImplementedError(f"Unknown potential '{self.potential_name}'.")

    def convergence_error(self, T_numerical, x_test: Optional[np.ndarray] = None) -> dict:
        if x_test is None:
            x_test = self.sample(1000, random_state=0)

        x_test = _ensure_2d_points(x_test, self.dim)
        T_exact = _ensure_2d_points(self.exact_transformation(x_test), self.dim)
        T_num = _ensure_2d_points(T_numerical(x_test), self.dim)
        diff = T_exact - T_num
        pointwise = np.linalg.norm(diff, axis=1)

        return {
            "l2_error": float(np.sqrt(np.mean(pointwise**2))),
            "linf_error": float(np.max(pointwise)),
            "T_exact": T_exact,
            "T_numerical": T_num,
        }

    def summary(self) -> str:
        return (
            f"{self.name}: dim={self.dim}, potential={self.potential_name}\n"
            f"{self.description}"
        )


class Gaussian1DToyProblem(ToyProblem):
    """Section 6.1.1: one-dimensional standard Gaussian, already Stein-type."""

    def __init__(self):
        super().__init__(
            name="gaussian_1d",
            dim=1,
            potential_name="quadratic_sum",
            description="One-dimensional standard Gaussian. Exact Stein-type transformation is the identity.",
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        return norm.pdf(points[:, 0])

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        return norm.logpdf(points[:, 0])

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        return rng.normal(size=(n, 1))

    def exact_transformation(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        return _restore_input_shape(points.copy(), np.asarray(x))


class GaussianBiUnitToyProblem(ToyProblem):
    """Section 6.1.1: two-dimensional Gaussian with positive definite bi-unit covariance."""

    def __init__(self, covariance: Optional[np.ndarray] = None):
        covariance = np.eye(2) if covariance is None else np.asarray(covariance, dtype=float)
        self._validate_covariance(covariance)
        self.covariance = covariance
        self._distribution = multivariate_normal(mean=np.zeros(2), cov=self.covariance)

        super().__init__(
            name="gaussian_2d_biunit",
            dim=2,
            potential_name="quadratic_sum",
            description=(
                "Two-dimensional centered Gaussian with positive definite bi-unit covariance. "
                "By the paper's exercise, the exact Stein-type transformation is the identity."
            ),
        )

    @staticmethod
    def _validate_covariance(covariance: np.ndarray) -> None:
        if covariance.shape != (2, 2):
            raise ValueError("The bi-unit Gaussian toy problem expects a 2x2 covariance matrix.")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("Covariance matrix must be symmetric.")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0:
            raise ValueError("Covariance matrix must be positive definite.")
        if not np.allclose(covariance @ np.ones(2), np.ones(2), atol=1e-10):
            raise ValueError("Covariance matrix must be bi-unit: Sigma 1 = 1.")

    def pdf(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        return self._distribution.pdf(points)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        return self._distribution.logpdf(points)

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        return rng.multivariate_normal(mean=np.zeros(2), cov=self.covariance, size=n)

    def exact_transformation(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        return _restore_input_shape(points.copy(), np.asarray(x))


class IndependentJointToyProblem(ToyProblem):
    """
    Section 6.1.2 and 6.1.3:
    independent input density with known coordinate-wise exact transformation.

    For the quadratic potential, the target marginal is standard normal and
    T_i(x_i) = Phi^{-1}(F_i(x_i)).

    For V(x_1, x_2) = |x_1 + x_2|, the paper states that the independent
    Stein-type target has logistic marginals, so the exact map becomes
    T_i(x_i) = G^{-1}(F_i(x_i)) with G the logistic CDF.
    """

    def __init__(
        self,
        marginals: Sequence[OneDimensionalDistribution],
        target_marginal: OneDimensionalDistribution,
        name: str,
        potential_name: str,
        description: str,
    ):
        if len(marginals) < 1:
            raise ValueError("At least one marginal distribution is required.")
        self.marginals = tuple(marginals)
        self.target_marginal = target_marginal

        super().__init__(
            name=name,
            dim=len(self.marginals),
            potential_name=potential_name,
            description=description,
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        density = np.ones(points.shape[0])
        for j, marginal in enumerate(self.marginals):
            density *= marginal.pdf(points[:, j])
        return density

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        log_density = np.zeros(points.shape[0])
        for j, marginal in enumerate(self.marginals):
            log_density += marginal.logpdf(points[:, j])
        return log_density

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        seeds = rng.integers(0, 2**32 - 1, size=self.dim)
        columns = [
            np.asarray(marginal.rvs(size=n, random_state=int(seed)), dtype=float)
            for marginal, seed in zip(self.marginals, seeds)
        ]
        return np.column_stack(columns)

    def exact_transformation(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        transformed = np.zeros_like(points)
        eps = np.finfo(float).eps

        for j, marginal in enumerate(self.marginals):
            probabilities = marginal.cdf(points[:, j])
            probabilities = np.clip(probabilities, eps, 1.0 - eps)
            transformed[:, j] = self.target_marginal.ppf(probabilities)

        return _restore_input_shape(transformed, np.asarray(x))


class AbsoluteValuePotentialIndependentToyProblem(IndependentJointToyProblem):
    """Section 6.1.3 with logistic Stein-type target marginal."""

    def __init__(self, marginals: Sequence[OneDimensionalDistribution]):
        super().__init__(
            marginals=marginals,
            target_marginal=STANDARD_LOGISTIC,
            name="independent_absolute_value",
            potential_name="absolute_sum",
            description=(
                "Independent joint density under the modified potential V(x1, x2)=|x1+x2|. "
                "The paper states that the independent Stein-type target has logistic marginals."
            ),
        )


class PiecewiseTransformationToyProblem(ToyProblem):
    """
    Section 6.1.4 and Example 2 in Sei (2022).

    Density on [-1, 1]^3 with two high-mass octants and six low-mass octants.
    """

    _HIGH_DENSITY_SIGNS = np.array(
        [
            [-1, +1, +1],
            [+1, -1, -1],
        ],
        dtype=int,
    )
    _LOW_DENSITY_SIGNS = np.array(
        [
            [-1, -1, -1],
            [-1, -1, +1],
            [-1, +1, -1],
            [+1, -1, +1],
            [+1, +1, -1],
            [+1, +1, +1],
        ],
        dtype=int,
    )

    def __init__(self):
        self.constants = np.array([1.2490, 0.3445, 0.3445], dtype=float)
        super().__init__(
            name="piecewise_transformation_3d",
            dim=3,
            potential_name="quadratic_sum",
            description=(
                "Three-dimensional piecewise-constant density on [-1,1]^3 with known closed-form "
                "Stein-type transformation from Sei (2022), Example 2."
            ),
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        density = np.zeros(points.shape[0])

        inside_cube = np.all((points >= -1.0) & (points <= 1.0), axis=1)
        if not np.any(inside_cube):
            return density

        pts = points[inside_cube]
        first_region = (pts[:, 0] <= 0.0) & (pts[:, 1] >= 0.0) & (pts[:, 2] >= 0.0)
        second_region = (pts[:, 0] >= 0.0) & (pts[:, 1] <= 0.0) & (pts[:, 2] <= 0.0)
        high_region = first_region | second_region

        density_inside = np.where(high_region, 3.0 / 8.0, 1.0 / 24.0)
        density[inside_cube] = density_inside
        return density

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        density = self.pdf(x)
        log_density = np.full_like(density, fill_value=-np.inf, dtype=float)
        positive = density > 0.0
        log_density[positive] = np.log(density[positive])
        return log_density

    def sample(self, n: int, random_state: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        choose_high = rng.random(n) < 0.75
        samples = np.zeros((n, 3), dtype=float)

        if np.any(choose_high):
            sign_choices = self._HIGH_DENSITY_SIGNS[rng.integers(0, len(self._HIGH_DENSITY_SIGNS), size=np.sum(choose_high))]
            magnitudes = rng.random((np.sum(choose_high), 3))
            samples[choose_high] = np.where(sign_choices > 0, magnitudes, -magnitudes)

        if np.any(~choose_high):
            sign_choices = self._LOW_DENSITY_SIGNS[rng.integers(0, len(self._LOW_DENSITY_SIGNS), size=np.sum(~choose_high))]
            magnitudes = rng.random((np.sum(~choose_high), 3))
            samples[~choose_high] = np.where(sign_choices > 0, magnitudes, -magnitudes)

        return samples

    def exact_transformation(self, x: np.ndarray) -> np.ndarray:
        points = _ensure_2d_points(x, self.dim)
        transformed = np.zeros_like(points)

        for j in range(self.dim):
            c = self.constants[j]
            negative = points[:, j] < 0.0
            transformed[negative, j] = -c + norm.ppf((1.0 + points[negative, j]) * norm.cdf(c))
            transformed[~negative, j] = c - norm.ppf((1.0 - points[~negative, j]) * norm.cdf(c))

        return _restore_input_shape(transformed, np.asarray(x))


class PaperToyProblems:
    """Convenience registry mirroring the manuscript's toy examples."""

    def __init__(self):
        independent_default_marginals = (LAPLACE, STUDENT_T5)
        self.problems = {
            "gaussian_1d": Gaussian1DToyProblem(),
            "gaussian_2d_biunit": GaussianBiUnitToyProblem(),
            "independent_quadratic": IndependentJointToyProblem(
                marginals=independent_default_marginals,
                target_marginal=STANDARD_NORMAL,
                name="independent_quadratic",
                potential_name="quadratic_sum",
                description=(
                    "Independent joint density with non-Gaussian marginals. "
                    "Exact map is component-wise Phi^{-1}(F_i)."
                ),
            ),
            "independent_absolute_value": AbsoluteValuePotentialIndependentToyProblem(
                marginals=independent_default_marginals
            ),
            "piecewise_transformation_3d": PiecewiseTransformationToyProblem(),
        }

    def register_problem(self, problem: ToyProblem) -> None:
        self.problems[problem.name] = problem

    def get_problem(self, name: str) -> ToyProblem:
        if name not in self.problems:
            available = ", ".join(sorted(self.problems))
            raise KeyError(f"Unknown toy problem '{name}'. Available problems: {available}.")
        return self.problems[name]

    def list_problems(self) -> list[str]:
        return sorted(self.problems)


if __name__ == "__main__":
    registry = PaperToyProblems()

    print("Toy problems from Section 6.1:")
    for problem_name in registry.list_problems():
        problem = registry.get_problem(problem_name)
        print(f"- {problem.name}: {problem.description}")

    example = registry.get_problem("independent_quadratic")
    x = example.sample(5, random_state=42)
    y = example.exact_transformation(x)

    print("\nSample points from the independent quadratic toy problem:")
    print(x)
    print("\nExact coordinate-wise Stein-type transformation at those points:")
    print(y)
