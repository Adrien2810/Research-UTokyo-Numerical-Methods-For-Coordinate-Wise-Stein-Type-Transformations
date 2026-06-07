import numpy as np
from scipy.optimize import LinearConstraint
from scipy.linalg import block_diag
from scipy.interpolate import BSpline
from scipy.optimize import minimize


# Packages you may want later, depending on your implementation:
# from scipy.stats import norm
from dataclasses import dataclass

ArrayLike = np.ndarray


@dataclass
class BSplineConfig:
    """Configuration of a one-dimensional B-spline."""

    degree: int # We use degree 3 and not the order
    knots: np.ndarray #That's our k
    domain: tuple[float, float] #The restriction on the mass of density

    # Derive the properties of the B-spline from the degree and the knot vector
    # For example, the number of basis functions is determined by the degree and the number of knots.
    @property
    def n_basis(self) -> int:
        return len(self.knots) - self.degree -1
    
    # Check if the input x is within the domain of the B-spline. This is important for ensuring that we only evaluate the spline where it is defined.
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


class BSplineApproach:
    """
    Personal implementation scaffold for the B-spline method.

    Suggested workflow:
    1. Choose a knot vector and spline degree.
    2. Build a basis for T_1 and T_2.
    3. Draw or fix Monte Carlo samples from the toy problem.
    4. Write down the empirical objective.
    5. Optimize over the spline coefficients.
    6. Check monotonicity and the stationary condition.
    """

    def __init__(self, problem, config: BSplineConfig | list[BSplineConfig], samples: MonteCarloSamples):
        self.problem = problem
        self.samples = samples
        # Allow either one shared spline configuration or one configuration
        # per transform coordinate. (Good for NFL dataset)
        if isinstance(config, BSplineConfig):
            self.configs = [config]
        else:
            self.configs = list(config)

        if not self.configs:
            raise ValueError("At least one spline configuration is required.")

        # With one-dimensional code.
        self.config = self.configs[0]

        # Cache basis splines and their derivatives separately for each transform.
        self._basis_splines_by_transform = {}
        self._derivative_splines_by_transform = {}
        self._basis_signatures = {}

    # Clearly state which configuration belongs to which transform
    # If only one configuration, then return the same config
    def _get_config(self, transform_index: int) -> BSplineConfig:
        """Return the spline configuration for one transform coordinate."""
        if transform_index < 0:
            raise ValueError("`transform_index` must be non-negative.")

        if len(self.configs) == 1:
            return self.configs[0]

        if transform_index >= len(self.configs):
            raise ValueError(
                f"Requested transform {transform_index}, but only "
                f"{len(self.configs)} spline configurations are available."
            )

        return self.configs[transform_index]

    # return current knot vector and current degree if necessary to check for consistency
    def _basis_signature(self, transform_index: int) -> tuple[tuple[float, ...], int]:
        """Return a lightweight signature used to detect config changes."""
        config = self._get_config(transform_index)
        return (tuple(np.asarray(config.knots, dtype=float)), config.degree)

    # We use the Scipy BSpline package to build B Splines basis and the derivatives
    def _build_basis(self, transform_index: int) -> None:
        """
        Build and cache the canonical B-spline basis for one transform.

        Each basis function is obtained by choosing one unit coefficient vector.
        The derivative basis is then computed directly from SciPy's spline
        object, which is cleaner than hard-coding derivative coefficients.
        """
        config = self._get_config(transform_index)
        basis_splines = []
        derivative_splines = []

        for j in range(config.n_basis):
            coeffs = np.zeros(config.n_basis, dtype=float)
            coeffs[j] = 1.0 # This creates a unit vector that selects the j-th basis function when passed to BSpline. just an initialization

            spline = BSpline(
                config.knots,
                coeffs,
                config.degree,
                extrapolate=True,
            )
            basis_splines.append(spline)
            derivative_splines.append(spline.derivative())

        self._basis_splines_by_transform[transform_index] = basis_splines
        self._derivative_splines_by_transform[transform_index] = derivative_splines
        self._basis_signatures[transform_index] = self._basis_signature(transform_index)

    # As the basis functions depend on the spline configuration, we need to check if the basis is ready before evaluating it. If not, we build it on demand. This allows us to change the spline configuration dynamically without having to worry about manually rebuilding the basis every time.
    def _ensure_basis_ready(self, transform_index: int) -> None:
        """
        Lazily build the basis when first needed and rebuild it automatically
        if the corresponding knots or degree have changed.
        """
        signature = self._basis_signature(transform_index)

        if (
            transform_index not in self._basis_splines_by_transform
            or transform_index not in self._derivative_splines_by_transform
            or self._basis_signatures.get(transform_index) != signature
        ):
            self._build_basis(transform_index)


    # Codex build this checking function to see if the shape of coefs match the theory.
    def _validate_coefficients(
        self,
        coefficients: ArrayLike,
        transform_index: int,
    ) -> np.ndarray:
        """
        Validate and convert the coefficient vector for one transform.

        For transform i, the spline is represented as
            T_i(x) = sum_j c_{i,j} B_{i,j}(x),
        so the coefficient vector must match the number of basis functions of
        the corresponding spline configuration.
        """
        config = self._get_config(transform_index)
        coefficients = np.asarray(coefficients, dtype=float)

        if coefficients.shape != (config.n_basis,):
            raise ValueError(
                f"Expected coefficients of shape ({config.n_basis},) for "
                f"transform {transform_index}, got {coefficients.shape}."
            )

        return coefficients

    # Evaluate either the basis matrix or the derivative basis matrix for one
    # selected transform coordinate.
    # We write the B Spline basis in matrices for numerical stability when evaluating
    def spline_matrix(
        self,
        x: ArrayLike,
        derivative: bool = False,
        transform_index: int = 0,
    ) -> np.ndarray:
        self._ensure_basis_ready(transform_index)
        x = np.atleast_1d(np.asarray(x, dtype=float))

        if derivative:
            splines = self._derivative_splines_by_transform[transform_index]
        else:
            splines = self._basis_splines_by_transform[transform_index]

        return np.column_stack([spline(x) for spline in splines])
    
    # Now we use the ansatz that trafo = sum coefs*b_splines evaluated in x
    # T(x) = sum_j c_j B_j(x),  T'(x) = sum_j c_j B_j'(x)
    def transform(self, x, coefficients, transform_index: int = 0):
        coefficients = self._validate_coefficients(coefficients, transform_index)
        basis_matrix = self.spline_matrix(x, derivative=False, transform_index=transform_index)
        return basis_matrix @ coefficients

    def transform_derivative(self, x, coefficients, transform_index: int = 0):
        coefficients = self._validate_coefficients(coefficients, transform_index)
        derivative_matrix = self.spline_matrix(x, derivative=True, transform_index=transform_index)
        return derivative_matrix @ coefficients

    # Check how many Monte Carlo samples fall into the spline domain for each
    # requested transform coordinate.
    def sample_domain_report(self, number_transforms: int) -> dict:
        points = np.asarray(self.samples.points, dtype=float)

        if points.ndim != 2:
            raise ValueError(
                f"Expected `samples.points` to be a 2D array, got shape {points.shape}."
            )

        if number_transforms < 1:
            raise ValueError("`number_transforms` must be at least 1.")

        if number_transforms > points.shape[1]:
            raise ValueError(
                f"Requested {number_transforms} transforms, but samples only have "
                f"{points.shape[1]} coordinate columns."
            )

        if len(self.configs) not in (1, number_transforms):
            raise ValueError(
                "The number of spline configurations must be either 1 "
                "(shared across all transforms) or exactly equal to "
                "`number_transforms`."
            )

        selected_points = points[:, :number_transforms]
        configs = [self._get_config(j) for j in range(number_transforms)]
        in_domain_per_transform = [
            config.contains(selected_points[:, j]) for j, config in enumerate(configs)
        ]
        in_domain = np.logical_and.reduce(in_domain_per_transform)

        return {
            "n_samples": int(self.samples.n_samples),
            "n_in_domain": int(np.sum(in_domain)),
            "n_out_of_domain": int(np.sum(~in_domain)),
            "fraction_in_domain": float(np.mean(in_domain)),
            "domains": [
                tuple(float(v) for v in config.domain)
                for config in configs
            ],
        }

    # Now we check that indeed the MC points are within the spline domains
    #_active_sample_mask(number_transforms) builds a Boolean row-mask selecting exactly those Monte Carlo samples whose first number_transforms coordinates all lie inside their spline domains, 
    # while _validate_active_samples(number_transforms) checks that this mask is True for every sample and raises an error if any sample falls outside the allowed domains.
    def _active_sample_mask(self, number_transforms) -> ArrayLike:
        points = np.asarray(self.samples.points, dtype=float)

        if points.ndim != 2:
            raise ValueError(
                f"Expected `samples.points` to be a 2D array, got shape {points.shape}."
            )

        if number_transforms < 1:
            raise ValueError("`number_transforms` must be at least 1.")

        if number_transforms > points.shape[1]:
            raise ValueError(
                f"Requested {number_transforms} transforms, but samples only have "
                f"{points.shape[1]} coordinate columns."
            )

        if len(self.configs) not in (1, number_transforms):
            raise ValueError(
                "The number of spline configurations must be either 1 "
                "(shared across all transforms) or exactly equal to "
                "`number_transforms`."
            )

        selected_points = points[:, :number_transforms]
        in_domain_per_transform = [
            self._get_config(j).contains(selected_points[:, j])
            for j in range(number_transforms)
        ]
        return np.logical_and.reduce(in_domain_per_transform)

    def _validate_active_samples(self, number_transforms) -> None:
        mask = self._active_sample_mask(number_transforms)
        if not np.all(mask):
            raise ValueError(
                f"Not all samples are within the spline domains for the requested "
                f"{number_transforms} transforms. "
                f"Number of active samples: {np.sum(mask)} out of {mask.shape[0]}."
            )

    # We want to optimize over the coefficients of all transforms simultaneously. Hence we need to build blocks out of the flat coefficeients
    # Moerover, with this representation of coefficients, we can naturally implement the monotonicity constraints

    def _split_coefficients(self, flat_coefficients, number_transforms):
        """
        Split one flat optimization vector into one coefficient block per transform.
        """
        flat_coefficients = np.asarray(flat_coefficients, dtype=float).ravel()

        blocks = []
        start = 0
        for j in range(number_transforms):
            n_basis_j = self._get_config(j).n_basis
            stop = start + n_basis_j
            if stop > flat_coefficients.size:
                raise ValueError(
                    f"Coefficient vector is too short for {number_transforms} transforms."
                )
            blocks.append(flat_coefficients[start:stop])
            start = stop

        if start != flat_coefficients.size:
            raise ValueError(
                f"Coefficient vector has length {flat_coefficients.size}, but only "
                f"{start} entries are used by the requested transforms."
            )

        return blocks

    # Now we want to implement the basic monte Carlo Objective function given in equation 19
    def MonteCarloObjective(self, flat_coefficients, number_transforms):
        self._validate_active_samples(number_transforms)

        coeff_blocks = self._split_coefficients(flat_coefficients, number_transforms)

        active_mask = self._active_sample_mask(number_transforms)
        active_samples = np.asarray(self.samples.points, dtype=float)[
            active_mask, :number_transforms
        ]

        transformed_points = np.zeros_like(active_samples, dtype=float)
        transformed_derivatives = np.zeros_like(active_samples, dtype=float)

        tolerance = 1e-6

        for j in range(number_transforms):
            coefficients_j = coeff_blocks[j]

            transformed_points[:, j] = self.transform(
                active_samples[:, j],
                coefficients_j,
                transform_index=j,
            )
            transformed_derivatives[:, j] = self.transform_derivative(
                active_samples[:, j],
                coefficients_j,
                transform_index=j,
            )

            min_derivative_j = np.min(transformed_derivatives[:, j])
            if min_derivative_j <= tolerance:
                return 1e12 #if a derivative is too small we give a very large penalty to the optimizer
                # Thhis enables us to continue the procedure even if the monotonicity constraint is not satisfied, which can be useful in the early stages of optimization when the coefficients are still far from optimal and the monotonicity condition is likely to be violated.

        first_term = -np.sum(np.mean(np.log(transformed_derivatives), axis=0))
        second_term = 0.5 * np.mean(np.sum(transformed_points, axis=1) ** 2)

        return first_term + second_term
        
    # Now we implement the monotonicity check, which is a necessary condition for the transform to be valid. We check that the derivative of the transform is positive for all active samples. This ensures that the transform is strictly increasing and therefore invertible.
    # Moreover we define the constraint for the upcoming optimization procedure

    # We first define a matrix which builds up the differences c_i-c_(i-1) for each of the coefs blocks
    def _difference_matrix(self, n_basis):
        """
        Build the matrix D such that D @ c = [c_1-c_0, c_2-c_1, ..., c_{m-1}-c_{m-2}].
        """
        if n_basis < 2:
            raise ValueError("Need at least 2 basis functions to impose monotonicity.")

        D = np.zeros((n_basis - 1, n_basis))
        for j in range(n_basis - 1):
            D[j, j] = -1.0
            D[j, j + 1] = 1.0
        return D  
    
    # Now we define the monotonicty constraints. For this we multiply the difference matrix with the coefficient vector and check that all entries are positive. This ensures that c_i - c_(i-1) > 0 for all i, which in turn ensures that the derivative of the transform is positive.
    def build_monotonicity_constraint(self, number_transforms):
        """
        Build the block-diagonal linear constraint enforcing
        c_{i,j} - c_{i,j-1} >= 0 for every transform i.
        We use the package LinearConstraints from Scipy.optimize as the constraint is linear and we can substitue it in our optimization solver
        """
        blocks = []
        for j in range(number_transforms):
            n_basis_j = self._get_config(j).n_basis
            blocks.append(self._difference_matrix(n_basis_j))

        A = block_diag(*blocks)
        # To exlude the constant sequence when initiliazing
        lb = 1e-4 * np.ones(A.shape[0])
        ub = np.full(A.shape[0], np.inf)

        return LinearConstraint(A, lb=lb, ub=ub)
    
    # After we implemented the OVF and the constraint, we want a callback function for analyzing the results
    def callback(self, number_transforms):
        # initialize the history of the optimization procedure
        history = {
            "iteration": [],
            "objective": [],
            "monotonicity_satisfied": [],
            "min_derivatives": [],
        }
        # Define the callback function that will be called at each iteration of the optimization procedure. This function will check the monotonicity condition and record the objective value and the minimum derivative for each transform.
        def callback_fn(xk): #xk is the current flat coefficient vector
            history["iteration"].append(np.copy(xk))
            objective_value = self.MonteCarloObjective(xk, number_transforms)
            history["objective"].append(objective_value)

            # Check monotonicity and record the minimum derivative for each transform
            coeff_blocks = self._split_coefficients(xk, number_transforms)
            active_mask = self._active_sample_mask(number_transforms)
            points = np.asarray(self.samples.points, dtype=float)[active_mask, :number_transforms]
            # Go through each transform and compute min derivative and check monotonicity
            min_dev = np.zeros(number_transforms)
            for j in range(number_transforms):
                coefficients_j = coeff_blocks[j]
                transformed_derivatives = self.transform_derivative(
                    points[:, j],
                    coefficients_j,
                    transform_index=j,
                )
                min_dev[j] = np.min(transformed_derivatives)

            history["min_derivatives"].append(min_dev)
            history["monotonicity_satisfied"].append(np.all(min_dev > 0))
        return callback_fn, history
        
    # We now have the objective function and the linear constraint. We can now minimize over the coefs simultaneously!
    def optimize(self, initial_guess, number_transforms, maxiter=250):
        constraint = self.build_monotonicity_constraint(number_transforms)
        # include callback function for monitoring the optimization procedure
        callback_fn, history = self.callback(number_transforms)
        # SLSQP stands for Sequential Least Squares Programming. Very roughly, it solves your constrained nonlinear problem by repeatedly replacing it with a local quadratic approximation and then solving that simpler subproblem.
        result = minimize(
            fun=self.MonteCarloObjective,
            x0=np.asarray(initial_guess, dtype=float),
            args=(number_transforms,),
            method="SLSQP",
            constraints=[constraint],
            callback=callback_fn,
            options={"maxiter": maxiter},
        )

        result.history = history 
        return result

    # From the optimize function we get a concatenated vector of the optimal coefficients. From this 
    # we can construct our coordinate wise transforms in the B-Spline representation.
    # First we check if the result obtained satisfy the monotonicty condition. As an input we assume flat coefficients given by resul from minimize
    def check_monotonicity(self, flat_coefficients, number_transforms, tolerance=1e-10):
        self._validate_active_samples(number_transforms)

        coeff_blocks = self._split_coefficients(flat_coefficients, number_transforms)
        active_mask = self._active_sample_mask(number_transforms)
        points = np.asarray(self.samples.points, dtype=float)[
            active_mask, :number_transforms
        ]

        transformed_derivatives = np.zeros_like(points, dtype=float)

        for j in range(number_transforms):
            coefficients_j = coeff_blocks[j]
            transformed_derivatives[:, j] = self.transform_derivative(
                points[:, j],
                coefficients_j,
                transform_index=j,
            )

            min_derivative_j = np.min(transformed_derivatives[:, j])
            if min_derivative_j <= tolerance:
                raise ValueError(
                    f"Derivative of transform {j} is too close to zero for some samples. "
                    f"Minimum derivative: {min_derivative_j}"
                )

        return True

    # Finally we know that over the p-fiber, the unique solution to our problem is given by the Stein-Type density function
    # That is, that it should satisfy the Stein-Identity, as it is the stationary condition
    # Hence, for a few trivial test functions, we check if Stein-Identity is satisfied
    
    def check_stationary_condition(
    self,
    flat_coefficients,
    number_transforms,
    potential_function_derivative,
    test_function,
    test_function_derivative,
    tolerance=1e-4, # we want the stein-type identity to be satisfied up to some numerical tolerance, as we are working with finite samples and numerical approximations
    ):
        """
        For potentials of the form V(y) = psi(y_1 + ... + y_n), stationarity implies
        E[u(T_i(X_i)) psi'(T_1(X_1) + ... + T_n(X_n))] = E[u'(T_i(X_i))]
        for each transform i and smooth test functions u.
        """
        self._validate_active_samples(number_transforms)

        coeff_blocks = self._split_coefficients(flat_coefficients, number_transforms)
        active_mask = self._active_sample_mask(number_transforms)
        points = np.asarray(self.samples.points, dtype=float)[
            active_mask, :number_transforms
        ]

        transformed_points = np.zeros_like(points, dtype=float)
        for j in range(number_transforms):
            coefficients_j = coeff_blocks[j]
            transformed_points[:, j] = self.transform(
                points[:, j],
                coefficients_j,
                transform_index=j,
            )

        total = np.sum(transformed_points, axis=1)
        grad_potential = potential_function_derivative(total)

        residuals = np.zeros(number_transforms, dtype=float)

        for j in range(number_transforms):
            coefficients_j = coeff_blocks[j]
            transformed_points_j = self.transform(
                points[:, j],
                coefficients_j,
                transform_index=j,
            )
            test_function_values = test_function(transformed_points_j)
            test_function_derivative_values = test_function_derivative(transformed_points_j)

            residuals[j] = (
                np.mean(test_function_values * grad_potential)
                - np.mean(test_function_derivative_values)
            )

            if abs(residuals[j]) > tolerance:
                raise ValueError(
                    f"Stationary condition not satisfied for transform {j}. "
                    f"Residual: {residuals[j]}"
                )

        return residuals


if __name__ == "__main__":
    print("Use this file for your own implementation.")
