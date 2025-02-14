"""
Implements differentiable Lambert's problem solver using slightly modified method described in 
Peet, Matthew M. - Lecture 10: Rendezvous and Targeting - Lambert's Problem
https://control.asu.edu/Classes/MAE462/462Lecture10.pdf
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix


@jax.jit
def chord(r1, r2):
    """The chord length between the two points

    @param r1, r2 Locations in Barycentric frame
    """
    return jnp.linalg.norm(r1 - r2)


@jax.jit
def semi_perimeter(r1, r2):
    """The semi_perimeter for two points

    @param r1, r2 Locations in Barycentric frame
    """
    return (chord(r1, r2) + jnp.linalg.norm(r1) + jnp.linalg.norm(r2)) / 2.0


@jax.jit
def min_a(r1, r2, c):
    return jnp.linalg.norm(r1 + r2 + c) / 4


@jax.jit
def alpha_beta(s, c, a):
    """
    Lambert equation parameters alpha and beta, describing angles on a triangles based on the chord
    See [here](https://prussing.ae.illinois.edu/Prussing.alpha-beta.JGCD.2(5).pdf) for intuition

    Note that alpha and beta are functions of a, so must be differentiable when put into the solver

    @param s - semiperimeter
    @param c - chord
    @param a - semimajor axis
    """
    alpha = 2 * jnp.arcsin(jnp.sqrt(s / (2.0 * a)))
    beta = 2 * jnp.arcsin(jnp.sqrt((s - c) / (2.0 * a)))
    return alpha, beta


@jax.jit
def A_B(a, alpha, beta, mu):
    """
    Parameters A and B describing the geometry, used to find the initial and final velocities
    @param a - semimajor axis
    @param alpha, beta - Geometric lambert parameters
    @param mu - Standard gravitational parameter
    """
    AB_coeff = jnp.sqrt(mu / (4 * a))
    return AB_coeff / jnp.tan(alpha / 2), AB_coeff / jnp.tan(beta / 2)


@jax.jit
def initial_final_velocities(r1, r2, A, B):
    """
    Initial and final velocities for a transfer orbit described by initial and final points
    r1 and r2, along with geometric parameters A and B
    @param r1, r2 Locations in Barycentric frame
    @param A, B geometric parameters
    """

    u1 = r1 / jnp.linalg.norm(r1)
    u2 = r2 / jnp.linalg.norm(r2)
    uc = (r2 - r1) / chord(r1, r2)
    v1 = (B + A) * uc + (B - A) * u1
    v2 = (B + A) * uc - (B - A) * u2
    return v1, v2


# Params = [a]
@jax.jit
def lambert_equation_residual(params, args):
    """
    Squared difference between LHS and RHS for Lambert's equation. Should be near zero for
    a solution to lambert solver.

    The jaxopt rules say the first argument is the one you're minimising with respect to
    The rest are constant parameters.

    @param params - Array of parameters to be able to differentiate against. Just [a] in this case
    @param delta_t - time difference between reaching points a and b
    @param s - semiperimeter
    @param c - chord
    @param mu - Standard gravitational parameter
    """
    a = params
    delta_t, s, c, mu = args
    alpha, beta = alpha_beta(s, c, a)
    return delta_t - jnp.sqrt(a**3 / mu) * (
        alpha - beta - (jnp.sin(alpha) - jnp.sin(beta))
    )


# Not jitted due to current limitation of jaxopt with jitting scipy minimiser
# Other minimisers seem to have issues with early stopping and don't properly converge
@jax.jit
def lambert_equation_solve(a_input, delta_t, s, c, mu, a_min, tol=1e-20):
    """
    Solve lambert's equation by minimising the residual with a differential optimiser

    @param a_input - Initial condition to initialise a to. Too low a value may cause nan due to unsolvable a
    @param delta_t - time difference between reaching points a and b
    @param s - semiperimeter
    """
    solver = optimistix.Bisection(rtol=1e-7, atol=1e-7)
    a0 = jnp.array(a_input)
    sol = optimistix.root_find(
        lambert_equation_residual,
        solver,
        a0,
        (delta_t, s, c, mu),
        options=dict(lower=s / 2, upper=10000 * s),
    )
    return sol.value


def plot_residuals(delta_t, s, c, mu):
    """
    Mostly a debug function to make sure solver is working as expected
    """
    import numpy as np
    a_s = np.arange(0, 10, 0.01)
    residuals = [
        np.array(lambert_equation_residual(i, (delta_t, s, c, mu))) for i in a_s
    ]
    plt.plot(a_s, residuals)
    plt.show()


@jax.jit
def lambert_solver(mu, r1, r2, delta_t, debug=False):
    """
    Lambert solver that matches API used by LambertHub solvers

    @param mu - Standard gravitational parameter
    @param r1, r2 Locations in Barycentric frame
    @param delta_t - time difference between reaching points a and b
    """
    s = semi_perimeter(r1, r2)
    c = chord(r1, r2)
    a_min = min_a(r1, r2, c)

    delta_t_p = jnp.sqrt(2) / 3 * jnp.sqrt(s**3 / mu) * (1 - ((s - c) / s) ** (3 / 2))

    a = lambert_equation_solve(s, delta_t, s, c, mu, a_min)
    if debug:
        jax.debug.print("FOUND MINIMISING a IS: {a} with s={s}", a=a, s=s)

    alpha, beta = alpha_beta(s, c, a)
    A, B = A_B(a, alpha, beta, mu)
    return initial_final_velocities(r1, r2, A, B)
