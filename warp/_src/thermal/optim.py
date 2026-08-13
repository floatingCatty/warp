# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimality criteria update for volume-constrained topology optimization.

For a single resource constraint, the optimality criteria update is the classical choice in
density-based topology optimization: it is derived from the KKT conditions rather than from a
generic descent direction, so it makes large, well-scaled moves without any learning rate to
tune. At the optimum, material is distributed so that every element with intermediate density
delivers the same objective improvement per unit volume, and the update drives the design
toward that condition directly:

.. math::
    \\rho_e \\leftarrow \\rho_e \\left(
    \\frac{-\\partial J / \\partial \\rho_e}{\\lambda\\, \\partial V / \\partial \\rho_e}
    \\right)^{\\eta},

with :math:`\\lambda` found by bisection so the volume constraint holds exactly, and each step
limited to a move ``m`` so the design cannot jump past the linearization it was derived from.

The bisection runs on the host because it needs a scalar comparison per iteration, but the
candidate designs and the volume reduction stay on device; a few dozen scalar readbacks are
negligible next to one radiation solve.

.. note::
    This assumes adding material improves the objective, so that the sensitivity is
    non-positive. That holds for compliance-like objectives including minimizing temperature
    under a heat load. Positive sensitivities are clipped to zero, which stalls those
    elements rather than letting the update take a root of a negative number. For objectives
    or constraint sets where that assumption fails, a general-purpose method such as MMA is
    the right tool.
"""

from __future__ import annotations

import math

import warp as wp

__all__ = ["optimality_criteria_step"]


@wp.kernel(enable_backward=False)
def _candidate(
    density: wp.array[float],
    sensitivity: wp.array[float],
    multiplier: float,
    move: float,
    eta: float,
    lower: float,
    upper: float,
    out: wp.array[float],
):
    e = wp.tid()

    rho = density[e]
    # Only material that reduces the objective earns a positive scaling factor.
    descent = wp.max(0.0, -sensitivity[e])
    ratio = wp.pow(descent / multiplier, eta)

    candidate = rho * ratio
    candidate = wp.min(candidate, wp.min(upper, rho + move))
    candidate = wp.max(candidate, wp.max(lower, rho - move))
    out[e] = candidate


@wp.kernel(enable_backward=False)
def _sum(values: wp.array[float], out: wp.array[float]):
    e = wp.tid()
    wp.atomic_add(out, 0, values[e])


def optimality_criteria_step(
    density: wp.array,
    sensitivity: wp.array,
    volume_fraction: float,
    move: float = 0.2,
    eta: float = 0.5,
    lower: float = 0.0,
    upper: float = 1.0,
    bisection_iterations: int = 60,
    multiplier_bounds: tuple[float, float] = (1.0e-12, 1.0e12),
) -> float:
    """Advance ``density`` one optimality criteria step, in place.

    Args:
        density: Per-element design variable, updated in place.
        sensitivity: :math:`\\partial J / \\partial \\rho`, per element.
        volume_fraction: Target mean density. The constraint is met exactly, to bisection
            tolerance, so it stays active throughout the optimization.
        move: Largest change any element may make in one step.
        eta: Damping exponent on the optimality ratio.
        lower: Lower bound on density.
        upper: Upper bound on density.
        bisection_iterations: Bisection steps used to find the Lagrange multiplier.
        multiplier_bounds: Bracket for the multiplier search.

    Returns:
        The Lagrange multiplier that satisfies the volume constraint.
    """
    device = density.device
    n = density.shape[0]

    candidate = wp.zeros(n, dtype=float, device=device)
    total = wp.zeros(1, dtype=float, device=device)
    target = volume_fraction * n

    # Bisect on the logarithm. The multiplier is a price per unit volume whose magnitude
    # follows the sensitivity's, which changes with the objective's scale and by orders of
    # magnitude over an optimization; bisecting linearly across a bracket spanning that many
    # decades resolves small multipliers to nothing and lets the volume constraint drift.
    low, high = math.log(multiplier_bounds[0]), math.log(multiplier_bounds[1])
    for _ in range(bisection_iterations):
        mid = 0.5 * (low + high)

        wp.launch(
            _candidate,
            dim=n,
            inputs=[density, sensitivity, math.exp(mid), move, eta, lower, upper],
            outputs=[candidate],
            device=device,
        )
        total.zero_()
        wp.launch(_sum, dim=n, inputs=[candidate], outputs=[total], device=device)

        # Volume falls as the multiplier rises, so an overshoot means the price is too low.
        if float(total.numpy()[0]) > target:
            low = mid
        else:
            high = mid

    multiplier = math.exp(0.5 * (low + high))
    wp.launch(
        _candidate,
        dim=n,
        inputs=[density, sensitivity, multiplier, move, eta, lower, upper],
        outputs=[candidate],
        device=device,
    )
    density.assign(candidate)
    return multiplier
