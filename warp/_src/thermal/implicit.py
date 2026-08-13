# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiating a converged nonlinear solve.

A physics solve is a map from design parameters to the state satisfying
:math:`R(u, \\theta) = 0`. Backpropagating through the Newton iterates would make memory grow
with iteration count and would make the gradient only as converged as the solve. Treating the
solve as an implicit layer avoids both: differentiating the residual identity gives

.. math::
    \\frac{\\partial R}{\\partial u} \\frac{du}{d\\theta} + \\frac{\\partial R}{\\partial \\theta} = 0,

so for an objective :math:`J(u, \\theta)` the total derivative follows from one linear solve
against the *transposed* tangent,

.. math::
    \\left(\\frac{\\partial R}{\\partial u}\\right)^{\\mathsf T} \\lambda
    = -\\left(\\frac{\\partial J}{\\partial u}\\right)^{\\mathsf T},
    \\qquad
    \\frac{dJ}{d\\theta}
    = \\frac{\\partial J}{\\partial \\theta} + \\lambda^{\\mathsf T} \\frac{\\partial R}{\\partial \\theta}.

The cost is one linear solve regardless of how many Newton iterations the forward pass took.

Both routines here are agnostic to the physics. A residual only has to provide the protocol
below, so conduction with radiation, conduction with radiosity, and later a fluid transport
coupling all reuse this machinery rather than each growing their own adjoint.

.. rubric:: Residual protocol

``evaluate(state, out)``
    Write :math:`R(u, \\theta)` into ``out``.
``tangent(v, out)``
    Write :math:`(\\partial R/\\partial u)\\, v` into ``out``, linearized at the last
    ``relinearize`` point.
``tangent_transpose(v, out)``
    Write :math:`(\\partial R/\\partial u)^{\\mathsf T} v`. Required because a coupled
    conduction-radiation tangent is **not** symmetric.
``relinearize(state)``
    Prepare the tangent at ``state``.
``project(v)``
    Zero ``v`` on constrained degrees of freedom, restricting to the free subspace.
``preconditioner_diagonal(out)``
    Write an approximate diagonal of the tangent, used for Jacobi preconditioning.
"""

from __future__ import annotations

import warp as wp
from warp._src.optim.linear import LinearOperator, bicgstab

__all__ = ["NewtonResult", "adjoint_solve", "newton_solve"]


class NewtonResult:
    """Outcome of a Newton solve.

    Attributes:
        converged: Whether the residual norm reached the requested tolerance.
        iterations: Newton iterations performed.
        residual_norm: Final residual norm on the free subspace.
        initial_norm: Residual norm of the initial guess.
        history: Residual norm after each iteration.
        stalled: Whether iteration stopped because the residual stopped decreasing. This is
            the expected outcome once the residual reaches the precision floor of the
            arithmetic rather than a sign of failure, so check ``converged`` alongside it.
    """

    def __init__(
        self,
        converged: bool,
        iterations: int,
        residual_norm: float,
        initial_norm: float,
        history: list[float],
        stalled: bool = False,
    ):
        self.converged = converged
        self.iterations = iterations
        self.residual_norm = residual_norm
        self.initial_norm = initial_norm
        self.history = history
        self.stalled = stalled

    def __repr__(self):
        state = "converged" if self.converged else "NOT converged"
        if self.stalled:
            state += ", stalled"
        return (
            f"NewtonResult({state}, iterations={self.iterations}, "
            f"residual_norm={self.residual_norm:.3e}, initial_norm={self.initial_norm:.3e})"
        )


@wp.kernel(enable_backward=False)
def _axpy(x: wp.array[float], alpha: float, y: wp.array[float]):
    i = wp.tid()
    y[i] = y[i] + alpha * x[i]


@wp.kernel(enable_backward=False)
def _reciprocal_or_one(x: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    v = x[i]
    out[i] = wp.where(v > 1.0e-30 or v < -1.0e-30, 1.0 / v, 1.0)


def _jacobi(residual, node_count, device):
    """Build a Jacobi preconditioner from the residual's approximate tangent diagonal."""
    diag = wp.zeros(node_count, dtype=float, device=device)
    residual.preconditioner_diagonal(diag)
    inverse = wp.zeros(node_count, dtype=float, device=device)
    wp.launch(_reciprocal_or_one, dim=node_count, inputs=[diag], outputs=[inverse], device=device)

    def matvec(x, y, z, alpha, beta):
        wp.launch(_scaled_add, dim=node_count, inputs=[x, inverse, y, alpha, beta], outputs=[z], device=device)

    return LinearOperator((node_count, node_count), float, device, matvec)


@wp.kernel(enable_backward=False)
def _scaled_add(
    x: wp.array[float],
    scale: wp.array[float],
    y: wp.array[float],
    alpha: float,
    beta: float,
    z: wp.array[float],
):
    i = wp.tid()
    z[i] = alpha * scale[i] * x[i] + beta * y[i]


def _tangent_operator(residual, node_count, device, transpose: bool):
    """Wrap the residual's tangent as a matrix-free operator on the free subspace."""
    scratch = wp.zeros(node_count, dtype=float, device=device)
    apply = residual.tangent_transpose if transpose else residual.tangent

    def matvec(x, y, z, alpha, beta):
        apply(x, scratch)
        # Constrained rows and columns are removed symmetrically, so the projected operator
        # stays consistent with the projected right-hand side.
        residual.project(scratch)
        wp.launch(_combine, dim=node_count, inputs=[scratch, y, alpha, beta], outputs=[z], device=device)

    return LinearOperator((node_count, node_count), float, device, matvec)


@wp.kernel(enable_backward=False)
def _combine(a: wp.array[float], y: wp.array[float], alpha: float, beta: float, z: wp.array[float]):
    i = wp.tid()
    z[i] = alpha * a[i] + beta * y[i]


def _norm(v: wp.array) -> float:
    x = v.numpy()
    return float((x @ x) ** 0.5)


def newton_solve(
    residual,
    state: wp.array,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
    max_iterations: int = 25,
    linear_tol: float = 1.0e-8,
    linear_max_iterations: int = 500,
    line_search_steps: int = 6,
    quiet: bool = True,
) -> NewtonResult:
    """Drive ``residual`` to zero by Newton iteration, updating ``state`` in place.

    Each step solves the tangent system matrix-free with BiCGSTAB, which does not assume a
    symmetric operator; a radiative tangent is not symmetric. Steps are backtracked when they
    would increase the residual, which matters because a cold start far from equilibrium
    makes the quartic emission term overshoot badly.

    Args:
        residual: Object implementing the residual protocol.
        state: Initial guess, overwritten with the converged solution.
        rtol: Convergence tolerance relative to the initial residual norm.
        atol: Absolute floor added to the convergence threshold.
        max_iterations: Maximum Newton iterations.
        linear_tol: Relative tolerance for each tangent solve.
        linear_max_iterations: Maximum iterations for each tangent solve.
        line_search_steps: Maximum step halvings per iteration. Zero disables backtracking.
        quiet: If False, print the residual norm at each iteration.

    Returns:
        A :class:`NewtonResult` describing the outcome.

    Note:
        The default ``rtol`` reflects that the module computes in single precision. The
        residual cannot be evaluated more accurately than roughly ``1e-6`` relative, so
        asking for more produces a stall rather than a better answer. Iteration stops on
        stagnation and reports it through :attr:`NewtonResult.stalled`.
    """
    device = state.device
    n = state.shape[0]

    r = wp.zeros(n, dtype=float, device=device)
    delta = wp.zeros(n, dtype=float, device=device)
    trial = wp.zeros(n, dtype=float, device=device)

    def residual_norm(x):
        residual.evaluate(x, r)
        residual.project(r)
        return _norm(r)

    norm = residual_norm(state)
    initial_norm = norm
    threshold = atol + rtol * max(initial_norm, 1.0e-30)

    history = [norm]
    converged = norm <= threshold
    stalled = False
    iterations = 0

    while not converged and iterations < max_iterations:
        iterations += 1

        residual.relinearize(state)
        delta.zero_()
        bicgstab(
            _tangent_operator(residual, n, device, transpose=False),
            b=r,
            x=delta,
            tol=linear_tol,
            maxiter=linear_max_iterations,
            M=_jacobi(residual, n, device),
            use_cuda_graph=False,
        )
        residual.project(delta)

        # The Newton step is -K^{-1} R, and delta holds K^{-1} R.
        step = -1.0
        accepted = norm
        for _ in range(line_search_steps + 1):
            trial.assign(state)
            wp.launch(_axpy, dim=n, inputs=[delta, step], outputs=[trial], device=device)
            accepted = residual_norm(trial)
            if accepted < norm or step > -1.0e-4:
                break
            step *= 0.5

        state.assign(trial)
        previous, norm = norm, accepted
        history.append(norm)
        if not quiet:
            print(f"  newton {iterations:3d}  |R| = {norm:.6e}")

        if norm <= threshold:
            converged = True
        elif norm > 0.9 * previous:
            # No longer making progress: either at the precision floor or diverging.
            stalled = True
            break

    # `r` holds the residual of the last trial state, which is the current state.
    return NewtonResult(converged, iterations, norm, initial_norm, history, stalled)


def adjoint_solve(
    residual,
    state: wp.array,
    objective_gradient: wp.array,
    adjoint: wp.array,
    tol: float = 1.0e-10,
    max_iterations: int = 500,
) -> None:
    """Solve the adjoint system for a converged state.

    Computes :math:`\\lambda` from
    :math:`(\\partial R/\\partial u)^{\\mathsf T} \\lambda = -(\\partial J/\\partial u)^{\\mathsf T}`.

    Args:
        residual: Object implementing the residual protocol.
        state: Converged state, used as the linearization point.
        objective_gradient: :math:`\\partial J/\\partial u`, per degree of freedom.
        adjoint: Output. Overwritten with :math:`\\lambda`.
        tol: Relative tolerance for the linear solve.
        max_iterations: Maximum iterations for the linear solve.
    """
    device = state.device
    n = state.shape[0]

    residual.relinearize(state)

    rhs = wp.zeros(n, dtype=float, device=device)
    wp.launch(_axpy, dim=n, inputs=[objective_gradient, -1.0], outputs=[rhs], device=device)
    residual.project(rhs)

    adjoint.zero_()
    bicgstab(
        _tangent_operator(residual, n, device, transpose=True),
        b=rhs,
        x=adjoint,
        tol=tol,
        maxiter=max_iterations,
        M=_jacobi(residual, n, device),
        use_cuda_graph=False,
    )
    residual.project(adjoint)
