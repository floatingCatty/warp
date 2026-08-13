# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable volumetric radiative transport along cached ray paths.

A ray leaving element :math:`e` with geometric weight :math:`w` traverses elements
:math:`d_1, \\ldots, d_n` with absorptivities :math:`f_k`, is attenuated by the
transmittance :math:`\\tau_k = \\tau_{k-1}(1 - f_k)`, and deposits what it carries into
each element it crosses. The energy it delivers back to its emitter is

.. math::
    g = w \\left[ \\sum_k \\tau_{k-1} f_k E_{d_k} + \\tau_n E_{\\mathrm{exit}} \\right],

where :math:`E_{\\mathrm{exit}}` is the environment's emissive power if the ray escapes to
the environment, or the emitter's own if it escapes in an adiabatic direction. Summing
over all rays of an element yields :math:`\\sum_E F_{eE} E_E + F_{e,\\mathrm{env}} E_{\\mathrm{env}}`
without ever forming the exchange-factor matrix :math:`F`.

Both the forward pass and the adjoint use the equivalent front-to-back compositing
recurrence

.. math::
    A_{n+1} = E_{\\mathrm{exit}}, \\qquad A_k = f_k E_{d_k} + (1 - f_k) A_{k+1},
    \\qquad g = w A_1,

whose derivative is division-free:

.. math::
    \\frac{\\partial g}{\\partial f_j} = w\\, \\tau_{j-1} \\left( E_{d_j} - A_{j+1} \\right).

.. warning::
    The adjoint here is hand-written and the kernels are built with
    ``enable_backward=False`` **on purpose**. Warp does not replay dynamic loops in the
    backward pass, so letting it differentiate the transmittance product would produce
    silently wrong gradients — off by O(1), with no error raised. See
    :ref:`dynamic_loops` in the differentiability guide.

The operator acts on *absorptivity*, not on density. Material interpolation (SIMP, filtering)
is an ordinary elementwise map that Warp differentiates correctly on its own, so keeping it
outside confines the hand-written adjoint to the one kernel that actually needs it, and lets
the same operator serve surface emissivity in the explicit-geometry regime.
"""

from __future__ import annotations

import warp as wp
from warp._src.thermal.volumetric.paths import EXIT_ADIABATIC, RayPaths

__all__ = ["transport", "transport_transpose", "transport_vjp"]


@wp.kernel(enable_backward=False)
def _transport_forward(
    source: wp.array[int],
    weight: wp.array[float],
    exit_kind: wp.array[int],
    path_begin: wp.array[int],
    cells: wp.array[int],
    absorptivity: wp.array[float],
    emissive: wp.array[float],
    environment_emissive: wp.array[float],
    irradiation: wp.array[float],
):
    """Accumulate per-element incident radiation by compositing back to front."""
    r = wp.tid()

    e = source[r]
    beg = path_begin[r]
    end = path_begin[r + 1]

    # Energy still in flight when the ray leaves the domain returns to the emitter in an
    # adiabatic direction, and reaches the environment otherwise.
    if exit_kind[r] == EXIT_ADIABATIC:
        acc = emissive[e]
    else:
        acc = environment_emissive[0]

    for k in range(end - 1, beg - 1, -1):
        c = cells[k]
        f = absorptivity[c]
        acc = f * emissive[c] + (1.0 - f) * acc

    wp.atomic_add(irradiation, e, weight[r] * acc)


@wp.kernel(enable_backward=False)
def _transport_vjp(
    source: wp.array[int],
    weight: wp.array[float],
    exit_kind: wp.array[int],
    path_begin: wp.array[int],
    cells: wp.array[int],
    absorptivity: wp.array[float],
    emissive: wp.array[float],
    environment_emissive: wp.array[float],
    adj_irradiation: wp.array[float],
    transmittance: wp.array[float],
    adj_absorptivity: wp.array[float],
    adj_emissive: wp.array[float],
    adj_environment_emissive: wp.array[float],
):
    """Propagate the adjoint of :func:`transport` back to absorptivity and emissive power.

    Two walks over the cached path: forward to lay down the prefix transmittance
    :math:`\\tau_{k-1}`, then backward to build :math:`A_{k+1}` and emit both adjoints.
    Neither walk divides by a transmittance, so opaque cells (:math:`f \\to 1`) are safe.
    """
    r = wp.tid()

    e = source[r]
    beg = path_begin[r]
    end = path_begin[r + 1]
    gbar = adj_irradiation[e] * weight[r]

    # Forward walk: transmittance[k] holds tau_{k-1}, the fraction still in flight on
    # entering step k. Each entry is written by exactly one ray, so there is no race.
    tau = float(1.0)
    for k in range(beg, end):
        transmittance[k] = tau
        tau = tau * (1.0 - absorptivity[cells[k]])

    if exit_kind[r] == EXIT_ADIABATIC:
        acc = emissive[e]
    else:
        acc = environment_emissive[0]

    # Backward walk: `acc` enters iteration k holding A_{k+1}.
    for k in range(end - 1, beg - 1, -1):
        c = cells[k]
        f = absorptivity[c]
        t = transmittance[k]

        wp.atomic_add(adj_absorptivity, c, gbar * t * (emissive[c] - acc))
        wp.atomic_add(adj_emissive, c, gbar * t * f)

        acc = f * emissive[c] + (1.0 - f) * acc

    # `tau` now holds tau_n, the fraction that survives to the far end of the ray.
    if exit_kind[r] == EXIT_ADIABATIC:
        wp.atomic_add(adj_emissive, e, gbar * tau)
    else:
        wp.atomic_add(adj_environment_emissive, 0, gbar * tau)


@wp.kernel(enable_backward=False)
def _transport_transpose(
    source: wp.array[int],
    weight: wp.array[float],
    exit_kind: wp.array[int],
    path_begin: wp.array[int],
    cells: wp.array[int],
    absorptivity: wp.array[float],
    v: wp.array[float],
    out: wp.array[float],
    out_environment: wp.array[float],
):
    """Scatter form of the transport operator, applying its transpose to ``v``."""
    r = wp.tid()

    e = source[r]
    beg = path_begin[r]
    end = path_begin[r + 1]
    s = v[e] * weight[r]

    tau = float(1.0)
    for k in range(beg, end):
        c = cells[k]
        f = absorptivity[c]
        wp.atomic_add(out, c, s * tau * f)
        tau = tau * (1.0 - f)

    if exit_kind[r] == EXIT_ADIABATIC:
        wp.atomic_add(out, e, s * tau)
    else:
        wp.atomic_add(out_environment, 0, s * tau)


def transport(
    paths: RayPaths,
    absorptivity: wp.array,
    emissive: wp.array,
    irradiation: wp.array,
    environment_emissive: wp.array,
):
    """Accumulate incident radiative power on every element.

    Computes :math:`\\sum_E F_{eE} E_E + F_{e,\\mathrm{env}} E_{\\mathrm{env}}` for each element
    :math:`e`, matrix-free. The result is *linear* in ``emissive`` and
    ``environment_emissive``, so this same call also applies the tangent operator
    :math:`\\partial(\\text{irradiation})/\\partial E \\cdot v`: pass ``v`` as ``emissive``
    and zero as ``environment_emissive``.

    Args:
        paths: Cached ray traversal topology.
        absorptivity: Per-element absorptivity in ``[0, 1]``.
        emissive: Per-element emissive power, e.g. :math:`\\tilde T_e^4`.
        irradiation: Output, per element. Zeroed by this call.
        environment_emissive: Single-element array holding the environment's emissive power.
    """
    irradiation.zero_()
    wp.launch(
        _transport_forward,
        dim=paths.ray_count,
        inputs=[
            paths.source,
            paths.weight,
            paths.exit_kind,
            paths.path_begin,
            paths.cells,
            absorptivity,
            emissive,
            environment_emissive,
        ],
        outputs=[irradiation],
        device=paths.device,
    )


def transport_vjp(
    paths: RayPaths,
    absorptivity: wp.array,
    emissive: wp.array,
    environment_emissive: wp.array,
    adj_irradiation: wp.array,
    adj_absorptivity: wp.array,
    adj_emissive: wp.array,
    adj_environment_emissive: wp.array,
):
    """Accumulate the vector-Jacobian product of :func:`transport`.

    Adjoint outputs are accumulated, not overwritten, matching Warp's tape convention.

    Args:
        paths: Cached ray traversal topology, as used in the forward pass.
        absorptivity: Linearization point, per element.
        emissive: Linearization point, per element.
        environment_emissive: Single-element array, as in the forward pass.
        adj_irradiation: Incoming adjoint of the forward output.
        adj_absorptivity: Output, accumulated.
        adj_emissive: Output, accumulated.
        adj_environment_emissive: Single-element output, accumulated.
    """
    wp.launch(
        _transport_vjp,
        dim=paths.ray_count,
        inputs=[
            paths.source,
            paths.weight,
            paths.exit_kind,
            paths.path_begin,
            paths.cells,
            absorptivity,
            emissive,
            environment_emissive,
            adj_irradiation,
            paths.transmittance,
        ],
        outputs=[adj_absorptivity, adj_emissive, adj_environment_emissive],
        device=paths.device,
    )


def transport_transpose(
    paths: RayPaths,
    absorptivity: wp.array,
    v: wp.array,
    out: wp.array,
    out_environment: wp.array,
):
    """Apply the transpose of the emissive-power tangent operator to ``v``.

    Needed because the radiative tangent is not symmetric: the adjoint solve of the coupled
    conduction-radiation system uses :math:`K^{\\mathsf T}`, not :math:`K`.

    Args:
        paths: Cached ray traversal topology.
        absorptivity: Linearization point, per element.
        v: Vector to apply the transpose to, per element.
        out: Output, per element. Zeroed by this call.
        out_environment: Single-element output for the environment component. Zeroed.
    """
    out.zero_()
    out_environment.zero_()
    wp.launch(
        _transport_transpose,
        dim=paths.ray_count,
        inputs=[
            paths.source,
            paths.weight,
            paths.exit_kind,
            paths.path_begin,
            paths.cells,
            absorptivity,
            v,
        ],
        outputs=[out, out_environment],
        device=paths.device,
    )
