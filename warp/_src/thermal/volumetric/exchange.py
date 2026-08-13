# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assembled exchange factors, an optional fast backing for the forward transport apply.

Absorptivity is constant across a design iteration, so the exchange factors
:math:`F_{eE}` are constant too, while the coupled Newton-Krylov solve applies the
transport operator on the order of a hundred times against changing temperatures.
Assembling :math:`F` once and reducing each of those applies to a matrix-vector product
trades memory for a large reduction in repeated work.

The trade only pays below a size threshold. Storage is :math:`O(N_{\\mathrm{elem}}^2)`,
which is a few tens of megabytes for a 2D design domain but grows out of reach in 3D, so
this is a *backing strategy* selected by problem size, not the interface. Callers go
through :class:`~warp._src.thermal.volumetric.operator.VolumetricRadiationOperator`,
which falls back to re-marching rays when assembly does not fit.

The adjoint never uses this path. The dependence of :math:`F` on absorptivity runs through
the transmittance products along each ray, so the VJP needs the ray-level recurrence in
:mod:`~warp._src.thermal.volumetric.transport` regardless — and it runs once per design
iteration, where the repeated-work argument does not apply.
"""

from __future__ import annotations

import warp as wp
from warp._src.thermal.volumetric.paths import EXIT_ADIABATIC, RayPaths

__all__ = ["ExchangeMatrix"]


@wp.kernel(enable_backward=False)
def _assemble(
    source: wp.array[int],
    weight: wp.array[float],
    exit_kind: wp.array[int],
    path_begin: wp.array[int],
    cells: wp.array[int],
    absorptivity: wp.array[float],
    factors: wp.array2d[float],
    environment_factors: wp.array[float],
):
    """Deposit each ray's attenuated contribution into the exchange factors.

    Identical traversal to the scatter form of the transport operator, so the two stay
    consistent by construction rather than by coincidence.
    """
    r = wp.tid()

    e = source[r]
    beg = path_begin[r]
    end = path_begin[r + 1]
    w = weight[r]

    tau = float(1.0)
    for k in range(beg, end):
        c = cells[k]
        f = absorptivity[c]
        wp.atomic_add(factors, e, c, w * tau * f)
        tau = tau * (1.0 - f)

    if exit_kind[r] == EXIT_ADIABATIC:
        wp.atomic_add(factors, e, e, w * tau)
    else:
        wp.atomic_add(environment_factors, e, w * tau)


@wp.kernel(enable_backward=False)
def _apply(
    factors: wp.array2d[float],
    environment_factors: wp.array[float],
    emissive: wp.array[float],
    environment_emissive: wp.array[float],
    irradiation: wp.array[float],
):
    e = wp.tid()

    n = factors.shape[1]
    tail = n % 4

    # Four independent accumulators: the dependency chain of a single running sum is what
    # stops this loop from pipelining, and the apply is the innermost cost of every Krylov
    # iteration. Measured at 1.36x over the scalar form.
    a0 = environment_factors[e] * environment_emissive[0]
    a1 = float(0.0)
    a2 = float(0.0)
    a3 = float(0.0)

    for c in range(0, n - tail, 4):
        a0 += factors[e, c] * emissive[c]
        a1 += factors[e, c + 1] * emissive[c + 1]
        a2 += factors[e, c + 2] * emissive[c + 2]
        a3 += factors[e, c + 3] * emissive[c + 3]

    for c in range(n - tail, n):
        a0 += factors[e, c] * emissive[c]

    irradiation[e] = (a0 + a1) + (a2 + a3)


@wp.kernel(enable_backward=False)
def _apply_transpose(
    factors: wp.array2d[float],
    v: wp.array[float],
    out: wp.array[float],
):
    c = wp.tid()

    acc = float(0.0)
    for e in range(factors.shape[0]):
        acc += factors[e, c] * v[e]

    out[c] = acc


@wp.kernel(enable_backward=False)
def _apply_transpose_environment(
    environment_factors: wp.array[float],
    v: wp.array[float],
    out_environment: wp.array[float],
):
    e = wp.tid()
    wp.atomic_add(out_environment, 0, environment_factors[e] * v[e])


class ExchangeMatrix:
    """Dense exchange factors :math:`F_{eE}` and :math:`F_{e,\\mathrm{env}}`.

    Args:
        element_count: Number of elements in the domain.
        device: Device to allocate on.

    Attributes:
        factors: Element-to-element exchange factors, shape ``(element_count, element_count)``.
        environment_factors: Element-to-environment exchange factors, shape ``(element_count,)``.
    """

    def __init__(self, element_count: int, device=None):
        self.element_count = element_count
        self.device = device
        self.factors = wp.zeros((element_count, element_count), dtype=float, device=device)
        self.environment_factors = wp.zeros(element_count, dtype=float, device=device)

    @staticmethod
    def storage_bytes(element_count: int) -> int:
        """Bytes a dense assembly would occupy for a domain of ``element_count`` elements."""
        return 4 * (element_count * element_count + element_count)

    def assemble(self, paths: RayPaths, absorptivity: wp.array):
        """Trace the ray bundle once and accumulate the exchange factors.

        Call once per design iteration, after absorptivity changes and before any apply.

        Args:
            paths: Cached ray traversal topology.
            absorptivity: Per-element absorptivity in ``[0, 1]``.
        """
        self.factors.zero_()
        self.environment_factors.zero_()
        wp.launch(
            _assemble,
            dim=paths.ray_count,
            inputs=[
                paths.source,
                paths.weight,
                paths.exit_kind,
                paths.path_begin,
                paths.cells,
                absorptivity,
            ],
            outputs=[self.factors, self.environment_factors],
            device=self.device,
        )

    def apply(self, emissive: wp.array, irradiation: wp.array, environment_emissive: wp.array):
        """Accumulate incident radiative power, equivalent to
        :func:`~warp._src.thermal.volumetric.transport.transport`.

        Linear in ``emissive``, so this also applies the tangent operator: pass the tangent
        direction as ``emissive`` and zero as ``environment_emissive``.
        """
        wp.launch(
            _apply,
            dim=self.element_count,
            inputs=[self.factors, self.environment_factors, emissive, environment_emissive],
            outputs=[irradiation],
            device=self.device,
        )

    def apply_transpose(self, v: wp.array, out: wp.array, out_environment: wp.array):
        """Apply the transpose of the emissive-power tangent, equivalent to
        :func:`~warp._src.thermal.volumetric.transport.transport_transpose`."""
        out_environment.zero_()
        wp.launch(
            _apply_transpose,
            dim=self.element_count,
            inputs=[self.factors, v],
            outputs=[out],
            device=self.device,
        )
        wp.launch(
            _apply_transpose_environment,
            dim=self.element_count,
            inputs=[self.environment_factors, v],
            outputs=[out_environment],
            device=self.device,
        )
