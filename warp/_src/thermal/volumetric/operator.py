# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Volumetric radiative exchange operator, with independent forward and adjoint strategies.

The forward apply and the adjoint run at completely different frequencies within a design
iteration, so tying them to one implementation leaves performance on the table:

===================  ==========================  ==========================================
Stage                Calls per design iteration  Strategy
===================  ==========================  ==========================================
forward / tangent    ~100 (Newton x Krylov)      assemble exchange factors once, then apply
adjoint (VJP)        1                           ray march with the transmittance recurrence
===================  ==========================  ==========================================

Assembly amortizes over the Krylov iterations but costs :math:`O(N_{\\mathrm{elem}}^2)`
memory, so it is selected by problem size and silently replaced by re-marching when it
does not fit. The adjoint gains nothing from assembly — the dependence of the exchange
factors on absorptivity lives in the per-ray transmittance products, so it needs the ray
recurrence either way, and it runs once.

Both strategies are held behind one interface, so callers never choose. Whether a matrix
exists is an implementation detail of the forward path, not a property of the operator.
"""

from __future__ import annotations

import warp as wp
from warp._src.thermal.volumetric.exchange import ExchangeMatrix

__all__ = ["VolumetricRadiationOperator"]

_DEFAULT_ASSEMBLY_LIMIT = 512 * 1024 * 1024


class VolumetricRadiationOperator:
    """Radiative exchange between elements of a participating medium.

    Args:
        rays: Ray bundle over the domain, either a
            :class:`~warp._src.thermal.volumetric.paths.RayPaths` holding its traversals
            explicitly or a :class:`~warp._src.thermal.volumetric.grid.GridRayBundle2D`
            generating them on the fly.
        element_count: Number of elements in the domain.
        backend: ``"assembled"`` to always assemble the exchange factors, ``"march"`` to
            always re-march rays, or ``"auto"`` to decide from problem size.
        assembly_limit_bytes: Memory ceiling for ``"auto"`` selection.
        device: Device to allocate on. Defaults to the device of ``paths``.

    Attributes:
        backend: The strategy actually in use, ``"assembled"`` or ``"march"``, resolved
            from ``backend`` at construction.
        work_ratio: Cost of one assembled apply relative to one marching apply,
            :math:`N_{\\mathrm{elem}}^2 / (\\text{ray steps})`. Below one, assembly wins.

    Note:
        ``"auto"`` applies two independent gates. Assembly must fit in
        ``assembly_limit_bytes``, and it must also be less work: an assembled apply costs
        :math:`O(N_{\\mathrm{elem}}^2)` no matter how the rays fall, while a marching apply
        costs one step per ray step. Assembly pays only when rays revisit the same element
        pairs often enough to deduplicate, which needs a dense angular discretization —
        measured on CPU, assembly wins by 17-22x at ``work_ratio`` near 0.1 and loses at
        2.1. A memory-only gate would pick assembly for coarse ray bundles where marching
        is faster.
    """

    def __init__(
        self,
        rays,
        element_count: int,
        backend: str = "auto",
        assembly_limit_bytes: int = _DEFAULT_ASSEMBLY_LIMIT,
        device=None,
    ):
        if backend not in ("auto", "assembled", "march"):
            raise ValueError(f"Unknown backend '{backend}', expected 'auto', 'assembled', or 'march'")

        self.rays = rays
        self.element_count = element_count
        self.device = device if device is not None else rays.device

        required = ExchangeMatrix.storage_bytes(element_count)
        step_count = rays.step_count
        self.work_ratio = (element_count * element_count) / max(step_count, 1)

        if backend == "auto":
            fits = required <= assembly_limit_bytes
            cheaper = element_count * element_count <= step_count
            backend = "assembled" if (fits and cheaper) else "march"

        self.backend = backend
        self.assembly_bytes = required if backend == "assembled" else 0

        self._exchange = ExchangeMatrix(element_count, device=self.device) if backend == "assembled" else None
        self._absorptivity = None

    def update(self, absorptivity: wp.array):
        """Re-linearize the operator for a new absorptivity field.

        Call once per design iteration, before any apply or VJP. For the assembled backend
        this is where the ray bundle is traced; for the march backend it only records the
        field, and every apply traces instead.

        Args:
            absorptivity: Per-element absorptivity in ``[0, 1]``.
        """
        self._absorptivity = absorptivity
        if self._exchange is not None:
            self.rays.assemble(self._exchange, absorptivity)

    def apply(self, emissive: wp.array, irradiation: wp.array, environment_emissive: wp.array):
        """Accumulate incident radiative power on every element.

        Linear in ``emissive`` and ``environment_emissive``, so this doubles as the tangent
        operator: pass the tangent direction as ``emissive`` and zero as
        ``environment_emissive``.

        Args:
            emissive: Per-element emissive power, e.g. :math:`\\tilde T_e^4`.
            irradiation: Output, per element. Overwritten.
            environment_emissive: Single-element array, the environment's emissive power.
        """
        self._require_update()
        if self._exchange is not None:
            self._exchange.apply(emissive, irradiation, environment_emissive)
        else:
            self.rays.transport(self._absorptivity, emissive, irradiation, environment_emissive)

    def apply_transpose(self, v: wp.array, out: wp.array, out_environment: wp.array):
        """Apply the transpose of the emissive-power tangent to ``v``.

        The radiative tangent is nonsymmetric, so the adjoint solve of the coupled system
        needs this rather than a second :meth:`apply`.

        Args:
            v: Vector to apply the transpose to, per element.
            out: Output, per element. Overwritten.
            out_environment: Single-element output for the environment component. Overwritten.
        """
        self._require_update()
        if self._exchange is not None:
            self._exchange.apply_transpose(v, out, out_environment)
        else:
            self.rays.transport_transpose(self._absorptivity, v, out, out_environment)

    def vjp(
        self,
        emissive: wp.array,
        environment_emissive: wp.array,
        adj_irradiation: wp.array,
        adj_absorptivity: wp.array,
        adj_emissive: wp.array,
        adj_environment_emissive: wp.array,
    ):
        """Accumulate the vector-Jacobian product of :meth:`apply`.

        Always takes the ray-march path, whatever the forward backend: the exchange factors
        depend on absorptivity through per-ray transmittance products that an assembled
        matrix does not retain.

        Adjoint outputs are accumulated, not overwritten, matching Warp's tape convention.
        """
        self._require_update()
        self.rays.transport_vjp(
            self._absorptivity,
            emissive,
            environment_emissive,
            adj_irradiation,
            adj_absorptivity,
            adj_emissive,
            adj_environment_emissive,
        )

    def _require_update(self):
        if self._absorptivity is None:
            raise RuntimeError("VolumetricRadiationOperator.update() must be called before use")
