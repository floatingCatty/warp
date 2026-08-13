# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cached ray traversal topology for volumetric radiative transport.

On a fixed mesh the *set* of elements a ray passes through depends only on the geometry
and the ray directions, never on the density field: density enters solely through the
per-element absorptivity multiplying each traversal step. :class:`RayPaths` caches that
topology once so every design iteration reuses it, and so the transport operators in
:mod:`warp._src.thermal.volumetric.transport` reduce to a walk over a flat array
rather than a re-traced ray.

The layout is CSR over rays::

    ray r visits cells[path_begin[r]], ..., cells[path_begin[r + 1] - 1]

in propagation order, starting from the first element *past* the emitting element.
"""

from __future__ import annotations

import numpy as np

import warp as wp
from warp._src.context import DeviceLike

__all__ = ["EXIT_ADIABATIC", "EXIT_ENVIRONMENT", "RayPaths", "trace_grid_2d"]


EXIT_ENVIRONMENT = 0
"""Ray leaves the domain toward the external environment at ``environment_emissive``."""

EXIT_ADIABATIC = 1
"""Ray leaves the domain in an adiabatic direction; its residual returns to the emitter."""


class RayPaths:
    """Traversal topology of a bundle of rays over a fixed mesh.

    Args:
        source: Emitting element index of each ray, shape ``(ray_count,)``.
        weight: Geometric weight of each ray, shape ``(ray_count,)``. For the zonal
            view-factor discretization this is :math:`C A_a \\cos\\theta_c / (N_{pt} A_e)`,
            so that summing over all rays of an element reproduces its view factors.
        exit_kind: :data:`EXIT_ENVIRONMENT` or :data:`EXIT_ADIABATIC` per ray, shape
            ``(ray_count,)``.
        path_begin: CSR offsets into ``cells``, shape ``(ray_count + 1,)``.
        cells: Element indices along each ray in propagation order, shape ``(step_count,)``.
        device: Device the arrays live on.

    Attributes:
        transmittance: Scratch array of shape ``(step_count,)`` holding the prefix
            transmittance :math:`\\tau_{k-1}` at each traversal step. Filled by the
            transport VJP; exposed because it is also useful for diagnostics.
    """

    def __init__(
        self,
        source: wp.array,
        weight: wp.array,
        exit_kind: wp.array,
        path_begin: wp.array,
        cells: wp.array,
        device: DeviceLike = None,
    ):
        self.source = source
        self.weight = weight
        self.exit_kind = exit_kind
        self.path_begin = path_begin
        self.cells = cells
        self.device = device if device is not None else source.device

        self.transmittance = wp.zeros(cells.shape[0], dtype=float, device=self.device)

    @property
    def ray_count(self) -> int:
        """Number of rays in the bundle."""
        return self.source.shape[0]

    @property
    def step_count(self) -> int:
        """Total number of traversal steps across all rays."""
        return self.cells.shape[0]

    # The methods below mirror `GridRayBundle2D`, so a radiation operator can take either
    # an explicitly stored bundle or an implicit one without knowing which it holds.

    def assemble(self, exchange, absorptivity: wp.array):
        """Accumulate exchange factors into ``exchange``."""
        exchange.assemble(self, absorptivity)

    def transport(
        self,
        absorptivity: wp.array,
        emissive: wp.array,
        irradiation: wp.array,
        environment_emissive: wp.array,
    ):
        """Accumulate incident radiative power on every element."""
        from warp._src.thermal.volumetric import transport as _transport  # noqa: PLC0415

        _transport.transport(self, absorptivity, emissive, irradiation, environment_emissive)

    def transport_transpose(
        self,
        absorptivity: wp.array,
        v: wp.array,
        out: wp.array,
        out_environment: wp.array,
    ):
        """Apply the transpose of the emissive-power tangent to ``v``."""
        from warp._src.thermal.volumetric import transport as _transport  # noqa: PLC0415

        _transport.transport_transpose(self, absorptivity, v, out, out_environment)

    def transport_vjp(
        self,
        absorptivity: wp.array,
        emissive: wp.array,
        environment_emissive: wp.array,
        adj_irradiation: wp.array,
        adj_absorptivity: wp.array,
        adj_emissive: wp.array,
        adj_environment_emissive: wp.array,
    ):
        """Accumulate the vector-Jacobian product of :meth:`transport`."""
        from warp._src.thermal.volumetric import transport as _transport  # noqa: PLC0415

        _transport.transport_vjp(
            self,
            absorptivity,
            emissive,
            environment_emissive,
            adj_irradiation,
            adj_absorptivity,
            adj_emissive,
            adj_environment_emissive,
        )

    @staticmethod
    def from_numpy(
        source: np.ndarray,
        weight: np.ndarray,
        exit_kind: np.ndarray,
        paths: list[np.ndarray] | list[list[int]],
        device: DeviceLike = None,
    ) -> RayPaths:
        """Build a :class:`RayPaths` from per-ray element lists.

        Args:
            source: Emitting element index per ray.
            weight: Geometric weight per ray.
            exit_kind: Exit classification per ray.
            paths: One sequence of element indices per ray, in propagation order.
            device: Device to allocate on.
        """
        lengths = np.array([len(p) for p in paths], dtype=np.int32)
        path_begin = np.zeros(len(paths) + 1, dtype=np.int32)
        np.cumsum(lengths, out=path_begin[1:])

        flat = np.concatenate([np.asarray(p, dtype=np.int32) for p in paths]) if paths else np.zeros(0, np.int32)

        return RayPaths(
            source=wp.array(np.asarray(source, dtype=np.int32), dtype=int, device=device),
            weight=wp.array(np.asarray(weight, dtype=np.float32), dtype=float, device=device),
            exit_kind=wp.array(np.asarray(exit_kind, dtype=np.int32), dtype=int, device=device),
            path_begin=wp.array(path_begin, dtype=int, device=device),
            cells=wp.array(flat, dtype=int, device=device),
            device=device,
        )


def trace_grid_2d(
    res: tuple[int, int],
    origin: tuple[float, float],
    direction: tuple[float, float],
    max_steps: int = 0,
) -> list[int]:
    """Walk a 2D structured grid with an Amanatides-Woo DDA and return the visited cells.

    The grid spans the unit square ``[0, 1]^2`` with ``res[0] x res[1]`` cells, and cells are
    numbered row-major as ``i * res[1] + j``. The cell containing ``origin`` is *not*
    included: rays are launched from an element face and the emitter attenuates nothing.

    Args:
        res: Number of cells along each axis.
        origin: Ray launch point in ``[0, 1]^2``.
        direction: Ray direction; need not be normalized.
        max_steps: Stop after this many steps, or run to the domain boundary if ``0``.

    Returns:
        Visited cell indices in propagation order.

    Note:
        This is a NumPy/Python reference walker used to *build* the cached topology, not a
        per-iteration hot path. It runs once per geometry, not once per design iteration.
    """
    nx, ny = res
    dx, dy = 1.0 / nx, 1.0 / ny

    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        return []
    d /= norm

    i = min(max(int(origin[0] * nx), 0), nx - 1)
    j = min(max(int(origin[1] * ny), 0), ny - 1)

    step_i = 1 if d[0] > 0 else -1
    step_j = 1 if d[1] > 0 else -1

    # Parametric distance to the next grid line on each axis, and between grid lines.
    if d[0] != 0.0:
        next_x = (i + (1 if d[0] > 0 else 0)) * dx
        t_max_x = (next_x - origin[0]) / d[0]
        t_delta_x = abs(dx / d[0])
    else:
        t_max_x = np.inf
        t_delta_x = np.inf

    if d[1] != 0.0:
        next_y = (j + (1 if d[1] > 0 else 0)) * dy
        t_max_y = (next_y - origin[1]) / d[1]
        t_delta_y = abs(dy / d[1])
    else:
        t_max_y = np.inf
        t_delta_y = np.inf

    visited: list[int] = []
    limit = max_steps if max_steps > 0 else (nx + ny) * 2

    while len(visited) < limit:
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            i += step_i
        else:
            t_max_y += t_delta_y
            j += step_j

        if i < 0 or i >= nx or j < 0 or j >= ny:
            break

        visited.append(i * ny + j)

    return visited
