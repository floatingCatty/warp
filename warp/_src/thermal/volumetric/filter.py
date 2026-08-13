# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Density filtering for structured-grid topology optimization.

Left alone, every element's density is an independent design variable, and the optimizer
exploits that freedom to produce checkerboards and single-element features that mean nothing
physically and depend entirely on the mesh. Convolving the design variables with a cone
kernel of radius :math:`R_{\\min}` removes both: it imposes a minimum length scale set by a
physical radius rather than by the discretization.

.. math::
    \\tilde\\rho_e = \\frac{\\sum_f w_{ef} \\rho_f}{\\sum_f w_{ef}},
    \\qquad w_{ef} = \\max(0,\\; R_{\\min} - d_{ef}).

The filter is linear and its weights depend only on the grid, so the matrix is fixed for the
whole optimization and its transpose is the same neighborhood traversal with the roles of the
two indices exchanged. The weights themselves are symmetric; the normalization is not, which
is why the transpose divides by the *neighbor's* weight sum rather than its own.
"""

from __future__ import annotations

import math

import warp as wp

__all__ = ["DensityFilter2D"]


@wp.func
def _cone_weight(i: int, j: int, p: int, q: int, hx: float, hy: float, radius: float) -> float:
    dx = (float(i) - float(p)) * hx
    dy = (float(j) - float(q)) * hy
    d = wp.sqrt(dx * dx + dy * dy)
    return wp.max(0.0, radius - d)


@wp.kernel(enable_backward=False)
def _weight_sums(
    nx: int,
    ny: int,
    span_x: int,
    span_y: int,
    radius: float,
    out: wp.array[float],
):
    i, j = wp.tid()

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)

    total = float(0.0)
    for di in range(-span_x, span_x + 1):
        for dj in range(-span_y, span_y + 1):
            p = i + di
            q = j + dj
            if p >= 0 and p < nx and q >= 0 and q < ny:
                total += _cone_weight(i, j, p, q, hx, hy, radius)

    out[i * ny + j] = total


@wp.kernel(enable_backward=False)
def _apply(
    nx: int,
    ny: int,
    span_x: int,
    span_y: int,
    radius: float,
    weight_sum: wp.array[float],
    density: wp.array[float],
    out: wp.array[float],
):
    i, j = wp.tid()

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)

    acc = float(0.0)
    for di in range(-span_x, span_x + 1):
        for dj in range(-span_y, span_y + 1):
            p = i + di
            q = j + dj
            if p >= 0 and p < nx and q >= 0 and q < ny:
                acc += _cone_weight(i, j, p, q, hx, hy, radius) * density[p * ny + q]

    out[i * ny + j] = acc / weight_sum[i * ny + j]


@wp.kernel(enable_backward=False)
def _apply_transpose(
    nx: int,
    ny: int,
    span_x: int,
    span_y: int,
    radius: float,
    weight_sum: wp.array[float],
    v: wp.array[float],
    out: wp.array[float],
):
    """Gather form of the transpose: element ``f`` collects from every ``e`` that sees it.

    The normalization belongs to the *source* element of the forward filter, so it is read
    at the neighbor rather than at the thread's own element.
    """
    i, j = wp.tid()

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)

    acc = float(0.0)
    for di in range(-span_x, span_x + 1):
        for dj in range(-span_y, span_y + 1):
            p = i + di
            q = j + dj
            if p >= 0 and p < nx and q >= 0 and q < ny:
                neighbor = p * ny + q
                acc += _cone_weight(i, j, p, q, hx, hy, radius) * v[neighbor] / weight_sum[neighbor]

    out[i * ny + j] = acc


class DensityFilter2D:
    """Cone-kernel density filter over a uniform 2D grid on the unit square.

    Args:
        res: Number of cells along each axis.
        radius: Filter radius :math:`R_{\\min}`, in the same units as the unit domain.
        device: Device to allocate on.

    Note:
        A radius smaller than half a cell makes the filter the identity, since no neighbor
        falls inside the cone. That is a legitimate way to disable filtering, but it also
        removes the length-scale control that suppresses checkerboarding.
    """

    def __init__(self, res: tuple[int, int], radius: float, device=None):
        self.res = res
        self.radius = radius
        self.device = device

        # Widest neighborhood the cone can reach along each axis.
        self.span_x = max(1, int(math.ceil(radius * res[0])))
        self.span_y = max(1, int(math.ceil(radius * res[1])))

        self.weight_sum = wp.zeros(res[0] * res[1], dtype=float, device=device)
        wp.launch(
            _weight_sums,
            dim=res,
            inputs=[res[0], res[1], self.span_x, self.span_y, radius],
            outputs=[self.weight_sum],
            device=device,
        )

    @property
    def element_count(self) -> int:
        """Number of elements."""
        return self.res[0] * self.res[1]

    def apply(self, density: wp.array, out: wp.array):
        """Write the filtered density into ``out``.

        Args:
            density: Per-element design variable.
            out: Per-element filtered density. Overwritten.
        """
        wp.launch(
            _apply,
            dim=self.res,
            inputs=[self.res[0], self.res[1], self.span_x, self.span_y, self.radius, self.weight_sum, density],
            outputs=[out],
            device=self.device,
        )

    def apply_transpose(self, v: wp.array, out: wp.array):
        """Write the transpose of the filter applied to ``v`` into ``out``.

        Chains a sensitivity with respect to filtered density back to the design variables.

        Args:
            v: Per-element sensitivity with respect to the filtered density.
            out: Per-element sensitivity with respect to the design variable. Overwritten.
        """
        wp.launch(
            _apply_transpose,
            dim=self.res,
            inputs=[self.res[0], self.res[1], self.span_x, self.span_y, self.radius, self.weight_sum, v],
            outputs=[out],
            device=self.device,
        )
