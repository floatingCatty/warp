# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bilinear heat conduction on a 2D structured grid.

Discretizes ``-div(k(rho) grad T) = Q`` with bilinear (Q1) elements on a uniform grid over
the unit square. Temperature lives on nodes, conductivity on elements, which is what lets a
density field control the conduction path.

The element operator is evaluated with 2x2 Gauss quadrature, exact for Q1, and conductivity
is constant within an element. That makes the density derivative exactly the unit element
matrix scaled by ``dk/drho``, so the adjoint needs no autodiff and no assembled matrix:

.. math::
    \\frac{\\partial}{\\partial \\rho_e} \\left( \\lambda^{\\mathsf T} K(\\rho) T \\right)
    = \\frac{dk}{d\\rho_e}\\, \\lambda_e^{\\mathsf T} \\hat K \\lambda_e^{\\mathsf T} T_e .

Radiation couples through two transfer operators rather than through the conduction stencil.
:meth:`~ConductionOperator2D.element_average` reduces the nodal temperature to the
element-wise representative temperature that drives emission, and
:meth:`~ConductionOperator2D.scatter_element_source` returns a volumetric element source to
the nodes. They are adjoints of one another up to element volume, which is what makes the
coupled residual's transpose consistent.
"""

from __future__ import annotations

import numpy as np

import warp as wp

__all__ = ["ConductionOperator2D", "DirichletMask"]

# 2x2 Gauss-Legendre abscissa on the reference element; weights are all one.
_GAUSS = 0.5773502691896258


@wp.func
def _shape_gradients(xi: float, eta: float, hx: float, hy: float):
    """Physical gradients of the four bilinear shape functions at a reference point.

    Local nodes are ordered counter-clockwise from the lower-left corner of the cell.
    """
    # d/dxi and d/deta of N = (1 +- xi)(1 +- eta) / 4, mapped to physical coordinates by the
    # constant Jacobian of an axis-aligned rectangle.
    sx = 2.0 / hx
    sy = 2.0 / hy

    gx0 = -0.25 * (1.0 - eta) * sx
    gx1 = 0.25 * (1.0 - eta) * sx
    gx2 = 0.25 * (1.0 + eta) * sx
    gx3 = -0.25 * (1.0 + eta) * sx

    gy0 = -0.25 * (1.0 - xi) * sy
    gy1 = -0.25 * (1.0 + xi) * sy
    gy2 = 0.25 * (1.0 + xi) * sy
    gy3 = 0.25 * (1.0 - xi) * sy

    return wp.vec4(gx0, gx1, gx2, gx3), wp.vec4(gy0, gy1, gy2, gy3)


@wp.func
def _unit_element_matrix(a: int, b: int, hx: float, hy: float) -> float:
    """Entry ``(a, b)`` of the element stiffness matrix at unit conductivity."""
    detj = 0.25 * hx * hy

    acc = float(0.0)
    for p in range(2):
        for q in range(2):
            xi = wp.where(p == 0, -_GAUSS, _GAUSS)
            eta = wp.where(q == 0, -_GAUSS, _GAUSS)
            gx, gy = _shape_gradients(xi, eta, hx, hy)
            acc += (gx[a] * gx[b] + gy[a] * gy[b]) * detj

    return acc


@wp.func
def _local_node(cell_i: int, cell_j: int, a: int, ny: int) -> int:
    """Global index of local node ``a`` of cell ``(cell_i, cell_j)``."""
    di = wp.where(a == 1 or a == 2, 1, 0)
    dj = wp.where(a == 2 or a == 3, 1, 0)
    return (cell_i + di) * (ny + 1) + (cell_j + dj)


@wp.kernel(enable_backward=False)
def _apply(
    nx: int,
    ny: int,
    conductivity: wp.array[float],
    temperature: wp.array[float],
    out: wp.array[float],
):
    """Accumulate ``K(k) T`` element by element."""
    i, j = wp.tid()

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)
    k = conductivity[i * ny + j]

    for a in range(4):
        acc = float(0.0)
        for b in range(4):
            acc += _unit_element_matrix(a, b, hx, hy) * temperature[_local_node(i, j, b, ny)]
        wp.atomic_add(out, _local_node(i, j, a, ny), k * acc)


@wp.kernel(enable_backward=False)
def _apply_vjp(
    nx: int,
    ny: int,
    temperature: wp.array[float],
    adj_out: wp.array[float],
    adj_conductivity: wp.array[float],
):
    """Adjoint of :func:`_apply` with respect to conductivity.

    Conductivity is constant within an element, so its derivative is exactly the unit
    element matrix and the adjoint reduces to the bilinear form
    ``lambda_e^T Khat T_e`` on each element.
    """
    i, j = wp.tid()

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)

    contraction = float(0.0)
    for a in range(4):
        na = _local_node(i, j, a, ny)
        acc = float(0.0)
        for b in range(4):
            acc += _unit_element_matrix(a, b, hx, hy) * temperature[_local_node(i, j, b, ny)]
        contraction += adj_out[na] * acc

    wp.atomic_add(adj_conductivity, i * ny + j, contraction)


@wp.kernel(enable_backward=False)
def _apply_transpose(
    nx: int,
    ny: int,
    conductivity: wp.array[float],
    v: wp.array[float],
    out: wp.array[float],
):
    """``K`` is symmetric, so its transpose is itself; kept explicit for the residual API."""
    i, j = wp.tid()

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)
    k = conductivity[i * ny + j]

    for a in range(4):
        acc = float(0.0)
        for b in range(4):
            acc += _unit_element_matrix(a, b, hx, hy) * v[_local_node(i, j, b, ny)]
        wp.atomic_add(out, _local_node(i, j, a, ny), k * acc)


@wp.kernel(enable_backward=False)
def _diagonal(nx: int, ny: int, conductivity: wp.array[float], out: wp.array[float]):
    i, j = wp.tid()

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)
    k = conductivity[i * ny + j]

    for a in range(4):
        wp.atomic_add(out, _local_node(i, j, a, ny), k * _unit_element_matrix(a, a, hx, hy))


@wp.kernel(enable_backward=False)
def _element_average(nx: int, ny: int, nodal: wp.array[float], out: wp.array[float]):
    """Volume-averaged element value. Exact for Q1: every shape function integrates to V/4."""
    i, j = wp.tid()

    acc = float(0.0)
    for a in range(4):
        acc += nodal[_local_node(i, j, a, ny)]

    out[i * ny + j] = 0.25 * acc


@wp.kernel(enable_backward=False)
def _scatter_element_source(
    nx: int,
    ny: int,
    element: wp.array[float],
    scale: float,
    out: wp.array[float],
):
    """Distribute a volumetric element source to its nodes as ``int N_a Q dOmega``."""
    i, j = wp.tid()

    volume = scale / (float(nx) * float(ny))
    contribution = 0.25 * volume * element[i * ny + j]

    for a in range(4):
        wp.atomic_add(out, _local_node(i, j, a, ny), contribution)


@wp.kernel(enable_backward=False)
def _project(fixed: wp.array[int], v: wp.array[float]):
    n = wp.tid()
    if fixed[n] != 0:
        v[n] = 0.0


@wp.kernel(enable_backward=False)
def _apply_values(fixed: wp.array[int], values: wp.array[float], v: wp.array[float]):
    n = wp.tid()
    if fixed[n] != 0:
        v[n] = values[n]


class DirichletMask:
    """Prescribed-temperature nodes of a conduction problem.

    Args:
        fixed: Nonzero at nodes whose temperature is prescribed, shape ``(node_count,)``.
        values: Prescribed temperature at those nodes, shape ``(node_count,)``. Entries at
            free nodes are ignored.
    """

    def __init__(self, fixed: wp.array, values: wp.array):
        self.fixed = fixed
        self.values = values
        self.device = fixed.device

    def project(self, v: wp.array):
        """Zero ``v`` at prescribed nodes, restricting it to the free subspace."""
        wp.launch(_project, dim=v.shape[0], inputs=[self.fixed], outputs=[v], device=self.device)

    def apply_values(self, v: wp.array):
        """Write the prescribed temperatures into ``v``, leaving free nodes untouched."""
        wp.launch(_apply_values, dim=v.shape[0], inputs=[self.fixed, self.values], outputs=[v], device=self.device)


class ConductionOperator2D:
    """Bilinear heat conduction on a uniform 2D grid over the unit square.

    Elements are numbered row-major as ``i * res[1] + j`` and nodes as
    ``i * (res[1] + 1) + j``, so the element numbering matches
    :class:`~warp._src.thermal.volumetric.grid.GridRayBundle2D`.

    Args:
        res: Number of cells along each axis.
        device: Device to allocate on.
    """

    def __init__(self, res: tuple[int, int], device=None):
        self.res = res
        self.device = device

    @property
    def element_count(self) -> int:
        """Number of elements."""
        return self.res[0] * self.res[1]

    @property
    def node_count(self) -> int:
        """Number of nodes."""
        return (self.res[0] + 1) * (self.res[1] + 1)

    @property
    def element_volume(self) -> float:
        """Volume of one element, in the nondimensional unit domain."""
        return 1.0 / self.element_count

    def apply(self, conductivity: wp.array, temperature: wp.array, out: wp.array):
        """Accumulate ``K(k) T`` into ``out``, which is zeroed first.

        Args:
            conductivity: Per-element dimensionless conductivity.
            temperature: Per-node temperature.
            out: Per-node output.
        """
        out.zero_()
        wp.launch(
            _apply,
            dim=self.res,
            inputs=[self.res[0], self.res[1], conductivity, temperature],
            outputs=[out],
            device=self.device,
        )

    def apply_transpose(self, conductivity: wp.array, v: wp.array, out: wp.array):
        """Accumulate ``K(k)^T v``. The conduction operator is symmetric, so this equals
        :meth:`apply`; it exists so the coupled residual can transpose uniformly."""
        out.zero_()
        wp.launch(
            _apply_transpose,
            dim=self.res,
            inputs=[self.res[0], self.res[1], conductivity, v],
            outputs=[out],
            device=self.device,
        )

    def diagonal(self, conductivity: wp.array, out: wp.array):
        """Accumulate the diagonal of ``K(k)``, for Jacobi preconditioning."""
        out.zero_()
        wp.launch(
            _diagonal,
            dim=self.res,
            inputs=[self.res[0], self.res[1], conductivity],
            outputs=[out],
            device=self.device,
        )

    def apply_vjp(self, temperature: wp.array, adj_out: wp.array, adj_conductivity: wp.array):
        """Accumulate the conductivity adjoint of :meth:`apply`.

        The temperature adjoint is omitted because ``K`` is symmetric: it is
        :meth:`apply` driven by ``adj_out``, which callers already have.

        Args:
            temperature: Linearization point, per node.
            adj_out: Incoming adjoint of :meth:`apply`'s output, per node.
            adj_conductivity: Output, per element. Accumulated.
        """
        wp.launch(
            _apply_vjp,
            dim=self.res,
            inputs=[self.res[0], self.res[1], temperature, adj_out],
            outputs=[adj_conductivity],
            device=self.device,
        )

    def element_average(self, nodal: wp.array, out: wp.array):
        """Reduce a nodal field to its volume average on each element.

        This is the representative temperature that drives radiative emission.
        """
        wp.launch(
            _element_average,
            dim=self.res,
            inputs=[self.res[0], self.res[1], nodal],
            outputs=[out],
            device=self.device,
        )

    def scatter_element_source(self, element: wp.array, out: wp.array, scale: float = 1.0):
        """Accumulate a volumetric element source onto the nodes as ``int N_a Q dOmega``.

        Adjoint of :meth:`element_average` up to the element volume, which is what keeps
        the coupled residual's transpose consistent.

        Args:
            element: Per-element volumetric source.
            out: Per-node output. Accumulated, not zeroed.
            scale: Multiplier applied to the source.
        """
        wp.launch(
            _scatter_element_source,
            dim=self.res,
            inputs=[self.res[0], self.res[1], element, scale],
            outputs=[out],
            device=self.device,
        )

    def node_positions(self) -> np.ndarray:
        """Node coordinates in the unit square, shape ``(node_count, 2)``."""
        nx, ny = self.res
        i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
        return np.stack([i.ravel() / nx, j.ravel() / ny], axis=1)
