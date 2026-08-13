# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification of the bilinear conduction operator against analytical solutions.

Unlike the radiative transport tests, which compare two discretizations of the same integral,
these check the discretization itself against closed-form solutions of the heat equation.
A patch test pins consistency, a manufactured parabolic solution pins the source coupling,
and the operator's algebraic properties are checked directly.
"""

import unittest

import numpy as np

import warp as wp
import warp.thermal as thermal
from warp.tests.unittest_utils import *


def _solve_dense(op, conductivity, rhs, mask_fixed, fixed_values, device):
    """Reference solve by dense elimination, independent of any iterative machinery."""
    n = op.node_count
    columns = np.zeros((n, n))
    probe = wp.zeros(n, dtype=float, device=device)
    out = wp.zeros(n, dtype=float, device=device)
    basis = np.zeros(n, dtype=np.float32)

    for c in range(n):
        basis[:] = 0.0
        basis[c] = 1.0
        probe.assign(basis)
        op.apply(conductivity, probe, out)
        columns[:, c] = out.numpy()

    free = mask_fixed == 0
    b = rhs.copy()
    b -= columns @ np.where(free, 0.0, fixed_values)

    t = np.where(free, 0.0, fixed_values).astype(np.float64)
    sub = columns[np.ix_(free, free)]
    t[free] = np.linalg.solve(sub, b[free])
    return t, columns


def _uniform(op, value, device):
    return wp.array(np.full(op.element_count, value, dtype=np.float32), dtype=float, device=device)


def test_patch_test(test, device):
    """A linear temperature field must produce zero residual at every interior node.

    This is the standard consistency check: if the discrete operator cannot reproduce a
    field the continuous operator annihilates, it is not consistent at any resolution.
    """
    for res in [(4, 4), (7, 3)]:
        op = thermal.ConductionOperator2D(res, device=device)
        pos = op.node_positions()

        for grad in [(1.0, 0.0), (0.0, 1.0), (2.0, -3.0)]:
            field = 0.5 + grad[0] * pos[:, 0] + grad[1] * pos[:, 1]
            t = wp.array(field.astype(np.float32), dtype=float, device=device)
            out = wp.zeros(op.node_count, dtype=float, device=device)
            op.apply(_uniform(op, 1.0, device), t, out)

            # Interior nodes only: boundary nodes carry the (nonzero) reaction flux.
            i, j = np.meshgrid(np.arange(res[0] + 1), np.arange(res[1] + 1), indexing="ij")
            interior = ((i > 0) & (i < res[0]) & (j > 0) & (j < res[1])).ravel()
            np.testing.assert_allclose(out.numpy()[interior], 0.0, atol=1.0e-6)


def test_parabolic_manufactured_solution(test, device):
    """Uniform source between fixed walls must reproduce the analytical parabola.

    With ``-k T'' = Q`` on ``[0, 1]``, ``T(0) = T(1) = 0``, the solution is
    ``T = Q x (1 - x) / (2k)``. Bilinear elements reproduce it to nodal accuracy.
    """
    res = (12, 3)
    k_value = 2.5
    q_value = 3.0

    op = thermal.ConductionOperator2D(res, device=device)
    pos = op.node_positions()

    source = _uniform(op, q_value, device)
    rhs_wp = wp.zeros(op.node_count, dtype=float, device=device)
    op.scatter_element_source(source, rhs_wp)

    fixed = ((pos[:, 0] <= 0.0) | (pos[:, 0] >= 1.0)).astype(np.int32)
    values = np.zeros(op.node_count)

    t, _ = _solve_dense(op, _uniform(op, k_value, device), rhs_wp.numpy().astype(np.float64), fixed, values, device)

    exact = q_value * pos[:, 0] * (1.0 - pos[:, 0]) / (2.0 * k_value)
    np.testing.assert_allclose(t, exact, atol=1.0e-6)


def test_series_resistance(test, device):
    """Two materials in series must give the analytical interface temperature.

    For layers of conductivity ``k1`` and ``k2`` each spanning half the domain with
    ``T(0) = 0`` and ``T(1) = 1``, continuity of flux puts the interface at
    ``k2 / (k1 + k2)`` — most of the drop falls across the more resistive layer, so the
    interface sits nearer the hot side when ``k1 < k2``.
    """
    res = (8, 2)
    k1, k2 = 1.0, 4.0

    op = thermal.ConductionOperator2D(res, device=device)
    pos = op.node_positions()

    k = np.empty(op.element_count, dtype=np.float32)
    for i in range(res[0]):
        for j in range(res[1]):
            k[i * res[1] + j] = k1 if i < res[0] // 2 else k2

    fixed = ((pos[:, 0] <= 0.0) | (pos[:, 0] >= 1.0)).astype(np.int32)
    values = np.where(pos[:, 0] >= 1.0, 1.0, 0.0)

    t, _ = _solve_dense(op, wp.array(k, dtype=float, device=device), np.zeros(op.node_count), fixed, values, device)

    interface = np.isclose(pos[:, 0], 0.5)
    np.testing.assert_allclose(t[interface], k2 / (k1 + k2), atol=1.0e-6)


def test_operator_is_symmetric_positive_semidefinite(test, device):
    """Conduction must be symmetric with the constant field in its null space."""
    res = (5, 4)
    op = thermal.ConductionOperator2D(res, device=device)
    rng = np.random.default_rng(3)
    k = wp.array(rng.uniform(0.2, 3.0, op.element_count).astype(np.float32), dtype=float, device=device)

    _, columns = _solve_dense(
        op, k, np.zeros(op.node_count), np.zeros(op.node_count, dtype=np.int32), np.zeros(op.node_count), device
    )

    np.testing.assert_allclose(columns, columns.T, rtol=1.0e-5, atol=1.0e-7)

    # An insulated body has no preferred level: a constant temperature carries no flux.
    ones = wp.array(np.ones(op.node_count, dtype=np.float32), dtype=float, device=device)
    out = wp.zeros(op.node_count, dtype=float, device=device)
    op.apply(k, ones, out)
    np.testing.assert_allclose(out.numpy(), 0.0, atol=1.0e-6)

    eigenvalues = np.linalg.eigvalsh(0.5 * (columns + columns.T))
    test.assertGreater(eigenvalues.min(), -1.0e-6)


def test_diagonal_matches_operator(test, device):
    """The Jacobi diagonal must match the assembled operator's diagonal."""
    res = (6, 5)
    op = thermal.ConductionOperator2D(res, device=device)
    rng = np.random.default_rng(5)
    k = wp.array(rng.uniform(0.2, 3.0, op.element_count).astype(np.float32), dtype=float, device=device)

    diag = wp.zeros(op.node_count, dtype=float, device=device)
    op.diagonal(k, diag)

    _, columns = _solve_dense(
        op, k, np.zeros(op.node_count), np.zeros(op.node_count, dtype=np.int32), np.zeros(op.node_count), device
    )
    np.testing.assert_allclose(diag.numpy(), np.diag(columns), rtol=1.0e-5, atol=1.0e-7)


def test_average_and_scatter_are_adjoint(test, device):
    """Scatter must be the adjoint of element averaging, scaled by element volume.

    Radiation enters the residual through this pair, so a mismatch would make the coupled
    system's transpose inconsistent and silently corrupt the adjoint solve.
    """
    res = (5, 4)
    op = thermal.ConductionOperator2D(res, device=device)
    rng = np.random.default_rng(7)

    t = rng.uniform(-2.0, 2.0, op.node_count).astype(np.float32)
    q = rng.uniform(-2.0, 2.0, op.element_count).astype(np.float32)

    avg = wp.zeros(op.element_count, dtype=float, device=device)
    op.element_average(wp.array(t, dtype=float, device=device), avg)

    scattered = wp.zeros(op.node_count, dtype=float, device=device)
    op.scatter_element_source(wp.array(q, dtype=float, device=device), scattered)

    lhs = float(avg.numpy().astype(np.float64) @ q.astype(np.float64)) * op.element_volume
    rhs = float(t.astype(np.float64) @ scattered.numpy().astype(np.float64))
    test.assertAlmostEqual(lhs, rhs, delta=1.0e-6 * max(1.0, abs(lhs)))


def test_element_average_of_linear_field(test, device):
    """Averaging a linear field must return its value at each element centroid."""
    res = (6, 4)
    op = thermal.ConductionOperator2D(res, device=device)
    pos = op.node_positions()

    field = 1.0 - 2.0 * pos[:, 0] + 3.0 * pos[:, 1]
    avg = wp.zeros(op.element_count, dtype=float, device=device)
    op.element_average(wp.array(field.astype(np.float32), dtype=float, device=device), avg)

    i, j = np.meshgrid(np.arange(res[0]), np.arange(res[1]), indexing="ij")
    cx = (i.ravel() + 0.5) / res[0]
    cy = (j.ravel() + 0.5) / res[1]
    np.testing.assert_allclose(avg.numpy(), 1.0 - 2.0 * cx + 3.0 * cy, rtol=1.0e-5, atol=1.0e-6)


def test_conductivity_vjp_directional(test, device):
    """Directional gradient of the conduction residual with respect to conductivity."""
    res = (5, 5)
    op = thermal.ConductionOperator2D(res, device=device)
    rng = np.random.default_rng(11)

    k = rng.uniform(0.2, 3.0, op.element_count)
    t = rng.uniform(-2.0, 2.0, op.node_count).astype(np.float32)
    lam = rng.uniform(-1.0, 1.0, op.node_count)

    t_wp = wp.array(t, dtype=float, device=device)
    adj_k = wp.zeros(op.element_count, dtype=float, device=device)
    op.apply_vjp(t_wp, wp.array(lam.astype(np.float32), dtype=float, device=device), adj_k)

    def objective(field):
        out = wp.zeros(op.node_count, dtype=float, device=device)
        op.apply(wp.array(field.astype(np.float32), dtype=float, device=device), t_wp, out)
        return float(lam @ out.numpy().astype(np.float64))

    d = rng.normal(size=op.element_count)
    d /= np.linalg.norm(d)
    h = 1.0e-3
    fd = (objective(k + h * d) - objective(k - h * d)) / (2.0 * h)
    ad = float(adj_k.numpy().astype(np.float64) @ d)
    test.assertAlmostEqual(ad, fd, delta=1.0e-3 * max(1.0, abs(fd)))


def test_simp_interpolation(test, device):
    """SIMP must interpolate between the void floor and solid, and penalize in between."""
    n = 5
    rho = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    rho_wp = wp.array(rho, dtype=float, device=device)

    k = wp.zeros(n, dtype=float, device=device)
    thermal.conductivity_from_density(rho_wp, k, penalty=2.0, minimum=1.0e-8)
    expected = 1.0e-8 + (1.0 - 1.0e-8) * rho.astype(np.float64) ** 2
    np.testing.assert_allclose(k.numpy(), expected, rtol=1.0e-6, atol=1.0e-9)

    a = wp.zeros(n, dtype=float, device=device)
    thermal.absorptivity_from_density(rho_wp, a, penalty=2.0)
    np.testing.assert_allclose(a.numpy(), rho.astype(np.float64) ** 2, rtol=1.0e-6, atol=1.0e-9)

    # Penalization is what discourages intermediate densities: half the density must buy
    # less than half the conductivity.
    test.assertLess(k.numpy()[2], 0.5)


def test_dirichlet_mask(test, device):
    res = (4, 4)
    op = thermal.ConductionOperator2D(res, device=device)
    pos = op.node_positions()

    fixed = (pos[:, 0] <= 0.0).astype(np.int32)
    values = np.full(op.node_count, 7.0, dtype=np.float32)
    mask = thermal.DirichletMask(
        wp.array(fixed, dtype=int, device=device), wp.array(values, dtype=float, device=device)
    )

    v = wp.array(np.ones(op.node_count, dtype=np.float32), dtype=float, device=device)
    mask.project(v)
    np.testing.assert_allclose(v.numpy()[fixed != 0], 0.0)
    np.testing.assert_allclose(v.numpy()[fixed == 0], 1.0)

    mask.apply_values(v)
    np.testing.assert_allclose(v.numpy()[fixed != 0], 7.0)
    np.testing.assert_allclose(v.numpy()[fixed == 0], 1.0)


devices = get_test_devices()


class TestThermalConduction(unittest.TestCase):
    pass


add_function_test(TestThermalConduction, "test_patch_test", test_patch_test, devices=devices)
add_function_test(
    TestThermalConduction, "test_parabolic_manufactured_solution", test_parabolic_manufactured_solution, devices=devices
)
add_function_test(TestThermalConduction, "test_series_resistance", test_series_resistance, devices=devices)
add_function_test(
    TestThermalConduction,
    "test_operator_is_symmetric_positive_semidefinite",
    test_operator_is_symmetric_positive_semidefinite,
    devices=devices,
)
add_function_test(
    TestThermalConduction, "test_diagonal_matches_operator", test_diagonal_matches_operator, devices=devices
)
add_function_test(
    TestThermalConduction, "test_average_and_scatter_are_adjoint", test_average_and_scatter_are_adjoint, devices=devices
)
add_function_test(
    TestThermalConduction, "test_element_average_of_linear_field", test_element_average_of_linear_field, devices=devices
)
add_function_test(
    TestThermalConduction, "test_conductivity_vjp_directional", test_conductivity_vjp_directional, devices=devices
)
add_function_test(TestThermalConduction, "test_simp_interpolation", test_simp_interpolation, devices=devices)
add_function_test(TestThermalConduction, "test_dirichlet_mask", test_dirichlet_mask, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
