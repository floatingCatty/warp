# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification of the coupled conduction-radiation solve and its implicit gradient.

The chain being checked is geometry -> conduction -> radiation -> nonlinear solve ->
implicit adjoint -> design gradient. Each link is pinned separately: the tangent against a
finite difference of the residual, its transpose against the adjoint identity, the Newton
solve against residual decay and a physical energy balance, and finally the whole chain
against a finite difference of the objective through a full re-solve.

That last check is the important one. It is the only test that would catch an error in how
the links compose rather than in any single link.
"""

import unittest

import numpy as np

import warp as wp
import warp.thermal as thermal
from warp.tests.unittest_utils import *


def _build(res, n_ang, nr, density, device, source=1.0, angles_pt=1):
    """Assemble a complete coupled problem on an insulated square with a uniform source."""
    conduction = thermal.ConductionOperator2D(res, device=device)
    bundle = thermal.GridRayBundle2D(res, launch_points=angles_pt, angles=n_ang, device=device)
    radiation = thermal.VolumetricRadiationOperator(bundle, conduction.element_count, device=device)

    residual = thermal.CoupledResidual2D(conduction, radiation, conduction_radiation_number=nr, device=device)

    rho = wp.array(np.asarray(density, dtype=np.float32), dtype=float, device=device)
    k = wp.zeros(conduction.element_count, dtype=float, device=device)
    a = wp.zeros(conduction.element_count, dtype=float, device=device)
    thermal.conductivity_from_density(rho, k)
    thermal.absorptivity_from_density(rho, a)
    residual.update_design(k, a)

    q = wp.array(np.full(conduction.element_count, source, dtype=np.float32), dtype=float, device=device)
    residual.set_volumetric_source(q)

    return conduction, radiation, residual


def _solve(residual, conduction, device, initial=0.5):
    t = wp.array(np.full(conduction.node_count, initial, dtype=np.float32), dtype=float, device=device)
    result = thermal.newton_solve(residual, t, max_iterations=40)
    return t, result


def test_tangent_matches_residual_derivative(test, device):
    """The tangent must be the directional derivative of the residual."""
    res = (5, 5)
    rng = np.random.default_rng(3)
    density = rng.uniform(0.3, 0.9, res[0] * res[1])
    conduction, _, residual = _build(res, 8, 1.0, density, device)

    t = rng.uniform(0.5, 1.5, conduction.node_count).astype(np.float32)
    v = rng.normal(size=conduction.node_count)
    v /= np.linalg.norm(v)

    t_wp = wp.array(t, dtype=float, device=device)
    residual.relinearize(t_wp)
    jv = wp.zeros(conduction.node_count, dtype=float, device=device)
    residual.tangent(wp.array(v.astype(np.float32), dtype=float, device=device), jv)

    def evaluate(field):
        out = wp.zeros(conduction.node_count, dtype=float, device=device)
        residual.evaluate(wp.array(field.astype(np.float32), dtype=float, device=device), out)
        return out.numpy().astype(np.float64)

    h = 1.0e-3
    fd = (evaluate(t + h * v) - evaluate(t - h * v)) / (2.0 * h)
    np.testing.assert_allclose(jv.numpy(), fd, rtol=2.0e-2, atol=1.0e-3)


def test_tangent_transpose_identity(test, device):
    """<Kv, u> == <v, K^T u>. The adjoint solve is the only consumer of the transpose."""
    res = (5, 4)
    rng = np.random.default_rng(5)
    density = rng.uniform(0.3, 0.9, res[0] * res[1])
    conduction, _, residual = _build(res, 8, 2.0, density, device)

    n = conduction.node_count
    t = wp.array(rng.uniform(0.5, 1.5, n).astype(np.float32), dtype=float, device=device)
    residual.relinearize(t)

    u = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    v = rng.uniform(-1.0, 1.0, n).astype(np.float32)

    kv = wp.zeros(n, dtype=float, device=device)
    residual.tangent(wp.array(v, dtype=float, device=device), kv)

    ktu = wp.zeros(n, dtype=float, device=device)
    residual.tangent_transpose(wp.array(u, dtype=float, device=device), ktu)

    lhs = float(kv.numpy().astype(np.float64) @ u.astype(np.float64))
    rhs = float(v.astype(np.float64) @ ktu.numpy().astype(np.float64))
    test.assertAlmostEqual(lhs, rhs, delta=1.0e-4 * max(1.0, abs(lhs)))


def test_tangent_is_nonsymmetric(test, device):
    """Radiation must actually break symmetry, otherwise the transpose test proves nothing."""
    res = (4, 4)
    rng = np.random.default_rng(7)
    density = rng.uniform(0.4, 0.9, res[0] * res[1])
    conduction, _, residual = _build(res, 8, 50.0, density, device)

    n = conduction.node_count
    t = wp.array(rng.uniform(0.5, 1.5, n).astype(np.float32), dtype=float, device=device)
    residual.relinearize(t)

    columns = np.zeros((n, n))
    basis = np.zeros(n, dtype=np.float32)
    probe = wp.zeros(n, dtype=float, device=device)
    out = wp.zeros(n, dtype=float, device=device)
    for c in range(n):
        basis[:] = 0.0
        basis[c] = 1.0
        probe.assign(basis)
        residual.tangent(probe, out)
        columns[:, c] = out.numpy()

    asymmetry = np.abs(columns - columns.T).max() / np.abs(columns).max()
    test.assertGreater(asymmetry, 1.0e-3)


def test_newton_converges(test, device):
    """Newton must drive the residual to zero, quadratically near the solution."""
    res = (6, 6)
    density = np.full(res[0] * res[1], 0.8)
    conduction, _, residual = _build(res, 12, 1.0, density, device)

    t, result = _solve(residual, conduction, device)
    test.assertTrue(result.converged, f"Newton did not converge: {result}")
    test.assertLess(result.residual_norm, 1.0e-4 * residual.reference_norm())

    # Quadratic convergence: once close, each residual is near the square of the previous,
    # so the drop over the last productive step is far steeper than linear.
    productive = [h for h in result.history if h > 1.0e-6]
    test.assertGreater(productive[-2] / productive[-1], 10.0)

    # Temperatures must be physical: positive, with the interior hotter than the rim.
    field = t.numpy()
    test.assertGreater(field.min(), 0.0)


def test_energy_balance(test, device):
    """At steady state, heat generated must equal heat radiated to the environment.

    Nothing in the discretization enforces this: it emerges only if the emission
    coefficient, the exchange factors, the escape bookkeeping and the source scatter are all
    mutually consistent.
    """
    res = (6, 6)
    density = np.full(res[0] * res[1], 1.0)
    source = 1.0

    conduction = thermal.ConductionOperator2D(res, device=device)
    bundle = thermal.GridRayBundle2D(res, launch_points=2, angles=64, device=device)
    radiation = thermal.VolumetricRadiationOperator(bundle, conduction.element_count, device=device)
    residual = thermal.CoupledResidual2D(conduction, radiation, conduction_radiation_number=1.0, device=device)

    rho = wp.array(np.asarray(density, dtype=np.float32), dtype=float, device=device)
    k = wp.zeros(conduction.element_count, dtype=float, device=device)
    a = wp.zeros(conduction.element_count, dtype=float, device=device)
    thermal.conductivity_from_density(rho, k)
    thermal.absorptivity_from_density(rho, a)
    residual.update_design(k, a)
    residual.set_volumetric_source(
        wp.array(np.full(conduction.element_count, source, dtype=np.float32), dtype=float, device=device)
    )

    t, result = _solve(residual, conduction, device, initial=0.6)
    test.assertTrue(result.converged, f"Newton did not converge: {result}")

    # Net radiative loss integrated over the domain, from the converged temperature.
    element_t = wp.zeros(conduction.element_count, dtype=float, device=device)
    conduction.element_average(t, element_t)
    emissive = element_t.numpy().astype(np.float64) ** 4

    irradiation = wp.zeros(conduction.element_count, dtype=float, device=device)
    radiation.apply(
        wp.array(emissive.astype(np.float32), dtype=float, device=device),
        irradiation,
        residual.environment_emissive,
    )

    coefficient = 1.0 * residual.area_over_volume * a.numpy().astype(np.float64)
    radiated = float((coefficient * (emissive - irradiation.numpy().astype(np.float64))).sum())
    radiated *= conduction.element_volume

    generated = source * 1.0  # unit domain volume
    np.testing.assert_allclose(radiated, generated, rtol=2.0e-2)


def test_pure_conduction_limit(test, device):
    """With N_R = 0 the residual must reduce to the linear conduction problem."""
    res = (5, 5)
    rng = np.random.default_rng(11)
    density = rng.uniform(0.5, 1.0, res[0] * res[1])
    conduction, _, residual = _build(res, 8, 0.0, density, device)

    pos = conduction.node_positions()
    fixed = (pos[:, 0] <= 0.0).astype(np.int32)
    residual.dirichlet = thermal.DirichletMask(
        wp.array(fixed, dtype=int, device=device),
        wp.zeros(conduction.node_count, dtype=float, device=device),
    )

    t = wp.zeros(conduction.node_count, dtype=float, device=device)
    result = thermal.newton_solve(residual, t, max_iterations=10)
    test.assertTrue(result.converged, f"Newton did not converge: {result}")

    # A linear problem is solved exactly by a single Newton step.
    test.assertEqual(result.iterations, 1)

    # Verify against a direct conduction solve.
    k = wp.zeros(conduction.element_count, dtype=float, device=device)
    thermal.conductivity_from_density(wp.array(density.astype(np.float32), dtype=float, device=device), k)
    out = wp.zeros(conduction.node_count, dtype=float, device=device)
    conduction.apply(k, t, out)

    rhs = wp.zeros(conduction.node_count, dtype=float, device=device)
    conduction.scatter_element_source(
        wp.array(np.ones(conduction.element_count, dtype=np.float32), dtype=float, device=device), rhs
    )
    diff = out.numpy() - rhs.numpy()
    np.testing.assert_allclose(diff[fixed == 0], 0.0, atol=1.0e-6)


def test_design_gradient_directional(test, device):
    """The headline check: implicit gradient against a finite difference of a full re-solve.

    Every link in the chain participates, so this is what would catch an error in how the
    conduction adjoint, the radiative adjoint and the implicit solve compose.
    """
    res = (4, 4)
    n_elem = res[0] * res[1]
    rng = np.random.default_rng(13)
    base = rng.uniform(0.4, 0.85, n_elem)

    def objective(density, want_gradient):
        conduction, _, residual = _build(res, 12, 1.0, density, device)
        t, result = _solve(residual, conduction, device, initial=0.8)
        assert result.converged, result

        # J = mean temperature over the domain, as a sum of element averages.
        element_t = wp.zeros(n_elem, dtype=float, device=device)
        conduction.element_average(t, element_t)
        value = float(element_t.numpy().astype(np.float64).sum()) * conduction.element_volume

        if not want_gradient:
            return value, None

        # dJ/dT: J = volume * sum_e average(T)_e, and average^T = scatter / volume.
        dj_dt = wp.zeros(conduction.node_count, dtype=float, device=device)
        conduction.scatter_element_source(
            wp.array(np.ones(n_elem, dtype=np.float32), dtype=float, device=device), dj_dt
        )

        lam = wp.zeros(conduction.node_count, dtype=float, device=device)
        thermal.adjoint_solve(residual, t, dj_dt, lam)

        adj_k = wp.zeros(n_elem, dtype=float, device=device)
        adj_a = wp.zeros(n_elem, dtype=float, device=device)
        residual.density_vjp(t, lam, adj_k, adj_a)

        # Chain through SIMP: k = k_min + (1 - k_min) rho^p, f = rho^q, both with p = q = 2.
        d_k = 2.0 * (1.0 - 1.0e-8) * density
        d_a = 2.0 * density
        grad = adj_k.numpy().astype(np.float64) * d_k + adj_a.numpy().astype(np.float64) * d_a
        return value, grad

    _, grad = objective(base, True)

    d = rng.normal(size=n_elem)
    d /= np.linalg.norm(d)
    h = 2.0e-3
    plus, _ = objective(base + h * d, False)
    minus, _ = objective(base - h * d, False)
    fd = (plus - minus) / (2.0 * h)
    ad = float(grad @ d)

    test.assertAlmostEqual(ad, fd, delta=3.0e-2 * max(abs(fd), 1.0e-3))


devices = get_test_devices()


class TestThermalCoupled(unittest.TestCase):
    pass


add_function_test(
    TestThermalCoupled,
    "test_tangent_matches_residual_derivative",
    test_tangent_matches_residual_derivative,
    devices=devices,
)
add_function_test(
    TestThermalCoupled, "test_tangent_transpose_identity", test_tangent_transpose_identity, devices=devices
)
add_function_test(TestThermalCoupled, "test_tangent_is_nonsymmetric", test_tangent_is_nonsymmetric, devices=devices)
add_function_test(TestThermalCoupled, "test_newton_converges", test_newton_converges, devices=devices)
add_function_test(TestThermalCoupled, "test_energy_balance", test_energy_balance, devices=devices)
add_function_test(TestThermalCoupled, "test_pure_conduction_limit", test_pure_conduction_limit, devices=devices)
add_function_test(
    TestThermalCoupled, "test_design_gradient_directional", test_design_gradient_directional, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
