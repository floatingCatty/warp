# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification of the density filter and the optimality criteria update.

These two close the design loop. The filter must be a genuine transpose pair, since the
sensitivity is chained through it, and the update must hold the volume constraint exactly,
since the whole optimization is posed relative to an active constraint.
"""

import unittest

import numpy as np

import warp as wp
import warp.thermal as thermal
from warp.tests.unittest_utils import *


def _reference_filter(res, radius):
    """Dense filter matrix, built independently from element centre distances."""
    nx, ny = res
    n = nx * ny
    centres = np.array([[(i + 0.5) / nx, (j + 0.5) / ny] for i in range(nx) for j in range(ny)])
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    w = np.maximum(0.0, radius - d)
    return w / w.sum(axis=1, keepdims=True)


def test_filter_matches_reference(test, device):
    """The neighborhood traversal must reproduce the dense filter matrix."""
    for res, radius in [((6, 5), 0.25), ((8, 8), 0.12)]:
        n = res[0] * res[1]
        filt = thermal.DensityFilter2D(res, radius, device=device)
        rng = np.random.default_rng(3)
        rho = rng.uniform(0.0, 1.0, n).astype(np.float32)

        out = wp.zeros(n, dtype=float, device=device)
        filt.apply(wp.array(rho, dtype=float, device=device), out)

        expected = _reference_filter(res, radius) @ rho.astype(np.float64)
        np.testing.assert_allclose(out.numpy(), expected, rtol=2.0e-5, atol=1.0e-7)


def test_filter_preserves_constants(test, device):
    """Weights are normalized, so a uniform design must survive filtering unchanged.

    Without this the filter would bias the volume constraint, and the optimizer would chase
    a target it can never reach.
    """
    res, radius = (7, 6), 0.2
    n = res[0] * res[1]
    filt = thermal.DensityFilter2D(res, radius, device=device)

    out = wp.zeros(n, dtype=float, device=device)
    filt.apply(wp.array(np.full(n, 0.37, dtype=np.float32), dtype=float, device=device), out)
    np.testing.assert_allclose(out.numpy(), 0.37, rtol=1.0e-6)


def test_filter_transpose_identity(test, device):
    """<H rho, y> == <rho, H^T y>. The sensitivity is chained through this transpose."""
    res, radius = (6, 5), 0.22
    n = res[0] * res[1]
    filt = thermal.DensityFilter2D(res, radius, device=device)
    rng = np.random.default_rng(5)

    rho = rng.uniform(0.0, 1.0, n).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, n).astype(np.float32)

    h_rho = wp.zeros(n, dtype=float, device=device)
    filt.apply(wp.array(rho, dtype=float, device=device), h_rho)

    ht_y = wp.zeros(n, dtype=float, device=device)
    filt.apply_transpose(wp.array(y, dtype=float, device=device), ht_y)

    lhs = float(h_rho.numpy().astype(np.float64) @ y.astype(np.float64))
    rhs = float(rho.astype(np.float64) @ ht_y.numpy().astype(np.float64))
    test.assertAlmostEqual(lhs, rhs, delta=1.0e-5 * max(1.0, abs(lhs)))


def test_filter_spreads_a_point(test, device):
    """A single dense element must spread over the filter radius and nowhere further."""
    res, radius = (11, 11), 0.2
    n = res[0] * res[1]
    filt = thermal.DensityFilter2D(res, radius, device=device)

    rho = np.zeros(n, dtype=np.float32)
    rho[5 * res[1] + 5] = 1.0
    out = wp.zeros(n, dtype=float, device=device)
    filt.apply(wp.array(rho, dtype=float, device=device), out)
    field = out.numpy().reshape(res)

    test.assertGreater(field[5, 5], 0.0)
    test.assertGreater(field[5, 5], field[5, 6])
    test.assertGreater(field[5, 6], field[5, 7])

    # Beyond the radius the cone weight is zero, so nothing may leak.
    centres = np.array([[(i + 0.5) / res[0], (j + 0.5) / res[1]] for i in range(res[0]) for j in range(res[1])])
    far = np.linalg.norm(centres - centres[5 * res[1] + 5], axis=1) >= radius
    np.testing.assert_allclose(out.numpy()[far], 0.0, atol=1.0e-9)


def test_oc_meets_volume_constraint(test, device):
    """The update must land exactly on the volume target, whatever the sensitivity."""
    n = 400
    rng = np.random.default_rng(7)

    for fraction in (0.2, 0.4, 0.7):
        density = wp.array(rng.uniform(0.2, 0.8, n).astype(np.float32), dtype=float, device=device)
        sensitivity = wp.array(-rng.uniform(0.01, 5.0, n).astype(np.float32), dtype=float, device=device)

        thermal.optimality_criteria_step(density, sensitivity, fraction, move=1.0)
        test.assertAlmostEqual(float(density.numpy().mean()), fraction, delta=1.0e-4)


def test_oc_respects_bounds_and_move_limit(test, device):
    """Densities must stay in range and no element may move further than the limit."""
    n = 200
    rng = np.random.default_rng(11)
    start = rng.uniform(0.1, 0.9, n).astype(np.float32)
    density = wp.array(start.copy(), dtype=float, device=device)
    sensitivity = wp.array(-rng.uniform(0.001, 100.0, n).astype(np.float32), dtype=float, device=device)

    move = 0.15
    thermal.optimality_criteria_step(density, sensitivity, 0.5, move=move)
    result = density.numpy()

    test.assertGreaterEqual(result.min(), -1.0e-6)
    test.assertLessEqual(result.max(), 1.0 + 1.0e-6)
    test.assertLessEqual(np.abs(result - start).max(), move + 1.0e-6)


def test_oc_favors_sensitive_elements(test, device):
    """Material must flow toward elements where it does the most good.

    This is the whole content of the optimality condition: at fixed volume, an element with
    a stronger objective improvement per unit volume must end up denser.
    """
    n = 100
    density = wp.array(np.full(n, 0.5, dtype=np.float32), dtype=float, device=device)

    sens = np.full(n, -1.0, dtype=np.float32)
    sens[:10] = -100.0  # strongly beneficial
    sens[-10:] = -0.01  # nearly useless
    thermal.optimality_criteria_step(density, wp.array(sens, dtype=float, device=device), 0.5, move=0.2)

    result = density.numpy()
    test.assertGreater(result[:10].mean(), result[10:-10].mean())
    test.assertGreater(result[10:-10].mean(), result[-10:].mean())


def test_oc_ignores_ascent_directions(test, device):
    """Elements whose sensitivity says to add material to worsen the objective must not grow."""
    n = 50
    density = wp.array(np.full(n, 0.5, dtype=np.float32), dtype=float, device=device)
    sens = np.full(n, -1.0, dtype=np.float32)
    sens[:5] = 10.0  # positive: adding material here would hurt
    thermal.optimality_criteria_step(density, wp.array(sens, dtype=float, device=device), 0.5, move=0.5)

    test.assertLess(density.numpy()[:5].max(), 1.0e-6)


def test_example_objective_gradient(test, device):
    """The example's dJ/dT must be the derivative of its reported objective.

    This is the one link of the example's chain not already covered: the density VJP is
    finite-differenced in the coupled tests, the filter transpose has its own adjoint
    identity, and the SIMP derivatives are analytic. Differencing the objective with respect
    to the temperature field directly avoids going through the solver, where the objective's
    response to a design perturbation sits at the single-precision noise floor and a finite
    difference cannot resolve it.
    """
    import warp.examples.thermal.example_radiative_heat_sink as heat_sink

    e = heat_sink.Example(
        resolution=8, launch_points=1, angles=12, source_width=4, filter_radius=0.2, quiet=True, device=device
    )
    rng = np.random.default_rng(211)
    t = rng.uniform(0.5, 1.5, e.conduction.node_count)

    def objective(field):
        e.temperature.assign(field.astype(np.float32))
        return e.objective()

    grad = e.objective_gradient.numpy().astype(np.float64)

    d = rng.normal(size=e.conduction.node_count)
    d /= np.linalg.norm(d)
    # The objective is exactly linear in temperature, so a large step carries no truncation
    # error and keeps single-precision noise from dominating the difference quotient.
    h = 0.25
    fd = (objective(t + h * d) - objective(t - h * d)) / (2.0 * h)
    test.assertAlmostEqual(float(grad @ d), fd, delta=1.0e-5 * max(abs(fd), 1.0e-6))


def test_example_objective_is_an_integral(test, device):
    """The reported objective must be the temperature integral, not its mean.

    The reference result this example is compared against is an integral over the target
    region, so a mean would differ from it by the region's volume and invite a false
    conclusion either way.
    """
    import warp.examples.thermal.example_radiative_heat_sink as heat_sink

    e = heat_sink.Example(
        resolution=8, launch_points=1, angles=12, source_width=4, filter_radius=0.2, quiet=True, device=device
    )
    e.temperature.assign(np.full(e.conduction.node_count, 2.0, dtype=np.float32))

    expected = 2.0 * e.objective_mask.sum() * e.conduction.element_volume
    test.assertAlmostEqual(e.objective(), expected, delta=1.0e-9)


devices = get_test_devices()


class TestThermalOptimization(unittest.TestCase):
    pass


add_function_test(
    TestThermalOptimization, "test_filter_matches_reference", test_filter_matches_reference, devices=devices
)
add_function_test(
    TestThermalOptimization, "test_filter_preserves_constants", test_filter_preserves_constants, devices=devices
)
add_function_test(
    TestThermalOptimization, "test_filter_transpose_identity", test_filter_transpose_identity, devices=devices
)
add_function_test(TestThermalOptimization, "test_filter_spreads_a_point", test_filter_spreads_a_point, devices=devices)
add_function_test(
    TestThermalOptimization, "test_oc_meets_volume_constraint", test_oc_meets_volume_constraint, devices=devices
)
add_function_test(
    TestThermalOptimization,
    "test_oc_respects_bounds_and_move_limit",
    test_oc_respects_bounds_and_move_limit,
    devices=devices,
)
add_function_test(
    TestThermalOptimization, "test_oc_favors_sensitive_elements", test_oc_favors_sensitive_elements, devices=devices
)
add_function_test(
    TestThermalOptimization, "test_oc_ignores_ascent_directions", test_oc_ignores_ascent_directions, devices=devices
)
add_function_test(
    TestThermalOptimization, "test_example_objective_gradient", test_example_objective_gradient, devices=devices
)
add_function_test(
    TestThermalOptimization,
    "test_example_objective_is_an_integral",
    test_example_objective_is_an_integral,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
