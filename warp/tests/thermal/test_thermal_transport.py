# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification of the volumetric radiative transport primitive and its hand-written adjoint.

The reference implementations here deliberately use the *explicit sum* form of the transport
integral rather than the compositing recurrence the kernels use, so agreement is evidence
about the math and not just a restatement of the same code. All references run in float64.
"""

import unittest

import numpy as np

import warp as wp
import warp.thermal as thermal
from warp.tests.unittest_utils import *


def _make_random_paths(n_elem, n_rays, rng, device, max_len=12):
    """Random ray bundle, including zero-length paths and repeated cells within a path."""
    source = rng.integers(0, n_elem, n_rays)
    weight = rng.uniform(0.2, 1.5, n_rays)
    exit_kind = rng.integers(0, 2, n_rays)
    paths = [rng.integers(0, n_elem, rng.integers(0, max_len)) for _ in range(n_rays)]

    return (
        thermal.RayPaths.from_numpy(source, weight, exit_kind, paths, device=device),
        {"source": source, "weight": weight, "exit_kind": exit_kind, "paths": paths},
    )


def _ref_forward(spec, n_elem, f, e_pow, e_env):
    """Explicit-sum reference: g = w [ sum_k tau_{k-1} f_k E_k + tau_n E_exit ]."""
    g = np.zeros(n_elem, dtype=np.float64)
    for r, cells in enumerate(spec["paths"]):
        src = spec["source"][r]
        tau = 1.0
        acc = 0.0
        for c in cells:
            acc += tau * f[c] * e_pow[c]
            tau *= 1.0 - f[c]
        acc += tau * (e_pow[src] if spec["exit_kind"][r] == thermal.EXIT_ADIABATIC else e_env)
        g[src] += spec["weight"][r] * acc
    return g


def _ref_exchange_matrix(spec, n_elem, f):
    """Materialize F and F_env explicitly. Used only as a debug oracle, never in production."""
    mat = np.zeros((n_elem, n_elem), dtype=np.float64)
    env = np.zeros(n_elem, dtype=np.float64)
    for r, cells in enumerate(spec["paths"]):
        src = spec["source"][r]
        w = spec["weight"][r]
        tau = 1.0
        for c in cells:
            mat[src, c] += w * tau * f[c]
            tau *= 1.0 - f[c]
        if spec["exit_kind"][r] == thermal.EXIT_ADIABATIC:
            mat[src, src] += w * tau
        else:
            env[src] += w * tau
    return mat, env


def _run_forward(paths, f, e_pow, e_env, device):
    n = f.shape[0]
    out = wp.zeros(n, dtype=float, device=device)
    thermal.transport(
        paths,
        wp.array(f, dtype=float, device=device),
        wp.array(e_pow, dtype=float, device=device),
        out,
        wp.array([e_env], dtype=float, device=device),
    )
    return out.numpy().astype(np.float64)


def _run_vjp(paths, f, e_pow, e_env, adj_g, device):
    n = f.shape[0]
    adj_f = wp.zeros(n, dtype=float, device=device)
    adj_e = wp.zeros(n, dtype=float, device=device)
    adj_env = wp.zeros(1, dtype=float, device=device)
    thermal.transport_vjp(
        paths,
        wp.array(f, dtype=float, device=device),
        wp.array(e_pow, dtype=float, device=device),
        wp.array([e_env], dtype=float, device=device),
        wp.array(adj_g, dtype=float, device=device),
        adj_f,
        adj_e,
        adj_env,
    )
    return (
        adj_f.numpy().astype(np.float64),
        adj_e.numpy().astype(np.float64),
        float(adj_env.numpy()[0]),
    )


def test_forward_matches_explicit_sum(test, device):
    """The compositing recurrence must reproduce the explicit attenuated-sum integral."""
    rng = np.random.default_rng(7)
    n_elem = 24
    paths, spec = _make_random_paths(n_elem, 200, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    e_pow = rng.uniform(0.1, 4.0, n_elem).astype(np.float32)
    e_env = 0.37

    got = _run_forward(paths, f, e_pow, e_env, device)
    expected = _ref_forward(spec, n_elem, f.astype(np.float64), e_pow.astype(np.float64), e_env)

    np.testing.assert_allclose(got, expected, rtol=2.0e-5, atol=1.0e-7)


def test_forward_matches_exchange_matrix(test, device):
    """Matrix-free transport must equal F @ E + F_env * E_env with F built explicitly."""
    rng = np.random.default_rng(11)
    n_elem = 20
    paths, spec = _make_random_paths(n_elem, 150, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    e_pow = rng.uniform(0.1, 4.0, n_elem).astype(np.float32)
    e_env = 1.25

    mat, env = _ref_exchange_matrix(spec, n_elem, f.astype(np.float64))
    expected = mat @ e_pow.astype(np.float64) + env * e_env

    got = _run_forward(paths, f, e_pow, e_env, device)
    np.testing.assert_allclose(got, expected, rtol=2.0e-5, atol=1.0e-7)


def test_vjp_absorptivity_directional(test, device):
    """Headline check: <VJP, d> against a central finite difference along random d."""
    rng = np.random.default_rng(3)
    n_elem = 24
    paths, spec = _make_random_paths(n_elem, 200, rng, device)

    f = rng.uniform(0.1, 0.85, n_elem)
    e_pow = rng.uniform(0.1, 4.0, n_elem)
    e_env = 0.6
    adj_g = rng.uniform(-1.0, 1.0, n_elem)

    adj_f, _, _ = _run_vjp(
        paths, f.astype(np.float32), e_pow.astype(np.float32), e_env, adj_g.astype(np.float32), device
    )

    for trial in range(4):
        d = np.random.default_rng(100 + trial).normal(size=n_elem)
        d /= np.linalg.norm(d)
        h = 1.0e-6
        plus = _ref_forward(spec, n_elem, f + h * d, e_pow, e_env)
        minus = _ref_forward(spec, n_elem, f - h * d, e_pow, e_env)
        fd = float(adj_g @ (plus - minus) / (2.0 * h))
        ad = float(adj_f @ d)
        test.assertAlmostEqual(ad, fd, delta=1.0e-4 * max(1.0, abs(fd)))


def test_vjp_absorptivity_elementwise(test, device):
    """Per-element gradients, to catch bugs a single directional projection could hide."""
    rng = np.random.default_rng(5)
    n_elem = 16
    paths, spec = _make_random_paths(n_elem, 120, rng, device)

    f = rng.uniform(0.1, 0.85, n_elem)
    e_pow = rng.uniform(0.1, 4.0, n_elem)
    e_env = 0.9
    adj_g = rng.uniform(-1.0, 1.0, n_elem)

    adj_f, _, _ = _run_vjp(
        paths, f.astype(np.float32), e_pow.astype(np.float32), e_env, adj_g.astype(np.float32), device
    )

    h = 1.0e-6
    fd = np.zeros(n_elem)
    for k in range(n_elem):
        fp, fm = f.copy(), f.copy()
        fp[k] += h
        fm[k] -= h
        fd[k] = adj_g @ (_ref_forward(spec, n_elem, fp, e_pow, e_env) - _ref_forward(spec, n_elem, fm, e_pow, e_env))
        fd[k] /= 2.0 * h

    np.testing.assert_allclose(adj_f, fd, rtol=2.0e-4, atol=1.0e-6)


def test_vjp_survives_opaque_cells(test, device):
    """Fully opaque cells drive tau to exactly zero; the division-free adjoint must hold up.

    A formulation that recovered the prefix transmittance as tau_n / tau'_k would produce
    NaN or Inf here, which is the reason for the A_k recurrence.
    """
    rng = np.random.default_rng(17)
    n_elem = 18
    paths, spec = _make_random_paths(n_elem, 160, rng, device, max_len=10)

    f = rng.uniform(0.1, 0.6, n_elem)
    f[3] = 1.0
    f[9] = 1.0
    f[14] = 0.0
    e_pow = rng.uniform(0.1, 4.0, n_elem)
    e_env = 0.4
    adj_g = rng.uniform(-1.0, 1.0, n_elem)

    got = _run_forward(paths, f.astype(np.float32), e_pow.astype(np.float32), e_env, device)
    test.assertTrue(np.all(np.isfinite(got)))

    adj_f, adj_e, adj_env = _run_vjp(
        paths, f.astype(np.float32), e_pow.astype(np.float32), e_env, adj_g.astype(np.float32), device
    )
    test.assertTrue(np.all(np.isfinite(adj_f)))
    test.assertTrue(np.all(np.isfinite(adj_e)))
    test.assertTrue(np.isfinite(adj_env))

    h = 1.0e-6
    d = np.random.default_rng(23).normal(size=n_elem)
    # Stay inside [0, 1] when perturbing cells that sit on the bounds.
    d[3] = -abs(d[3])
    d[9] = -abs(d[9])
    d[14] = abs(d[14])
    d /= np.linalg.norm(d)

    plus = _ref_forward(spec, n_elem, f + h * d, e_pow, e_env)
    minus = _ref_forward(spec, n_elem, f - h * d, e_pow, e_env)
    fd = float(adj_g @ (plus - minus) / (2.0 * h))
    test.assertAlmostEqual(float(adj_f @ d), fd, delta=1.0e-4 * max(1.0, abs(fd)))


def test_transport_is_linear_in_emissive(test, device):
    """Irradiation is linear in emissive power, so the forward call doubles as the tangent."""
    rng = np.random.default_rng(29)
    n_elem = 20
    paths, _ = _make_random_paths(n_elem, 150, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    a = rng.uniform(0.1, 4.0, n_elem).astype(np.float32)
    b = rng.uniform(0.1, 4.0, n_elem).astype(np.float32)

    ga = _run_forward(paths, f, a, 0.0, device)
    gb = _run_forward(paths, f, b, 0.0, device)
    gab = _run_forward(paths, f, (2.0 * a + 3.0 * b).astype(np.float32), 0.0, device)

    np.testing.assert_allclose(gab, 2.0 * ga + 3.0 * gb, rtol=2.0e-5, atol=1.0e-6)


def test_transpose_adjoint_identity(test, device):
    """<J u, v> == <u, J^T v>, the test finite differences cannot perform.

    The radiative tangent is nonsymmetric, so a wrong transpose would still pass every
    forward and FD check while quietly corrupting the adjoint solve.
    """
    rng = np.random.default_rng(31)
    n_elem = 22
    paths, _ = _make_random_paths(n_elem, 180, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    u = rng.uniform(-2.0, 2.0, n_elem).astype(np.float32)
    v = rng.uniform(-2.0, 2.0, n_elem).astype(np.float32)

    f_wp = wp.array(f, dtype=float, device=device)

    # J u, via the forward operator with a zero environment term.
    ju = _run_forward(paths, f, u, 0.0, device)

    # J^T v
    out = wp.zeros(n_elem, dtype=float, device=device)
    out_env = wp.zeros(1, dtype=float, device=device)
    thermal.transport_transpose(paths, f_wp, wp.array(v, dtype=float, device=device), out, out_env)
    jtv = out.numpy().astype(np.float64)

    lhs = float(ju @ v.astype(np.float64))
    rhs = float(u.astype(np.float64) @ jtv)
    test.assertAlmostEqual(lhs, rhs, delta=1.0e-4 * max(1.0, abs(lhs)))


def test_vjp_emissive_matches_transpose(test, device):
    """The emissive half of the VJP must coincide with the transpose operator."""
    rng = np.random.default_rng(37)
    n_elem = 22
    paths, _ = _make_random_paths(n_elem, 180, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    e_pow = rng.uniform(0.1, 4.0, n_elem).astype(np.float32)
    adj_g = rng.uniform(-1.0, 1.0, n_elem).astype(np.float32)

    _, adj_e, adj_env = _run_vjp(paths, f, e_pow, 0.75, adj_g, device)

    out = wp.zeros(n_elem, dtype=float, device=device)
    out_env = wp.zeros(1, dtype=float, device=device)
    thermal.transport_transpose(
        paths,
        wp.array(f, dtype=float, device=device),
        wp.array(adj_g, dtype=float, device=device),
        out,
        out_env,
    )

    np.testing.assert_allclose(adj_e, out.numpy().astype(np.float64), rtol=2.0e-5, atol=1.0e-6)
    test.assertAlmostEqual(adj_env, float(out_env.numpy()[0]), delta=1.0e-5)


def test_grid_dda_paths(test, device):
    """End to end on real DDA traversals of a structured grid, not synthetic index lists."""
    res = (8, 8)
    n_elem = res[0] * res[1]
    rng = np.random.default_rng(41)

    source, weight, exit_kind, paths = [], [], [], []
    for i in range(res[0]):
        for j in range(res[1]):
            origin = ((i + 0.5) / res[0], (j + 0.5) / res[1])
            for c in range(12):
                angle = 2.0 * np.pi * (c + 0.5) / 12.0
                direction = (np.cos(angle), np.sin(angle))
                source.append(i * res[1] + j)
                weight.append(1.0 / 12.0)
                # Deep space above the horizon, adiabatic below, as in the paper's setup.
                exit_kind.append(thermal.EXIT_ENVIRONMENT if direction[1] > 0.0 else thermal.EXIT_ADIABATIC)
                paths.append(thermal.trace_grid_2d(res, origin, direction))

    ray_paths = thermal.RayPaths.from_numpy(source, weight, exit_kind, paths, device=device)
    spec = {"source": np.array(source), "weight": np.array(weight), "exit_kind": np.array(exit_kind), "paths": paths}

    f = rng.uniform(0.05, 0.95, n_elem)
    e_pow = rng.uniform(0.5, 3.0, n_elem)
    e_env = 0.0
    adj_g = rng.uniform(-1.0, 1.0, n_elem)

    got = _run_forward(ray_paths, f.astype(np.float32), e_pow.astype(np.float32), e_env, device)
    np.testing.assert_allclose(got, _ref_forward(spec, n_elem, f, e_pow, e_env), rtol=2.0e-5, atol=1.0e-7)

    adj_f, _, _ = _run_vjp(
        ray_paths, f.astype(np.float32), e_pow.astype(np.float32), e_env, adj_g.astype(np.float32), device
    )

    d = np.random.default_rng(43).normal(size=n_elem)
    d /= np.linalg.norm(d)
    h = 1.0e-6
    plus = _ref_forward(spec, n_elem, f + h * d, e_pow, e_env)
    minus = _ref_forward(spec, n_elem, f - h * d, e_pow, e_env)
    fd = float(adj_g @ (plus - minus) / (2.0 * h))
    test.assertAlmostEqual(float(adj_f @ d), fd, delta=1.0e-4 * max(1.0, abs(fd)))


def test_dda_stays_in_bounds(test, device):
    """The path builder must never emit an out-of-range cell, whatever the direction."""
    res = (11, 7)
    rng = np.random.default_rng(47)
    for _ in range(200):
        origin = (rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0))
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cells = thermal.trace_grid_2d(res, origin, (np.cos(angle), np.sin(angle)))
        if cells:
            test.assertGreaterEqual(min(cells), 0)
            test.assertLess(max(cells), res[0] * res[1])


devices = get_test_devices()


class TestThermalTransport(unittest.TestCase):
    pass


add_function_test(
    TestThermalTransport, "test_forward_matches_explicit_sum", test_forward_matches_explicit_sum, devices=devices
)
add_function_test(
    TestThermalTransport, "test_forward_matches_exchange_matrix", test_forward_matches_exchange_matrix, devices=devices
)
add_function_test(
    TestThermalTransport, "test_vjp_absorptivity_directional", test_vjp_absorptivity_directional, devices=devices
)
add_function_test(
    TestThermalTransport, "test_vjp_absorptivity_elementwise", test_vjp_absorptivity_elementwise, devices=devices
)
add_function_test(
    TestThermalTransport, "test_vjp_survives_opaque_cells", test_vjp_survives_opaque_cells, devices=devices
)
add_function_test(
    TestThermalTransport, "test_transport_is_linear_in_emissive", test_transport_is_linear_in_emissive, devices=devices
)
add_function_test(
    TestThermalTransport, "test_transpose_adjoint_identity", test_transpose_adjoint_identity, devices=devices
)
add_function_test(
    TestThermalTransport, "test_vjp_emissive_matches_transpose", test_vjp_emissive_matches_transpose, devices=devices
)
add_function_test(TestThermalTransport, "test_grid_dda_paths", test_grid_dda_paths, devices=devices)
add_function_test(TestThermalTransport, "test_dda_stays_in_bounds", test_dda_stays_in_bounds, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
