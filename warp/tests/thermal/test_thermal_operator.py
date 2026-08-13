# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification that the assembled and marching transport strategies are interchangeable.

The operator picks its forward strategy by problem size, so the two must be
indistinguishable to callers. Any divergence would show up as a design iteration whose
Newton solve and adjoint disagree about the physics, which is far harder to diagnose than
an outright failure here.
"""

import unittest

import numpy as np

import warp as wp
import warp.thermal as thermal
from warp.tests.unittest_utils import *


def _make_random_paths(n_elem, n_rays, rng, device, max_len=12):
    source = rng.integers(0, n_elem, n_rays)
    weight = rng.uniform(0.2, 1.5, n_rays)
    exit_kind = rng.integers(0, 2, n_rays)
    paths = [rng.integers(0, n_elem, rng.integers(0, max_len)) for _ in range(n_rays)]

    return (
        thermal.RayPaths.from_numpy(source, weight, exit_kind, paths, device=device),
        {"source": source, "weight": weight, "exit_kind": exit_kind, "paths": paths},
    )


def _ref_exchange_matrix(spec, n_elem, f):
    """Independent construction of F, used as an oracle for the assembly kernel."""
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


def _make_operator(paths, n_elem, backend, absorptivity, device):
    op = thermal.VolumetricRadiationOperator(paths, n_elem, backend=backend, device=device)
    op.update(wp.array(absorptivity, dtype=float, device=device))
    return op


def test_assembly_matches_reference(test, device):
    """The assembly kernel must reproduce F and F_env built independently."""
    rng = np.random.default_rng(101)
    n_elem = 20
    paths, spec = _make_random_paths(n_elem, 150, rng, device)
    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)

    exchange = thermal.ExchangeMatrix(n_elem, device=device)
    exchange.assemble(paths, wp.array(f, dtype=float, device=device))

    mat, env = _ref_exchange_matrix(spec, n_elem, f.astype(np.float64))
    np.testing.assert_allclose(exchange.factors.numpy(), mat, rtol=2.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(exchange.environment_factors.numpy(), env, rtol=2.0e-5, atol=1.0e-7)


def test_backends_agree_on_apply(test, device):
    """Assembled and marching forward applies must agree."""
    rng = np.random.default_rng(103)
    n_elem = 24
    paths, _ = _make_random_paths(n_elem, 200, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    e_pow = rng.uniform(0.1, 4.0, n_elem).astype(np.float32)
    e_env = wp.array([0.73], dtype=float, device=device)

    results = {}
    for backend in ("assembled", "march"):
        op = _make_operator(paths, n_elem, backend, f, device)
        out = wp.zeros(n_elem, dtype=float, device=device)
        op.apply(wp.array(e_pow, dtype=float, device=device), out, e_env)
        results[backend] = out.numpy().astype(np.float64)

    np.testing.assert_allclose(results["assembled"], results["march"], rtol=2.0e-5, atol=1.0e-7)


def test_backends_agree_on_transpose(test, device):
    """Assembled and marching transposes must agree, including the environment component."""
    rng = np.random.default_rng(107)
    n_elem = 24
    paths, _ = _make_random_paths(n_elem, 200, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    v = rng.uniform(-2.0, 2.0, n_elem).astype(np.float32)

    results = {}
    for backend in ("assembled", "march"):
        op = _make_operator(paths, n_elem, backend, f, device)
        out = wp.zeros(n_elem, dtype=float, device=device)
        out_env = wp.zeros(1, dtype=float, device=device)
        op.apply_transpose(wp.array(v, dtype=float, device=device), out, out_env)
        results[backend] = (out.numpy().astype(np.float64), float(out_env.numpy()[0]))

    np.testing.assert_allclose(results["assembled"][0], results["march"][0], rtol=2.0e-5, atol=1.0e-7)
    test.assertAlmostEqual(results["assembled"][1], results["march"][1], delta=1.0e-5)


def test_assembled_backend_adjoint_identity(test, device):
    """<J u, v> == <u, J^T v> must hold for the assembled path too, not just the march."""
    rng = np.random.default_rng(109)
    n_elem = 22
    paths, _ = _make_random_paths(n_elem, 180, rng, device)

    f = rng.uniform(0.05, 0.9, n_elem).astype(np.float32)
    u = rng.uniform(-2.0, 2.0, n_elem).astype(np.float32)
    v = rng.uniform(-2.0, 2.0, n_elem).astype(np.float32)

    op = _make_operator(paths, n_elem, "assembled", f, device)
    zero_env = wp.array([0.0], dtype=float, device=device)

    ju = wp.zeros(n_elem, dtype=float, device=device)
    op.apply(wp.array(u, dtype=float, device=device), ju, zero_env)

    jtv = wp.zeros(n_elem, dtype=float, device=device)
    jtv_env = wp.zeros(1, dtype=float, device=device)
    op.apply_transpose(wp.array(v, dtype=float, device=device), jtv, jtv_env)

    lhs = float(ju.numpy().astype(np.float64) @ v.astype(np.float64))
    rhs = float(u.astype(np.float64) @ jtv.numpy().astype(np.float64))
    test.assertAlmostEqual(lhs, rhs, delta=1.0e-4 * max(1.0, abs(lhs)))


def test_vjp_independent_of_forward_backend(test, device):
    """The adjoint always marches, so the forward strategy must not perturb it."""
    rng = np.random.default_rng(113)
    n_elem = 24
    paths, _ = _make_random_paths(n_elem, 200, rng, device)

    f = rng.uniform(0.1, 0.85, n_elem).astype(np.float32)
    e_pow = rng.uniform(0.1, 4.0, n_elem).astype(np.float32)
    e_env = wp.array([0.55], dtype=float, device=device)
    adj_g = rng.uniform(-1.0, 1.0, n_elem).astype(np.float32)

    results = {}
    for backend in ("assembled", "march"):
        op = _make_operator(paths, n_elem, backend, f, device)
        adj_f = wp.zeros(n_elem, dtype=float, device=device)
        adj_e = wp.zeros(n_elem, dtype=float, device=device)
        adj_env = wp.zeros(1, dtype=float, device=device)
        op.vjp(
            wp.array(e_pow, dtype=float, device=device),
            e_env,
            wp.array(adj_g, dtype=float, device=device),
            adj_f,
            adj_e,
            adj_env,
        )
        results[backend] = adj_f.numpy().astype(np.float64)

    np.testing.assert_allclose(results["assembled"], results["march"], rtol=1.0e-6, atol=1.0e-9)


def test_vjp_directional_through_operator(test, device):
    """End-to-end gradient check through the operator, against the assembled forward.

    Uses the assembled backend for the perturbed evaluations, so this checks the marching
    adjoint against the strategy the Newton solve will actually run.
    """
    rng = np.random.default_rng(127)
    n_elem = 20
    paths, _ = _make_random_paths(n_elem, 160, rng, device)

    f = rng.uniform(0.1, 0.85, n_elem)
    e_pow = wp.array(rng.uniform(0.1, 4.0, n_elem).astype(np.float32), dtype=float, device=device)
    e_env = wp.array([0.42], dtype=float, device=device)
    adj_g = rng.uniform(-1.0, 1.0, n_elem)

    op = _make_operator(paths, n_elem, "assembled", f.astype(np.float32), device)
    adj_f = wp.zeros(n_elem, dtype=float, device=device)
    adj_e = wp.zeros(n_elem, dtype=float, device=device)
    adj_env = wp.zeros(1, dtype=float, device=device)
    op.vjp(e_pow, e_env, wp.array(adj_g.astype(np.float32), dtype=float, device=device), adj_f, adj_e, adj_env)
    ad = float(adj_f.numpy().astype(np.float64) @ (d := rng.normal(size=n_elem) / np.sqrt(n_elem)))

    def objective(field):
        probe = _make_operator(paths, n_elem, "assembled", field.astype(np.float32), device)
        out = wp.zeros(n_elem, dtype=float, device=device)
        probe.apply(e_pow, out, e_env)
        return float(adj_g @ out.numpy().astype(np.float64))

    # float32 assembly limits how small h can usefully be; 1e-3 balances truncation
    # against cancellation, and the check is loose enough to tolerate what remains.
    h = 1.0e-3
    fd = (objective(f + h * d) - objective(f - h * d)) / (2.0 * h)
    test.assertAlmostEqual(ad, fd, delta=2.0e-2 * max(1.0, abs(fd)))


def test_auto_backend_selection_memory_gate(test, device):
    """Auto must fall back to marching when the assembly does not fit."""
    rng = np.random.default_rng(131)
    n_elem = 16
    # 600 rays of up to 12 steps keeps n_elem^2 = 256 well under the ray-step count.
    paths, _ = _make_random_paths(n_elem, 600, rng, device)

    fits = thermal.VolumetricRadiationOperator(paths, n_elem, backend="auto", device=device)
    test.assertEqual(fits.backend, "assembled")
    test.assertEqual(fits.assembly_bytes, thermal.ExchangeMatrix.storage_bytes(n_elem))

    starved = thermal.VolumetricRadiationOperator(paths, n_elem, backend="auto", assembly_limit_bytes=16, device=device)
    test.assertEqual(starved.backend, "march")
    test.assertEqual(starved.assembly_bytes, 0)


def test_auto_backend_selection_work_gate(test, device):
    """Auto must march for sparse ray bundles even when the assembly would fit easily.

    A coarse angular discretization deduplicates poorly, so an O(n_elem^2) apply does more
    work than re-marching. Selecting on memory alone would get this backwards.
    """
    rng = np.random.default_rng(149)
    n_elem = 64
    # Few short rays: n_elem^2 = 4096 far exceeds the total ray-step count.
    paths, _ = _make_random_paths(n_elem, 40, rng, device, max_len=4)
    test.assertLess(paths.step_count, n_elem * n_elem)

    op = thermal.VolumetricRadiationOperator(paths, n_elem, backend="auto", device=device)
    test.assertEqual(op.backend, "march")
    test.assertGreater(op.work_ratio, 1.0)

    # Same domain, dense ray bundle: now assembly is the cheaper apply.
    dense_paths, _ = _make_random_paths(n_elem, 4000, rng, device, max_len=20)
    test.assertGreater(dense_paths.step_count, n_elem * n_elem)

    dense_op = thermal.VolumetricRadiationOperator(dense_paths, n_elem, backend="auto", device=device)
    test.assertEqual(dense_op.backend, "assembled")
    test.assertLess(dense_op.work_ratio, 1.0)


def test_rejects_unknown_backend(test, device):
    rng = np.random.default_rng(137)
    paths, _ = _make_random_paths(8, 20, rng, device)
    with test.assertRaises(ValueError):
        thermal.VolumetricRadiationOperator(paths, 8, backend="sparse", device=device)


def test_requires_update_before_use(test, device):
    """Applying without re-linearizing is a silent-wrong-answer bug, so it must raise."""
    rng = np.random.default_rng(139)
    n_elem = 8
    paths, _ = _make_random_paths(n_elem, 20, rng, device)
    op = thermal.VolumetricRadiationOperator(paths, n_elem, backend="march", device=device)

    out = wp.zeros(n_elem, dtype=float, device=device)
    with test.assertRaises(RuntimeError):
        op.apply(wp.zeros(n_elem, dtype=float, device=device), out, wp.zeros(1, dtype=float, device=device))


devices = get_test_devices()


class TestThermalOperator(unittest.TestCase):
    pass


add_function_test(
    TestThermalOperator, "test_assembly_matches_reference", test_assembly_matches_reference, devices=devices
)
add_function_test(TestThermalOperator, "test_backends_agree_on_apply", test_backends_agree_on_apply, devices=devices)
add_function_test(
    TestThermalOperator, "test_backends_agree_on_transpose", test_backends_agree_on_transpose, devices=devices
)
add_function_test(
    TestThermalOperator,
    "test_assembled_backend_adjoint_identity",
    test_assembled_backend_adjoint_identity,
    devices=devices,
)
add_function_test(
    TestThermalOperator,
    "test_vjp_independent_of_forward_backend",
    test_vjp_independent_of_forward_backend,
    devices=devices,
)
add_function_test(
    TestThermalOperator, "test_vjp_directional_through_operator", test_vjp_directional_through_operator, devices=devices
)
add_function_test(
    TestThermalOperator,
    "test_auto_backend_selection_memory_gate",
    test_auto_backend_selection_memory_gate,
    devices=devices,
)
add_function_test(
    TestThermalOperator, "test_auto_backend_selection_work_gate", test_auto_backend_selection_work_gate, devices=devices
)
add_function_test(TestThermalOperator, "test_rejects_unknown_backend", test_rejects_unknown_backend, devices=devices)
add_function_test(
    TestThermalOperator, "test_requires_update_before_use", test_requires_update_before_use, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
