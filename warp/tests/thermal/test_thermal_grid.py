# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification of the implicit 2D grid ray bundle against explicitly stored ray paths.

The bundle generates rays and walks the grid inline, so nothing about its traversal is
inspectable at runtime. The reference below independently reproduces the same ray
generation and DDA in NumPy, materializes the result as a
:class:`~warp.thermal.RayPaths`, and drives the already-verified path-based operators with
it. Any disagreement is then a defect in the inline implementation rather than in the
physics, which the transport tests cover separately.
"""

import unittest

import numpy as np

import warp as wp
import warp.thermal as thermal
from warp.tests.unittest_utils import *


def _reference_bundle(res, n_pt, n_ang, device):
    """Reproduce GridRayBundle2D's rays and traversals independently, in float64."""
    nx, ny = res
    hx, hy = 1.0 / nx, 1.0 / ny
    perimeter = 2.0 * (hx + hy)
    quad = np.pi / (2.0 * n_ang)

    source, weight, exit_kind, paths = [], [], [], []

    for e in range(nx * ny):
        i, j = e // ny, e % ny
        for a in range(4):
            if a == 0:
                nrm, face_len, start = (-1.0, 0.0), hy, (i - 1, j)
            elif a == 1:
                nrm, face_len, start = (1.0, 0.0), hy, (i + 1, j)
            elif a == 2:
                nrm, face_len, start = (0.0, -1.0), hx, (i, j - 1)
            else:
                nrm, face_len, start = (0.0, 1.0), hx, (i, j + 1)

            for b in range(n_pt):
                s = (b + 0.5) / n_pt
                if a == 0:
                    p = (i * hx, j * hy + s * hy)
                elif a == 1:
                    p = ((i + 1) * hx, j * hy + s * hy)
                elif a == 2:
                    p = (i * hx + s * hx, j * hy)
                else:
                    p = (i * hx + s * hx, (j + 1) * hy)

                for c in range(n_ang):
                    theta = -0.5 * np.pi + np.pi * (c + 0.5) / n_ang
                    ct, st = np.cos(theta), np.sin(theta)
                    d = (nrm[0] * ct - nrm[1] * st, nrm[0] * st + nrm[1] * ct)

                    source.append(e)
                    weight.append(quad * face_len * ct / (n_pt * perimeter))
                    exit_kind.append(thermal.EXIT_ENVIRONMENT if d[1] > 0.0 else thermal.EXIT_ADIABATIC)
                    paths.append(_reference_walk(res, start, p, d))

    rays = thermal.RayPaths.from_numpy(source, weight, exit_kind, paths, device=device)
    return rays, {
        "source": np.array(source),
        "weight": np.array(weight),
        "exit_kind": np.array(exit_kind),
        "paths": paths,
    }


def _reference_walk(res, start, origin, direction):
    """Amanatides-Woo walk from ``start``, matching the kernel's tie-breaking exactly."""
    nx, ny = res
    hx, hy = 1.0 / nx, 1.0 / ny
    i, j = start
    if not (0 <= i < nx and 0 <= j < ny):
        return []

    dx, dy = direction
    inf = 1.0e30

    if dx > 0.0:
        t_max_x, t_del_x, step_i = ((i + 1) * hx - origin[0]) / dx, hx / dx, 1
    elif dx < 0.0:
        t_max_x, t_del_x, step_i = (i * hx - origin[0]) / dx, -hx / dx, -1
    else:
        t_max_x, t_del_x, step_i = inf, inf, 0

    if dy > 0.0:
        t_max_y, t_del_y, step_j = ((j + 1) * hy - origin[1]) / dy, hy / dy, 1
    elif dy < 0.0:
        t_max_y, t_del_y, step_j = (j * hy - origin[1]) / dy, -hy / dy, -1
    else:
        t_max_y, t_del_y, step_j = inf, inf, 0

    visited = []
    while 0 <= i < nx and 0 <= j < ny:
        visited.append(i * ny + j)
        if t_max_x < t_max_y:
            t_max_x += t_del_x
            i += step_i
        else:
            t_max_y += t_del_y
            j += step_j
    return visited


_CONFIGS = [((5, 4), 2, 6), ((6, 6), 1, 10)]


def _fields(n_elem, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(0.05, 0.9, n_elem).astype(np.float32),
        rng.uniform(0.1, 4.0, n_elem).astype(np.float32),
        rng.uniform(-1.0, 1.0, n_elem).astype(np.float32),
    )


def test_ray_count_and_steps_match_reference(test, device):
    """Ray count and total traversal steps must match the reference enumeration."""
    for res, n_pt, n_ang in _CONFIGS:
        bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, device=device)
        _, spec = _reference_bundle(res, n_pt, n_ang, device)

        test.assertEqual(bundle.ray_count, len(spec["source"]))
        test.assertEqual(bundle.step_count, sum(len(p) for p in spec["paths"]))
        test.assertLessEqual(max(len(p) for p in spec["paths"]), bundle.max_steps)


def test_quadrature_weights_conserve_energy(test, device):
    """Per-element ray weights must sum to one, up to angular discretization error.

    This is what makes the exchange factors conserve energy: every element emits exactly
    its own emissive power, no more.
    """
    for res, n_pt, n_ang in _CONFIGS:
        _, spec = _reference_bundle(res, n_pt, n_ang, device)
        totals = np.zeros(res[0] * res[1])
        np.add.at(totals, spec["source"], spec["weight"])

        # Midpoint rule on cos over the half-plane converges as O(1/n_ang^2).
        tol = 4.0 / (n_ang * n_ang)
        np.testing.assert_allclose(totals, 1.0, atol=tol)


def test_transport_matches_reference(test, device):
    """Inline traversal must reproduce the path-based forward transport."""
    for res, n_pt, n_ang in _CONFIGS:
        n_elem = res[0] * res[1]
        bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, device=device)
        ref_rays, _ = _reference_bundle(res, n_pt, n_ang, device)

        f, e_pow, _ = _fields(n_elem, 11)
        f_wp = wp.array(f, dtype=float, device=device)
        e_wp = wp.array(e_pow, dtype=float, device=device)
        env = wp.array([0.83], dtype=float, device=device)

        got = wp.zeros(n_elem, dtype=float, device=device)
        bundle.transport(f_wp, e_wp, got, env)

        want = wp.zeros(n_elem, dtype=float, device=device)
        ref_rays.transport(f_wp, e_wp, want, env)

        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=5.0e-5, atol=1.0e-6)


def test_assembly_matches_reference(test, device):
    """Inline assembly must reproduce the path-based exchange factors."""
    for res, n_pt, n_ang in _CONFIGS:
        n_elem = res[0] * res[1]
        bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, device=device)
        ref_rays, _ = _reference_bundle(res, n_pt, n_ang, device)

        f, _, _ = _fields(n_elem, 13)
        f_wp = wp.array(f, dtype=float, device=device)

        got = thermal.ExchangeMatrix(n_elem, device=device)
        bundle.assemble(got, f_wp)

        want = thermal.ExchangeMatrix(n_elem, device=device)
        ref_rays.assemble(want, f_wp)

        np.testing.assert_allclose(got.factors.numpy(), want.factors.numpy(), rtol=5.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(
            got.environment_factors.numpy(), want.environment_factors.numpy(), rtol=5.0e-5, atol=1.0e-6
        )


def test_transpose_matches_reference(test, device):
    for res, n_pt, n_ang in _CONFIGS:
        n_elem = res[0] * res[1]
        bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, device=device)
        ref_rays, _ = _reference_bundle(res, n_pt, n_ang, device)

        f, _, v = _fields(n_elem, 17)
        f_wp = wp.array(f, dtype=float, device=device)
        v_wp = wp.array(v, dtype=float, device=device)

        got, got_env = wp.zeros(n_elem, dtype=float, device=device), wp.zeros(1, dtype=float, device=device)
        bundle.transport_transpose(f_wp, v_wp, got, got_env)

        want, want_env = wp.zeros(n_elem, dtype=float, device=device), wp.zeros(1, dtype=float, device=device)
        ref_rays.transport_transpose(f_wp, v_wp, want, want_env)

        np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=5.0e-5, atol=1.0e-6)
        test.assertAlmostEqual(float(got_env.numpy()[0]), float(want_env.numpy()[0]), delta=1.0e-5)


def test_vjp_matches_reference(test, device):
    """The chunked inline adjoint must reproduce the verified path-based adjoint."""
    for res, n_pt, n_ang in _CONFIGS:
        n_elem = res[0] * res[1]
        bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, device=device)
        ref_rays, _ = _reference_bundle(res, n_pt, n_ang, device)

        f, e_pow, adj_g = _fields(n_elem, 19)
        f_wp = wp.array(f, dtype=float, device=device)
        e_wp = wp.array(e_pow, dtype=float, device=device)
        env = wp.array([0.61], dtype=float, device=device)
        adj_wp = wp.array(adj_g, dtype=float, device=device)

        out = []
        for rays in (bundle, ref_rays):
            adj_f = wp.zeros(n_elem, dtype=float, device=device)
            adj_e = wp.zeros(n_elem, dtype=float, device=device)
            adj_env = wp.zeros(1, dtype=float, device=device)
            rays.transport_vjp(f_wp, e_wp, env, adj_wp, adj_f, adj_e, adj_env)
            out.append((adj_f.numpy(), adj_e.numpy(), float(adj_env.numpy()[0])))

        np.testing.assert_allclose(out[0][0], out[1][0], rtol=5.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(out[0][1], out[1][1], rtol=5.0e-5, atol=1.0e-6)
        test.assertAlmostEqual(out[0][2], out[1][2], delta=1.0e-5)


def test_vjp_invariant_to_chunking(test, device):
    """Chunk size bounds scratch memory only; it must not change the result."""
    res, n_pt, n_ang = (6, 6), 2, 8
    n_elem = res[0] * res[1]
    f, e_pow, adj_g = _fields(n_elem, 23)
    f_wp = wp.array(f, dtype=float, device=device)
    e_wp = wp.array(e_pow, dtype=float, device=device)
    env = wp.array([0.5], dtype=float, device=device)
    adj_wp = wp.array(adj_g, dtype=float, device=device)

    results = []
    for chunk in (1 << 20, 997, 64):
        bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, chunk_rays=chunk, device=device)
        adj_f = wp.zeros(n_elem, dtype=float, device=device)
        adj_e = wp.zeros(n_elem, dtype=float, device=device)
        adj_env = wp.zeros(1, dtype=float, device=device)
        bundle.transport_vjp(f_wp, e_wp, env, adj_wp, adj_f, adj_e, adj_env)
        results.append(adj_f.numpy())

    # A chunk smaller than the bundle exercises the multi-launch path.
    test.assertLess(64, thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang).ray_count)
    np.testing.assert_allclose(results[0], results[1], rtol=1.0e-5, atol=1.0e-7)
    np.testing.assert_allclose(results[0], results[2], rtol=1.0e-5, atol=1.0e-7)


def test_vjp_directional_finite_difference(test, device):
    """Directional gradient check driven entirely through the bundle."""
    res, n_pt, n_ang = (6, 6), 2, 8
    n_elem = res[0] * res[1]
    bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, device=device)

    rng = np.random.default_rng(29)
    f = rng.uniform(0.15, 0.8, n_elem)
    e_pow = wp.array(rng.uniform(0.1, 4.0, n_elem).astype(np.float32), dtype=float, device=device)
    env = wp.array([0.37], dtype=float, device=device)
    adj_g = rng.uniform(-1.0, 1.0, n_elem)
    adj_wp = wp.array(adj_g.astype(np.float32), dtype=float, device=device)

    adj_f = wp.zeros(n_elem, dtype=float, device=device)
    adj_e = wp.zeros(n_elem, dtype=float, device=device)
    adj_env = wp.zeros(1, dtype=float, device=device)
    bundle.transport_vjp(
        wp.array(f.astype(np.float32), dtype=float, device=device), e_pow, env, adj_wp, adj_f, adj_e, adj_env
    )

    d = rng.normal(size=n_elem)
    d /= np.linalg.norm(d)
    ad = float(adj_f.numpy().astype(np.float64) @ d)

    def objective(field):
        out = wp.zeros(n_elem, dtype=float, device=device)
        bundle.transport(wp.array(field.astype(np.float32), dtype=float, device=device), e_pow, out, env)
        return float(adj_g @ out.numpy().astype(np.float64))

    h = 1.0e-3
    fd = (objective(f + h * d) - objective(f - h * d)) / (2.0 * h)
    test.assertAlmostEqual(ad, fd, delta=2.0e-2 * max(1.0, abs(fd)))


def test_operator_accepts_bundle(test, device):
    """Both ray sources must drive the operator identically."""
    res, n_pt, n_ang = (5, 4), 2, 6
    n_elem = res[0] * res[1]
    bundle = thermal.GridRayBundle2D(res, launch_points=n_pt, angles=n_ang, device=device)
    ref_rays, _ = _reference_bundle(res, n_pt, n_ang, device)

    f, e_pow, _ = _fields(n_elem, 31)
    f_wp = wp.array(f, dtype=float, device=device)
    e_wp = wp.array(e_pow, dtype=float, device=device)
    env = wp.array([0.44], dtype=float, device=device)

    got = []
    for rays in (bundle, ref_rays):
        op = thermal.VolumetricRadiationOperator(rays, n_elem, device=device)
        op.update(f_wp)
        out = wp.zeros(n_elem, dtype=float, device=device)
        op.apply(e_wp, out, env)
        got.append(out.numpy())

    np.testing.assert_allclose(got[0], got[1], rtol=5.0e-5, atol=1.0e-6)


def test_memory_independent_of_ray_count(test, device):
    """Scratch must scale with the chunk, not with the bundle.

    This is the property that makes a realistic angular discretization affordable: the
    same domain at 64x the ray count must not cost 64x the memory.
    """
    res = (16, 16)
    small = thermal.GridRayBundle2D(res, launch_points=1, angles=8, chunk_rays=4096, device=device)
    large = thermal.GridRayBundle2D(res, launch_points=8, angles=64, chunk_rays=4096, device=device)

    test.assertEqual(large.ray_count, 64 * small.ray_count)
    test.assertEqual(large.scratch_bytes(), small.scratch_bytes())


devices = get_test_devices()


class TestThermalGrid(unittest.TestCase):
    pass


add_function_test(
    TestThermalGrid,
    "test_ray_count_and_steps_match_reference",
    test_ray_count_and_steps_match_reference,
    devices=devices,
)
add_function_test(
    TestThermalGrid, "test_quadrature_weights_conserve_energy", test_quadrature_weights_conserve_energy, devices=devices
)
add_function_test(
    TestThermalGrid, "test_transport_matches_reference", test_transport_matches_reference, devices=devices
)
add_function_test(TestThermalGrid, "test_assembly_matches_reference", test_assembly_matches_reference, devices=devices)
add_function_test(
    TestThermalGrid, "test_transpose_matches_reference", test_transpose_matches_reference, devices=devices
)
add_function_test(TestThermalGrid, "test_vjp_matches_reference", test_vjp_matches_reference, devices=devices)
add_function_test(TestThermalGrid, "test_vjp_invariant_to_chunking", test_vjp_invariant_to_chunking, devices=devices)
add_function_test(
    TestThermalGrid, "test_vjp_directional_finite_difference", test_vjp_directional_finite_difference, devices=devices
)
add_function_test(TestThermalGrid, "test_operator_accepts_bundle", test_operator_accepts_bundle, devices=devices)
add_function_test(
    TestThermalGrid, "test_memory_independent_of_ray_count", test_memory_independent_of_ray_count, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
