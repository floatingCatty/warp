"""Confirm that Warp's AD mishandles the ray-transmissivity running product.

Models one ray marching through n cells with absorptivity f_k, accumulating
    tau_k = tau_{k-1} * (1 - f_k)
and depositing  tau_{k-1} * f_k * E_k  into an energy accumulator.

This is exactly the Noguchi Eq. (13) inner loop. Compare Warp AD against
central finite differences.
"""

import numpy as np

import warp as wp


@wp.kernel
def march_dynamic(f: wp.array[float], e: wp.array[float], out: wp.array[float], n: int):
    """Dynamic-trip-count loop: what a real DDA traversal compiles to."""
    tau = float(1.0)
    acc = float(0.0)
    for k in range(n):
        acc += tau * f[k] * e[k]
        tau = tau * (1.0 - f[k])
    wp.atomic_add(out, 0, acc)


@wp.kernel
def march_static(f: wp.array[float], e: wp.array[float], out: wp.array[float]):
    """Same body with a compile-time trip count, so Warp unrolls it."""
    tau = float(1.0)
    acc = float(0.0)
    for k in range(6):
        acc += tau * f[k] * e[k]
        tau = tau * (1.0 - f[k])
    wp.atomic_add(out, 0, acc)


N = 6


def forward_numpy(fv, ev):
    tau, acc = 1.0, 0.0
    for k in range(N):
        acc += tau * fv[k] * ev[k]
        tau *= 1.0 - fv[k]
    return acc


def warp_grad(kernel, fv, ev, dynamic):
    f = wp.array(fv, dtype=float, requires_grad=True)
    e = wp.array(ev, dtype=float, requires_grad=True)
    out = wp.zeros(1, dtype=float, requires_grad=True)
    with wp.Tape() as tape:
        inputs = [f, e, out, N] if dynamic else [f, e, out]
        wp.launch(kernel, dim=1, inputs=inputs)
    tape.backward(loss=out)
    return f.grad.numpy().copy()


def fd_grad(fv, ev, h=1e-6):
    g = np.zeros(N)
    for k in range(N):
        p, m = fv.copy(), fv.copy()
        p[k] += h
        m[k] -= h
        g[k] = (forward_numpy(p, ev) - forward_numpy(m, ev)) / (2.0 * h)
    return g


def main():
    rng = np.random.default_rng(0)
    fv = rng.uniform(0.15, 0.55, N).astype(np.float32)
    ev = rng.uniform(0.5, 2.0, N).astype(np.float32)

    ref = fd_grad(fv.astype(np.float64), ev.astype(np.float64))
    g_dyn = warp_grad(march_dynamic, fv, ev, dynamic=True)
    g_sta = warp_grad(march_static, fv, ev, dynamic=False)

    np.set_printoptions(precision=5, suppress=True)
    print("finite difference (reference) :", ref)
    print("warp AD, dynamic loop         :", g_dyn)
    print("warp AD, static/unrolled loop :", g_sta)
    print()
    print(f"max |dynamic - FD| = {np.abs(g_dyn - ref).max():.4e}")
    print(f"max |static  - FD| = {np.abs(g_sta - ref).max():.4e}")


if __name__ == "__main__":
    wp.init()
    main()
