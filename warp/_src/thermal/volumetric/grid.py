# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Implicit ray bundles over 2D structured grids.

:class:`~warp._src.thermal.volumetric.paths.RayPaths` stores the element sequence of every
ray, which is fine for small problems but does not survive a realistic angular
discretization: a 60x60 domain with four faces, ten launch points and a hundred angles per
element emits 1.4e7 rays covering roughly 6e8 traversal steps, several gigabytes of indices
that every consumer reads exactly once per design iteration.

This module keeps the bundle implicit instead. A ray is defined by its index alone, so
:class:`GridRayBundle2D` derives launch point, direction and weight from index arithmetic
and walks the grid with a DDA inline, storing nothing. Memory becomes independent of both
the ray count and the domain size.

Only the adjoint needs per-step state, because the derivative of a transmittance product
is a leave-one-out product and cannot be formed in a single pass. That state is confined
to a bounded chunk of rays at a time rather than the whole bundle, and a 2D grid walk
visits at most ``res[0] + res[1] + 1`` cells, which makes the chunk scratch exactly sized.

The angular quadrature follows the zonal view-factor discretization: rays leave each face
over the half-plane above its normal, weighted by :math:`C A_a \\cos\\theta_c / (N_{pt} A_e)`
with :math:`C = \\pi / (2 N_{ang})`. Those weights sum to one over each element, which is
what makes the exchange factors conserve energy.
"""

from __future__ import annotations

import warp as wp
from warp._src.thermal.volumetric.paths import EXIT_ADIABATIC, EXIT_ENVIRONMENT

__all__ = ["GridRayBundle2D"]

_DEFAULT_CHUNK_RAYS = 1 << 18


@wp.struct
class _GridRay:
    """A single ray of the bundle, reconstructed from its index."""

    source: int
    i: int
    j: int
    px: float
    py: float
    dx: float
    dy: float
    weight: float
    exit_kind: int
    inside: int


@wp.func
def _make_ray(r: int, nx: int, ny: int, n_pt: int, n_ang: int) -> _GridRay:
    """Reconstruct ray ``r`` from its index.

    Rays are ordered element-major, then face, launch point and angle, so consecutive
    threads walk neighbouring directions from the same face and stay close in memory.
    """
    c = r % n_ang
    rest = r // n_ang
    b = rest % n_pt
    rest = rest // n_pt
    a = rest % 4
    e = rest // 4

    i = e // ny
    j = e % ny

    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)

    # Face a of cell (i, j): outward normal, a point parameterized along the face, the
    # face's length, and the neighbouring cell the ray enters.
    ox = float(i) * hx
    oy = float(j) * hy
    s = (float(b) + 0.5) / float(n_pt)

    nrm_x = 0.0
    nrm_y = 0.0
    px = 0.0
    py = 0.0
    face_len = 0.0
    ni = i
    nj = j

    if a == 0:  # -x
        nrm_x = -1.0
        px = ox
        py = oy + s * hy
        face_len = hy
        ni = i - 1
    elif a == 1:  # +x
        nrm_x = 1.0
        px = ox + hx
        py = oy + s * hy
        face_len = hy
        ni = i + 1
    elif a == 2:  # -y
        nrm_y = -1.0
        px = ox + s * hx
        py = oy
        face_len = hx
        nj = j - 1
    else:  # +y
        nrm_y = 1.0
        px = ox + s * hx
        py = oy + hy
        face_len = hx
        nj = j + 1

    # Uniform in-plane angle over the half-plane above the normal.
    theta = -0.5 * wp.pi + wp.pi * (float(c) + 0.5) / float(n_ang)
    cos_t = wp.cos(theta)
    sin_t = wp.sin(theta)

    dx = nrm_x * cos_t - nrm_y * sin_t
    dy = nrm_x * sin_t + nrm_y * cos_t

    perimeter = 2.0 * (hx + hy)
    quad = wp.pi / (2.0 * float(n_ang))
    weight = quad * face_len * cos_t / (float(n_pt) * perimeter)

    ray = _GridRay()
    ray.source = e
    ray.i = ni
    ray.j = nj
    ray.px = px
    ray.py = py
    ray.dx = dx
    ray.dy = dy
    ray.weight = weight
    # Deep space above the horizon, adiabatic below, matching the space environment used
    # for radiative heat sink design.
    ray.exit_kind = wp.where(dy > 0.0, EXIT_ENVIRONMENT, EXIT_ADIABATIC)
    ray.inside = wp.where(ni >= 0 and ni < nx and nj >= 0 and nj < ny, 1, 0)
    return ray


@wp.func
def _dda_init(ray: _GridRay, nx: int, ny: int):
    """Parametric distances to the next grid line and between grid lines, per axis."""
    hx = 1.0 / float(nx)
    hy = 1.0 / float(ny)

    inf = 1.0e30

    if ray.dx > 0.0:
        t_max_x = (float(ray.i + 1) * hx - ray.px) / ray.dx
        t_del_x = hx / ray.dx
        step_i = 1
    elif ray.dx < 0.0:
        t_max_x = (float(ray.i) * hx - ray.px) / ray.dx
        t_del_x = -hx / ray.dx
        step_i = -1
    else:
        t_max_x = inf
        t_del_x = inf
        step_i = 0

    if ray.dy > 0.0:
        t_max_y = (float(ray.j + 1) * hy - ray.py) / ray.dy
        t_del_y = hy / ray.dy
        step_j = 1
    elif ray.dy < 0.0:
        t_max_y = (float(ray.j) * hy - ray.py) / ray.dy
        t_del_y = -hy / ray.dy
        step_j = -1
    else:
        t_max_y = inf
        t_del_y = inf
        step_j = 0

    return t_max_x, t_max_y, t_del_x, t_del_y, step_i, step_j


@wp.kernel(enable_backward=False)
def _count_steps(nx: int, ny: int, n_pt: int, n_ang: int, counts: wp.array[int]):
    r = wp.tid()
    ray = _make_ray(r, nx, ny, n_pt, n_ang)

    n = int(0)
    if ray.inside == 1:
        t_max_x, t_max_y, t_del_x, t_del_y, step_i, step_j = _dda_init(ray, nx, ny)
        i = ray.i
        j = ray.j
        while i >= 0 and i < nx and j >= 0 and j < ny:
            n += 1
            if t_max_x < t_max_y:
                t_max_x += t_del_x
                i += step_i
            else:
                t_max_y += t_del_y
                j += step_j

    counts[r] = n


@wp.kernel(enable_backward=False)
def _assemble(
    nx: int,
    ny: int,
    n_pt: int,
    n_ang: int,
    absorptivity: wp.array[float],
    factors: wp.array2d[float],
    environment_factors: wp.array[float],
):
    """Accumulate exchange factors, walking each ray once and storing nothing."""
    r = wp.tid()
    ray = _make_ray(r, nx, ny, n_pt, n_ang)
    e = ray.source

    tau = float(1.0)
    if ray.inside == 1:
        t_max_x, t_max_y, t_del_x, t_del_y, step_i, step_j = _dda_init(ray, nx, ny)
        i = ray.i
        j = ray.j
        while i >= 0 and i < nx and j >= 0 and j < ny:
            c = i * ny + j
            f = absorptivity[c]
            wp.atomic_add(factors, e, c, ray.weight * tau * f)
            tau = tau * (1.0 - f)

            if t_max_x < t_max_y:
                t_max_x += t_del_x
                i += step_i
            else:
                t_max_y += t_del_y
                j += step_j

    if ray.exit_kind == EXIT_ADIABATIC:
        wp.atomic_add(factors, e, e, ray.weight * tau)
    else:
        wp.atomic_add(environment_factors, e, ray.weight * tau)


@wp.kernel(enable_backward=False)
def _transport(
    nx: int,
    ny: int,
    n_pt: int,
    n_ang: int,
    absorptivity: wp.array[float],
    emissive: wp.array[float],
    environment_emissive: wp.array[float],
    irradiation: wp.array[float],
):
    """Forward transport in explicit attenuated-sum form, which needs only one pass.

    The compositing recurrence used by the path-based operator would require walking the
    ray backwards, which an inline DDA cannot do without storing it.
    """
    r = wp.tid()
    ray = _make_ray(r, nx, ny, n_pt, n_ang)

    tau = float(1.0)
    acc = float(0.0)
    if ray.inside == 1:
        t_max_x, t_max_y, t_del_x, t_del_y, step_i, step_j = _dda_init(ray, nx, ny)
        i = ray.i
        j = ray.j
        while i >= 0 and i < nx and j >= 0 and j < ny:
            c = i * ny + j
            f = absorptivity[c]
            acc += tau * f * emissive[c]
            tau = tau * (1.0 - f)

            if t_max_x < t_max_y:
                t_max_x += t_del_x
                i += step_i
            else:
                t_max_y += t_del_y
                j += step_j

    if ray.exit_kind == EXIT_ADIABATIC:
        acc += tau * emissive[ray.source]
    else:
        acc += tau * environment_emissive[0]

    wp.atomic_add(irradiation, ray.source, ray.weight * acc)


@wp.kernel(enable_backward=False)
def _transport_transpose(
    nx: int,
    ny: int,
    n_pt: int,
    n_ang: int,
    absorptivity: wp.array[float],
    v: wp.array[float],
    out: wp.array[float],
    out_environment: wp.array[float],
):
    r = wp.tid()
    ray = _make_ray(r, nx, ny, n_pt, n_ang)
    s = v[ray.source] * ray.weight

    tau = float(1.0)
    if ray.inside == 1:
        t_max_x, t_max_y, t_del_x, t_del_y, step_i, step_j = _dda_init(ray, nx, ny)
        i = ray.i
        j = ray.j
        while i >= 0 and i < nx and j >= 0 and j < ny:
            c = i * ny + j
            f = absorptivity[c]
            wp.atomic_add(out, c, s * tau * f)
            tau = tau * (1.0 - f)

            if t_max_x < t_max_y:
                t_max_x += t_del_x
                i += step_i
            else:
                t_max_y += t_del_y
                j += step_j

    if ray.exit_kind == EXIT_ADIABATIC:
        wp.atomic_add(out, ray.source, s * tau)
    else:
        wp.atomic_add(out_environment, 0, s * tau)


@wp.kernel(enable_backward=False)
def _transport_vjp(
    nx: int,
    ny: int,
    n_pt: int,
    n_ang: int,
    ray_begin: int,
    max_steps: int,
    absorptivity: wp.array[float],
    emissive: wp.array[float],
    environment_emissive: wp.array[float],
    adj_irradiation: wp.array[float],
    scratch_cells: wp.array[int],
    scratch_tau: wp.array[float],
    adj_absorptivity: wp.array[float],
    adj_emissive: wp.array[float],
    adj_environment_emissive: wp.array[float],
):
    """Adjoint of :func:`_transport`, buffering one chunk of rays at a time.

    The derivative with respect to a cell's absorptivity is a leave-one-out product over
    the ray, so the walk cannot be collapsed into a single pass. The forward walk records
    the visited cells and the prefix transmittance into a per-chunk buffer; the reverse
    walk replays that buffer to build ``A`` and emit both adjoints, never dividing by a
    transmittance.
    """
    slot = wp.tid()
    r = ray_begin + slot
    ray = _make_ray(r, nx, ny, n_pt, n_ang)
    e = ray.source
    gbar = adj_irradiation[e] * ray.weight
    base = slot * max_steps

    n = int(0)
    tau = float(1.0)
    if ray.inside == 1:
        t_max_x, t_max_y, t_del_x, t_del_y, step_i, step_j = _dda_init(ray, nx, ny)
        i = ray.i
        j = ray.j
        while i >= 0 and i < nx and j >= 0 and j < ny:
            scratch_cells[base + n] = i * ny + j
            scratch_tau[base + n] = tau
            tau = tau * (1.0 - absorptivity[i * ny + j])
            n += 1

            if t_max_x < t_max_y:
                t_max_x += t_del_x
                i += step_i
            else:
                t_max_y += t_del_y
                j += step_j

    if ray.exit_kind == EXIT_ADIABATIC:
        acc = emissive[e]
    else:
        acc = environment_emissive[0]

    # `acc` enters iteration k holding A_{k+1}.
    for k in range(n - 1, -1, -1):
        c = scratch_cells[base + k]
        f = absorptivity[c]
        t = scratch_tau[base + k]

        wp.atomic_add(adj_absorptivity, c, gbar * t * (emissive[c] - acc))
        wp.atomic_add(adj_emissive, c, gbar * t * f)

        acc = f * emissive[c] + (1.0 - f) * acc

    if ray.exit_kind == EXIT_ADIABATIC:
        wp.atomic_add(adj_emissive, e, gbar * tau)
    else:
        wp.atomic_add(adj_environment_emissive, 0, gbar * tau)


class GridRayBundle2D:
    """Implicit ray bundle over a 2D structured grid on the unit square.

    Elements are numbered row-major as ``i * res[1] + j``. Rays leave every face of every
    element, so the bundle holds ``4 * res[0] * res[1] * launch_points * angles`` rays
    while allocating nothing proportional to that count.

    Args:
        res: Number of cells along each axis.
        launch_points: Launch points per element face, :math:`N_{pt}`.
        angles: Ray directions per launch point, :math:`N_{ang}`.
        chunk_rays: Rays processed per adjoint launch. Bounds the scratch buffer, which
            holds ``chunk_rays * (res[0] + res[1] + 1)`` cells and transmittances.
        device: Device to allocate on.

    Attributes:
        max_steps: Upper bound on cells visited by one ray, exact for a 2D grid walk.
    """

    def __init__(
        self,
        res: tuple[int, int],
        launch_points: int = 1,
        angles: int = 16,
        chunk_rays: int = _DEFAULT_CHUNK_RAYS,
        device=None,
    ):
        self.res = res
        self.launch_points = launch_points
        self.angles = angles
        self.device = device
        self.element_count = res[0] * res[1]
        self.max_steps = res[0] + res[1] + 1
        self.chunk_rays = min(chunk_rays, self.ray_count)

        self._scratch_cells = None
        self._scratch_tau = None
        self._step_count = None

    @property
    def ray_count(self) -> int:
        """Number of rays in the bundle."""
        return 4 * self.element_count * self.launch_points * self.angles

    @property
    def step_count(self) -> int:
        """Total traversal steps across all rays.

        Counted once with a walk that reads no material data, then cached. Used to choose
        between assembled and marching transport, since it is the marching cost.
        """
        if self._step_count is None:
            counts = wp.zeros(self.ray_count, dtype=int, device=self.device)
            wp.launch(
                _count_steps,
                dim=self.ray_count,
                inputs=[self.res[0], self.res[1], self.launch_points, self.angles],
                outputs=[counts],
                device=self.device,
            )
            self._step_count = int(counts.numpy().sum())
        return self._step_count

    def _args(self):
        return [self.res[0], self.res[1], self.launch_points, self.angles]

    def assemble(self, exchange, absorptivity: wp.array):
        """Accumulate exchange factors into ``exchange``, walking every ray once."""
        exchange.factors.zero_()
        exchange.environment_factors.zero_()
        wp.launch(
            _assemble,
            dim=self.ray_count,
            inputs=[*self._args(), absorptivity],
            outputs=[exchange.factors, exchange.environment_factors],
            device=self.device,
        )

    def transport(
        self,
        absorptivity: wp.array,
        emissive: wp.array,
        irradiation: wp.array,
        environment_emissive: wp.array,
    ):
        """Accumulate incident radiative power on every element."""
        irradiation.zero_()
        wp.launch(
            _transport,
            dim=self.ray_count,
            inputs=[*self._args(), absorptivity, emissive, environment_emissive],
            outputs=[irradiation],
            device=self.device,
        )

    def transport_transpose(
        self,
        absorptivity: wp.array,
        v: wp.array,
        out: wp.array,
        out_environment: wp.array,
    ):
        """Apply the transpose of the emissive-power tangent to ``v``."""
        out.zero_()
        out_environment.zero_()
        wp.launch(
            _transport_transpose,
            dim=self.ray_count,
            inputs=[*self._args(), absorptivity, v],
            outputs=[out, out_environment],
            device=self.device,
        )

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
        """Accumulate the vector-Jacobian product of :meth:`transport`.

        Adjoint outputs are accumulated, not overwritten.
        """
        if self._scratch_cells is None:
            size = self.chunk_rays * self.max_steps
            self._scratch_cells = wp.zeros(size, dtype=int, device=self.device)
            self._scratch_tau = wp.zeros(size, dtype=float, device=self.device)

        for begin in range(0, self.ray_count, self.chunk_rays):
            count = min(self.chunk_rays, self.ray_count - begin)
            wp.launch(
                _transport_vjp,
                dim=count,
                inputs=[
                    *self._args(),
                    begin,
                    self.max_steps,
                    absorptivity,
                    emissive,
                    environment_emissive,
                    adj_irradiation,
                    self._scratch_cells,
                    self._scratch_tau,
                ],
                outputs=[adj_absorptivity, adj_emissive, adj_environment_emissive],
                device=self.device,
            )

    def scratch_bytes(self) -> int:
        """Bytes the adjoint scratch buffer occupies once allocated."""
        return 8 * self.chunk_rays * self.max_steps
