# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coupled conduction-radiation residual on a 2D structured grid.

Radiation enters the heat conduction equation as a volumetric source rather than a boundary
flux, which is what lets it act on structural boundaries that exist only implicitly in a
density field. Each element is treated as an isothermal zone emitting from its representative
temperature:

.. math::
    R(T, \\rho) = K_{\\mathrm{cond}}(\\rho)\\,T + F_{\\mathrm{rad}}(T, \\rho) - F_{\\mathrm{in}} = 0,

.. math::
    Q_{\\mathrm{rad},e} = N_R \\frac{A_e f(\\rho_e)}{V_e}
    \\left( \\tilde T_e^4 - \\sum_E F_{eE} \\tilde T_E^4 - F_{e,\\mathrm{env}} T_{\\mathrm{env}}^4 \\right).

The conduction-radiation number :math:`N_R = \\sigma T_{\\mathrm{ref}}^3 L / k_{\\max}` is the
only physical parameter left after nondimensionalization; it sets how strongly radiation
competes with conduction.

The tangent is where the two mechanisms genuinely mix. Emission contributes
:math:`4\\tilde T^3` on the diagonal while absorption couples every element to every element it
can see, so :math:`\\partial R/\\partial T` is **nonsymmetric and nonlocal**. It is never
assembled: transport supplies its action, and separately its transpose, which is what the
adjoint solve consumes.
"""

from __future__ import annotations

import warp as wp
from warp._src.thermal.volumetric.conduction import ConductionOperator2D, DirichletMask

__all__ = ["CoupledResidual2D"]


@wp.kernel(enable_backward=False)
def _emissive_power(temperature: wp.array[float], out: wp.array[float]):
    """Stefan-Boltzmann emissive power of each isothermal zone."""
    e = wp.tid()
    t = temperature[e]
    out[e] = t * t * t * t


@wp.kernel(enable_backward=False)
def _emissive_tangent(temperature: wp.array[float], v: wp.array[float], out: wp.array[float]):
    """Directional derivative of the emissive power, ``4 T^3 v``."""
    e = wp.tid()
    t = temperature[e]
    out[e] = 4.0 * t * t * t * v[e]


@wp.kernel(enable_backward=False)
def _net_radiation(
    emissive: wp.array[float],
    irradiation: wp.array[float],
    coefficient: wp.array[float],
    out: wp.array[float],
):
    """Net volumetric radiative loss, emitted minus absorbed."""
    e = wp.tid()
    out[e] = coefficient[e] * (emissive[e] - irradiation[e])


@wp.kernel(enable_backward=False)
def _radiation_coefficient(
    absorptivity: wp.array[float],
    area_over_volume: float,
    conduction_radiation_number: float,
    out: wp.array[float],
):
    e = wp.tid()
    out[e] = conduction_radiation_number * area_over_volume * absorptivity[e]


@wp.kernel(enable_backward=False)
def _scale(x: wp.array[float], factor: wp.array[float], out: wp.array[float]):
    e = wp.tid()
    out[e] = x[e] * factor[e]


@wp.kernel(enable_backward=False)
def _tangent_element_term(
    coefficient: wp.array[float],
    local: wp.array[float],
    absorbed: wp.array[float],
    out: wp.array[float],
):
    """``c_e (4 T_e^3 v_e - sum_E F_eE 4 T_E^3 v_E)``, the element-wise radiative tangent."""
    e = wp.tid()
    out[e] = coefficient[e] * (local[e] - absorbed[e])


@wp.kernel(enable_backward=False)
def _tangent_transpose_element_term(
    temperature: wp.array[float],
    y: wp.array[float],
    scattered: wp.array[float],
    out: wp.array[float],
):
    """``4 T_E^3 (y_E - (F^T y)_E)`` with ``y = c z``, the transpose of the radiative tangent."""
    e = wp.tid()
    t = temperature[e]
    out[e] = 4.0 * t * t * t * (y[e] - scattered[e])


@wp.kernel(enable_backward=False)
def _density_vjp_local(
    z: wp.array[float],
    emissive: wp.array[float],
    irradiation: wp.array[float],
    area_over_volume: float,
    conduction_radiation_number: float,
    adj_absorptivity: wp.array[float],
):
    """Absorptivity adjoint of the emission coefficient, at fixed exchange factors."""
    e = wp.tid()
    wp.atomic_add(
        adj_absorptivity,
        e,
        z[e] * conduction_radiation_number * area_over_volume * (emissive[e] - irradiation[e]),
    )


@wp.kernel(enable_backward=False)
def _negate_scaled(coefficient: wp.array[float], z: wp.array[float], out: wp.array[float]):
    e = wp.tid()
    out[e] = -coefficient[e] * z[e]


@wp.kernel(enable_backward=False)
def _radiation_diagonal(
    nx: int,
    ny: int,
    coefficient: wp.array[float],
    temperature: wp.array[float],
    element_volume: float,
    out: wp.array[float],
):
    """Local emission contribution to the tangent diagonal, for Jacobi preconditioning.

    Absorption from other elements is deliberately omitted: including it would need the
    exchange factor diagonal, and a preconditioner only has to be approximate.
    """
    i, j = wp.tid()
    e = i * ny + j
    t = temperature[e]
    contribution = coefficient[e] * 4.0 * t * t * t * element_volume * 0.0625

    for a in range(4):
        di = wp.where(a == 1 or a == 2, 1, 0)
        dj = wp.where(a == 2 or a == 3, 1, 0)
        wp.atomic_add(out, (i + di) * (ny + 1) + (j + dj), contribution)


class CoupledResidual2D:
    """Steady conduction-radiation residual, satisfying the implicit-solve protocol.

    Args:
        conduction: Conduction operator over the grid.
        radiation: Radiative exchange operator over the same elements.
        conduction_radiation_number: :math:`N_R`, the relative strength of radiation.
        dirichlet: Prescribed-temperature nodes, or None for a fully insulated problem.
        environment_temperature: Ambient temperature seen by escaping rays.
        device: Device to allocate on.

    Note:
        ``update_design`` must be called before any evaluation, and again whenever the
        density field changes; it is where the exchange factors are re-derived.
    """

    def __init__(
        self,
        conduction: ConductionOperator2D,
        radiation,
        conduction_radiation_number: float = 1.0,
        dirichlet: DirichletMask | None = None,
        environment_temperature: float = 0.0,
        device=None,
    ):
        self.conduction = conduction
        self.radiation = radiation
        self.conduction_radiation_number = conduction_radiation_number
        self.dirichlet = dirichlet
        self.device = device if device is not None else conduction.device

        nx, ny = conduction.res
        # Perimeter over area for a grid cell; the 2D analogue of surface area per volume.
        self.area_over_volume = 2.0 * (nx + ny)

        n_elem = conduction.element_count
        n_node = conduction.node_count

        self.environment_emissive = wp.array([environment_temperature**4], dtype=float, device=self.device)

        self._conductivity = wp.zeros(n_elem, dtype=float, device=self.device)
        self._absorptivity = wp.zeros(n_elem, dtype=float, device=self.device)
        self._coefficient = wp.zeros(n_elem, dtype=float, device=self.device)
        self._source = wp.zeros(n_node, dtype=float, device=self.device)

        self._element_t = wp.zeros(n_elem, dtype=float, device=self.device)
        self._emissive = wp.zeros(n_elem, dtype=float, device=self.device)
        self._irradiation = wp.zeros(n_elem, dtype=float, device=self.device)
        self._element_work = wp.zeros(n_elem, dtype=float, device=self.device)
        self._element_work2 = wp.zeros(n_elem, dtype=float, device=self.device)
        self._element_work3 = wp.zeros(n_elem, dtype=float, device=self.device)
        self._node_work = wp.zeros(n_node, dtype=float, device=self.device)
        self._environment_work = wp.zeros(1, dtype=float, device=self.device)

        self._linearized = wp.zeros(n_elem, dtype=float, device=self.device)

    # -- design and load -----------------------------------------------------------------

    def update_design(self, conductivity: wp.array, absorptivity: wp.array):
        """Re-linearize the operator for a new design.

        Args:
            conductivity: Per-element dimensionless conductivity.
            absorptivity: Per-element absorptivity, which by Kirchhoff's law is also the
                emissivity.
        """
        self._conductivity.assign(conductivity)
        self._absorptivity.assign(absorptivity)
        self.radiation.update(self._absorptivity)

        wp.launch(
            _radiation_coefficient,
            dim=self._coefficient.shape[0],
            inputs=[self._absorptivity, self.area_over_volume, self.conduction_radiation_number],
            outputs=[self._coefficient],
            device=self.device,
        )

    def set_volumetric_source(self, source: wp.array):
        """Set the internal heat generation, given per element.

        Args:
            source: Per-element dimensionless volumetric heat generation.
        """
        self._source.zero_()
        self.conduction.scatter_element_source(source, self._source)

    # -- residual protocol ---------------------------------------------------------------

    def evaluate(self, state: wp.array, out: wp.array):
        """Write the residual at ``state`` into ``out``."""
        self.conduction.apply(self._conductivity, state, out)

        self.conduction.element_average(state, self._element_t)
        wp.launch(
            _emissive_power,
            dim=self._element_t.shape[0],
            inputs=[self._element_t],
            outputs=[self._emissive],
            device=self.device,
        )
        self.radiation.apply(self._emissive, self._irradiation, self.environment_emissive)
        wp.launch(
            _net_radiation,
            dim=self._element_t.shape[0],
            inputs=[self._emissive, self._irradiation, self._coefficient],
            outputs=[self._element_work],
            device=self.device,
        )

        self.conduction.scatter_element_source(self._element_work, out)
        wp.launch(_axpy_neg, dim=out.shape[0], inputs=[self._source], outputs=[out], device=self.device)

    def relinearize(self, state: wp.array):
        """Store the element temperatures the tangent is linearized about."""
        self.conduction.element_average(state, self._linearized)

    def tangent(self, v: wp.array, out: wp.array):
        """Write ``(dR/dT) v`` into ``out``."""
        self.conduction.apply(self._conductivity, v, out)

        self.conduction.element_average(v, self._element_work)
        wp.launch(
            _emissive_tangent,
            dim=self._element_work.shape[0],
            inputs=[self._linearized, self._element_work],
            outputs=[self._element_work2],
            device=self.device,
        )
        # Transport is linear in emissive power, so the forward apply doubles as the tangent
        # of absorption; the environment term is constant and drops out.
        self._environment_work.zero_()
        self.radiation.apply(self._element_work2, self._element_work3, self._environment_work)
        wp.launch(
            _tangent_element_term,
            dim=self._element_work.shape[0],
            inputs=[self._coefficient, self._element_work2, self._element_work3],
            outputs=[self._element_work],
            device=self.device,
        )

        self.conduction.scatter_element_source(self._element_work, out)

    def tangent_transpose(self, v: wp.array, out: wp.array):
        """Write ``(dR/dT)^T v`` into ``out``.

        Conduction is symmetric so it transposes to itself, but the radiative term does not:
        emission is diagonal while absorption is a full exchange, and the two enter with
        opposite roles under transposition.
        """
        self.conduction.apply_transpose(self._conductivity, v, out)

        # scatter^T = V * average and average^T = scatter / V, so the element volume cancels
        # between them and the transpose reduces to scatter(D (I - F^T) C average(v)).
        n_elem = self._element_work.shape[0]

        self.conduction.element_average(v, self._element_work)
        wp.launch(
            _scale,
            dim=n_elem,
            inputs=[self._element_work, self._coefficient],
            outputs=[self._element_work2],
            device=self.device,
        )

        self._environment_work.zero_()
        self.radiation.apply_transpose(self._element_work2, self._element_work3, self._environment_work)
        wp.launch(
            _tangent_transpose_element_term,
            dim=n_elem,
            inputs=[self._linearized, self._element_work2, self._element_work3],
            outputs=[self._element_work],
            device=self.device,
        )

        self.conduction.scatter_element_source(self._element_work, out)

    def reference_norm(self) -> float:
        """Norm of the applied load, the characteristic scale of the residual.

        Newton measures convergence against this rather than against the initial residual,
        so a warm start from a neighbouring design is held to the same standard as a cold
        one instead of to an unreachable fraction of its own small starting residual.
        """
        f = self._source.numpy()
        return float((f @ f) ** 0.5)

    def project(self, v: wp.array):
        """Restrict ``v`` to the free subspace."""
        if self.dirichlet is not None:
            self.dirichlet.project(v)

    def preconditioner_diagonal(self, out: wp.array):
        """Write an approximate tangent diagonal for Jacobi preconditioning."""
        self.conduction.diagonal(self._conductivity, out)
        nx, ny = self.conduction.res
        wp.launch(
            _radiation_diagonal,
            dim=self.conduction.res,
            inputs=[nx, ny, self._coefficient, self._linearized, self.conduction.element_volume],
            outputs=[out],
            device=self.device,
        )

    # -- design sensitivity --------------------------------------------------------------

    def density_vjp(
        self,
        state: wp.array,
        adjoint: wp.array,
        adj_conductivity: wp.array,
        adj_absorptivity: wp.array,
    ):
        """Accumulate ``lambda^T dR/d(design)`` into the material adjoints.

        Splits into the conduction term, the emission coefficient's dependence on
        absorptivity, and the exchange factors' dependence on absorptivity. The last is the
        ray-march VJP; the temperature adjoint it also produces is discarded, because this is
        a partial derivative at fixed state.

        Args:
            state: Converged state.
            adjoint: Adjoint variable from :func:`~warp._src.thermal.implicit.adjoint_solve`.
            adj_conductivity: Output, per element. Accumulated.
            adj_absorptivity: Output, per element. Accumulated.
        """
        n_elem = self._element_t.shape[0]

        self.conduction.apply_vjp(state, adjoint, adj_conductivity)

        # Rebuild the forward quantities at the converged state.
        self.conduction.element_average(state, self._element_t)
        wp.launch(_emissive_power, dim=n_elem, inputs=[self._element_t], outputs=[self._emissive], device=self.device)
        self.radiation.apply(self._emissive, self._irradiation, self.environment_emissive)

        # z = scatter^T(lambda) = V * average(lambda)
        self.conduction.element_average(adjoint, self._element_work)
        wp.launch(
            _scale_by_constant,
            dim=n_elem,
            inputs=[self._element_work, self.conduction.element_volume],
            outputs=[self._element_work2],
            device=self.device,
        )

        wp.launch(
            _density_vjp_local,
            dim=n_elem,
            inputs=[
                self._element_work2,
                self._emissive,
                self._irradiation,
                self.area_over_volume,
                self.conduction_radiation_number,
            ],
            outputs=[adj_absorptivity],
            device=self.device,
        )

        wp.launch(
            _negate_scaled,
            dim=n_elem,
            inputs=[self._coefficient, self._element_work2],
            outputs=[self._element_work3],
            device=self.device,
        )
        self._environment_work.zero_()
        self.radiation.vjp(
            self._emissive,
            self.environment_emissive,
            self._element_work3,
            adj_absorptivity,
            self._element_work,
            self._environment_work,
        )


@wp.kernel(enable_backward=False)
def _axpy_neg(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = y[i] - x[i]


@wp.kernel(enable_backward=False)
def _scale_by_constant(x: wp.array[float], factor: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = x[i] * factor
