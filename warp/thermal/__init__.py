# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable heat transfer and thermal radiation.

Couples finite element heat conduction to ray-traced thermal radiation, with gradients
of the converged solution with respect to geometry and material parameters, for shape and
topology optimization of radiative cooling devices.

Usage:
    This module must be explicitly imported::

        import warp.thermal
"""

# isort: skip_file

# The source-to-public Warp module declarations for `warp.thermal` live in the
# top-level `warp/__init__.py`, so they are in effect before these imports run.

from warp._src.thermal.material import absorptivity_from_density as absorptivity_from_density
from warp._src.thermal.material import conductivity_from_density as conductivity_from_density
from warp._src.thermal.implicit import NewtonResult as NewtonResult
from warp._src.thermal.optim import optimality_criteria_step as optimality_criteria_step
from warp._src.thermal.implicit import adjoint_solve as adjoint_solve
from warp._src.thermal.implicit import newton_solve as newton_solve
from warp._src.thermal.volumetric.conduction import ConductionOperator2D as ConductionOperator2D
from warp._src.thermal.volumetric.coupled import CoupledResidual2D as CoupledResidual2D
from warp._src.thermal.volumetric.conduction import DirichletMask as DirichletMask
from warp._src.thermal.volumetric.exchange import ExchangeMatrix as ExchangeMatrix
from warp._src.thermal.volumetric.filter import DensityFilter2D as DensityFilter2D
from warp._src.thermal.volumetric.grid import GridRayBundle2D as GridRayBundle2D
from warp._src.thermal.volumetric.operator import VolumetricRadiationOperator as VolumetricRadiationOperator
from warp._src.thermal.volumetric.paths import EXIT_ADIABATIC as EXIT_ADIABATIC
from warp._src.thermal.volumetric.paths import EXIT_ENVIRONMENT as EXIT_ENVIRONMENT
from warp._src.thermal.volumetric.paths import RayPaths as RayPaths
from warp._src.thermal.volumetric.paths import trace_grid_2d as trace_grid_2d
from warp._src.thermal.volumetric.transport import transport as transport
from warp._src.thermal.volumetric.transport import transport_transpose as transport_transpose
from warp._src.thermal.volumetric.transport import transport_vjp as transport_vjp
