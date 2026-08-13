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

from warp._src.thermal.volumetric.paths import EXIT_ADIABATIC as EXIT_ADIABATIC
from warp._src.thermal.volumetric.paths import EXIT_ENVIRONMENT as EXIT_ENVIRONMENT
from warp._src.thermal.volumetric.paths import RayPaths as RayPaths
from warp._src.thermal.volumetric.paths import trace_grid_2d as trace_grid_2d
from warp._src.thermal.volumetric.transport import transport as transport
from warp._src.thermal.volumetric.transport import transport_transpose as transport_transpose
from warp._src.thermal.volumetric.transport import transport_vjp as transport_vjp
