# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material interpolation between void and solid.

Density-based design lets an element take any density between void and solid, and the
optimizer has to be discouraged from settling there. SIMP interpolation makes intermediate
densities buy less physical property than they cost in volume, by raising the density to a
penalization exponent.

Conduction and radiation are penalized independently, because their exponents control
different trade-offs: conductivity decides whether a partly solid element is worth using as a
heat path, absorptivity decides whether it is worth using as a radiating surface.

These are elementwise maps with no loops, so Warp differentiates them correctly. Keeping them
outside the radiative transport operator is what confines that operator's hand-written adjoint
to the one kernel that genuinely needs it.
"""

from __future__ import annotations

import warp as wp

__all__ = ["absorptivity_from_density", "conductivity_from_density"]


@wp.func
def simp_conductivity(density: float, penalty: float, minimum: float) -> float:
    """Dimensionless conductivity of an element, ``k_min + (1 - k_min) rho^p``.

    The floor keeps the conduction operator nonsingular where the design is fully void.
    """
    return minimum + (1.0 - minimum) * wp.pow(density, penalty)


@wp.func
def simp_absorptivity(density: float, penalty: float) -> float:
    """Absorptivity of an element, ``rho^q``.

    By Kirchhoff's law this is also its emissivity, which is what keeps radiative exchange
    across intermediate densities thermodynamically consistent.
    """
    return wp.pow(density, penalty)


@wp.kernel
def _conductivity_from_density(
    density: wp.array[float],
    penalty: float,
    minimum: float,
    conductivity: wp.array[float],
):
    e = wp.tid()
    conductivity[e] = simp_conductivity(density[e], penalty, minimum)


@wp.kernel
def _absorptivity_from_density(
    density: wp.array[float],
    penalty: float,
    absorptivity: wp.array[float],
):
    e = wp.tid()
    absorptivity[e] = simp_absorptivity(density[e], penalty)


def conductivity_from_density(
    density: wp.array,
    conductivity: wp.array,
    penalty: float = 2.0,
    minimum: float = 1.0e-8,
):
    """Map element densities to dimensionless conductivities.

    Args:
        density: Per-element density in ``[0, 1]``.
        conductivity: Output, per element.
        penalty: SIMP exponent for conduction.
        minimum: Conductivity floor for void elements.
    """
    wp.launch(
        _conductivity_from_density,
        dim=density.shape[0],
        inputs=[density, penalty, minimum],
        outputs=[conductivity],
        device=density.device,
    )


def absorptivity_from_density(density: wp.array, absorptivity: wp.array, penalty: float = 2.0):
    """Map element densities to absorptivities.

    Args:
        density: Per-element density in ``[0, 1]``.
        absorptivity: Output, per element.
        penalty: SIMP exponent for radiation.
    """
    wp.launch(
        _absorptivity_from_density,
        dim=density.shape[0],
        inputs=[density, penalty],
        outputs=[absorptivity],
        device=density.device,
    )
