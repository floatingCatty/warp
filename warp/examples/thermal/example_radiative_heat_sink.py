# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Topology Optimization of a Radiative Heat Sink
#
# Designs a two-dimensional radiative heat sink for a vacuum environment, where heat can
# only leave by thermal radiation. A heat source at the bottom of a square design domain
# must be kept cool using a limited amount of material.
#
# The design has to balance two mechanisms that pull against each other:
#
#   - Radiation scales with T^4, so it wants hot material with a large exposed area.
#   - Conduction must carry heat from the source out to that area, and long thin branches
#     conduct poorly.
#
# Neither effect alone predicts the answer, and material that shades other material makes it
# worse still, so the optimizer has to account for elements radiating at each other.
#
# Physics:
#   - Steady conduction with element-wise conductivity from a density field (SIMP).
#   - Zonal radiation: every element is an isothermal zone, exchanging energy along traced
#     rays. Intermediate densities act as a participating medium, which is what lets
#     radiation act on boundaries that exist only implicitly in the density field.
#   - Rays leaving upward reach deep space at absolute zero; rays leaving downward are
#     reflected, modelling a symmetry plane or an insulated mounting surface.
#
# Optimization:
#   - Minimize the mean temperature over the heat source, subject to a volume constraint.
#   - Sensitivities come from an implicit adjoint of the converged nonlinear solve, so their
#     cost does not grow with the number of Newton iterations.
#   - Densities are filtered with a cone kernel to impose a minimum length scale, and updated
#     by optimality criteria.
#
# The default settings follow Noguchi et al., "Topology optimization of conduction-radiation
# problems based on a ray-tracing approach" (arXiv:2607.28534), Section 3.2.
###########################################################################

import numpy as np

import warp as wp
import warp.thermal as thermal


class Example:
    def __init__(
        self,
        resolution=60,
        volume_fraction=0.3,
        conduction_radiation_number=1.0,
        filter_radius=0.05,
        launch_points=10,
        angles=100,
        source_width=12,
        penalty=2.0,
        quiet=False,
        device=None,
    ):
        self.res = (resolution, resolution)
        self.volume_fraction = volume_fraction
        self.penalty = penalty
        self.quiet = quiet
        self.device = device

        n_elem = resolution * resolution
        self.n_elem = n_elem

        self.conduction = thermal.ConductionOperator2D(self.res, device=device)
        bundle = thermal.GridRayBundle2D(self.res, launch_points=launch_points, angles=angles, device=device)
        self.radiation = thermal.VolumetricRadiationOperator(bundle, n_elem, device=device)
        self.residual = thermal.CoupledResidual2D(
            self.conduction,
            self.radiation,
            conduction_radiation_number=conduction_radiation_number,
            device=device,
        )
        self.filter = thermal.DensityFilter2D(self.res, filter_radius, device=device)

        # Heat source occupying a strip at the bottom centre, which is also the region whose
        # temperature the objective measures.
        lo = (resolution - source_width) // 2
        source = np.zeros(n_elem, dtype=np.float32)
        for i in range(lo, lo + source_width):
            source[i * resolution + 0] = 1.0
        self.objective_mask = source.copy()
        self.residual.set_volumetric_source(wp.array(source, dtype=float, device=device))

        # dJ/dT for J = mean temperature over the source region.
        weight = source / max(source.sum(), 1.0) / self.conduction.element_volume
        self.objective_gradient = wp.zeros(self.conduction.node_count, dtype=float, device=device)
        self.conduction.scatter_element_source(wp.array(weight, dtype=float, device=device), self.objective_gradient)

        self.density = wp.array(np.full(n_elem, volume_fraction, dtype=np.float32), dtype=float, device=device)
        self.filtered = wp.zeros(n_elem, dtype=float, device=device)
        self.conductivity = wp.zeros(n_elem, dtype=float, device=device)
        self.absorptivity = wp.zeros(n_elem, dtype=float, device=device)
        self.temperature = wp.array(np.full(self.conduction.node_count, 1.0, dtype=np.float32), dtype=float)
        self.adjoint = wp.zeros(self.conduction.node_count, dtype=float, device=device)

        self.adj_conductivity = wp.zeros(n_elem, dtype=float, device=device)
        self.adj_absorptivity = wp.zeros(n_elem, dtype=float, device=device)
        self.adj_filtered = wp.zeros(n_elem, dtype=float, device=device)
        self.sensitivity = wp.zeros(n_elem, dtype=float, device=device)
        self.element_temperature = wp.zeros(n_elem, dtype=float, device=device)

        self._adjoint_ready = False
        self.history = []

    def solve(self):
        """Filter the design, rebuild the physics, and solve to equilibrium."""
        self.filter.apply(self.density, self.filtered)
        thermal.conductivity_from_density(self.filtered, self.conductivity, penalty=self.penalty)
        thermal.absorptivity_from_density(self.filtered, self.absorptivity, penalty=self.penalty)
        self.residual.update_design(self.conductivity, self.absorptivity)
        return thermal.newton_solve(self.residual, self.temperature, max_iterations=40)

    def objective(self):
        """Mean temperature over the heat source region."""
        self.conduction.element_average(self.temperature, self.element_temperature)
        t = self.element_temperature.numpy().astype(np.float64)
        return float((t * self.objective_mask).sum() / self.objective_mask.sum())

    def gradient(self):
        """Sensitivity of the objective with respect to the design variables."""
        # The adjoint changes slowly between design iterations, so the previous one is a
        # far better starting point than zero.
        thermal.adjoint_solve(
            self.residual,
            self.temperature,
            self.objective_gradient,
            self.adjoint,
            warm_start=self._adjoint_ready,
        )
        self._adjoint_ready = True

        self.adj_conductivity.zero_()
        self.adj_absorptivity.zero_()
        self.residual.density_vjp(self.temperature, self.adjoint, self.adj_conductivity, self.adj_absorptivity)

        # Chain through the SIMP interpolations, then back through the density filter.
        rho = self.filtered.numpy().astype(np.float64)
        d_k = self.penalty * (1.0 - 1.0e-8) * rho ** (self.penalty - 1.0)
        d_a = self.penalty * rho ** (self.penalty - 1.0)
        chained = self.adj_conductivity.numpy().astype(np.float64) * d_k
        chained += self.adj_absorptivity.numpy().astype(np.float64) * d_a

        self.adj_filtered.assign(chained.astype(np.float32))
        self.filter.apply_transpose(self.adj_filtered, self.sensitivity)
        return self.sensitivity

    def step(self, iteration):
        result = self.solve()
        value = self.objective()
        sensitivity = self.gradient()
        thermal.optimality_criteria_step(self.density, sensitivity, self.volume_fraction)

        self.history.append(value)
        if not self.quiet:
            state = "" if result.converged else "  [solve did not converge]"
            newton = f"{result.iterations:2d}" + ("*" if result.stalled else " ")
            print(
                f"iter {iteration:4d}   J = {value:.6e}   "
                f"newton = {newton}  volume = {float(self.density.numpy().mean()):.4f}{state}"
            )
        return value

    def run(self, iterations=200, tolerance=1.0e-7):
        for i in range(iterations):
            value = self.step(i)
            if i > 0 and abs(value - self.history[-2]) <= tolerance * abs(value):
                if not self.quiet:
                    print(f"converged after {i + 1} design iterations")
                break
        return self.history[-1]

    def save(self, path):
        np.savez(
            path,
            density=self.density.numpy(),
            filtered=self.filtered.numpy(),
            temperature=self.temperature.numpy(),
            history=np.array(self.history),
            resolution=self.res[0],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=60, help="Elements along each axis.")
    parser.add_argument("--iterations", type=int, default=200, help="Maximum design iterations.")
    parser.add_argument("--volume-fraction", type=float, default=0.3, help="Fraction of the domain filled.")
    parser.add_argument(
        "--conduction-radiation-number",
        type=float,
        default=1.0,
        help="N_R, the strength of radiation relative to conduction.",
    )
    parser.add_argument("--launch-points", type=int, default=10, help="Ray launch points per element face.")
    parser.add_argument("--angles", type=int, default=100, help="Ray directions per launch point.")
    parser.add_argument("--filter-radius", type=float, default=0.05, help="Density filter radius.")
    parser.add_argument("--save", type=str, default=None, help="Path to save the optimized design as .npz.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            resolution=args.resolution,
            volume_fraction=args.volume_fraction,
            conduction_radiation_number=args.conduction_radiation_number,
            filter_radius=args.filter_radius,
            launch_points=args.launch_points,
            angles=args.angles,
            quiet=args.quiet,
        )
        final = example.run(iterations=args.iterations)
        print(f"final objective: {final:.6e}")

        if args.save:
            example.save(args.save)
