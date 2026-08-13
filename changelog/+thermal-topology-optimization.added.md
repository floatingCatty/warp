Add the design loop for conduction-radiation topology optimization: `warp.thermal.DensityFilter2D` applies
a cone-kernel filter that imposes a minimum length scale, and `warp.thermal.optimality_criteria_step`
updates densities under an exactly enforced volume constraint. The new
`warp/examples/thermal/example_radiative_heat_sink.py` puts these together with the coupled solve and its
implicit adjoint to design a radiative heat sink for a vacuum environment.
