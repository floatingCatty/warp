Add the coupled conduction-radiation solve and its implicit gradient. `warp.thermal.CoupledResidual2D`
combines conduction with volumetric radiative exchange into a single nonlinear residual, and
`warp.thermal.newton_solve` and `warp.thermal.adjoint_solve` provide the physics-agnostic forward solve and
implicit-function-theorem adjoint that any residual satisfying the protocol can reuse. This completes the
chain from a density field through a converged conduction-radiation solve to a design gradient, verified
against finite differences of a full re-solve.
