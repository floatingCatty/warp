Add `warp.thermal`, a module for differentiable heat transfer and thermal radiation. This first release
provides the volumetric radiative transport primitive: `warp.thermal.RayPaths` caches the ray traversal
topology of a fixed mesh, and `warp.thermal.transport()` accumulates incident radiative power on every
element without ever forming the exchange-factor matrix. `warp.thermal.transport_vjp()` and
`warp.thermal.transport_transpose()` supply the adjoint and the transpose of the nonsymmetric radiative
tangent, both hand-written because Warp does not replay the dynamic loops a ray march compiles to.
