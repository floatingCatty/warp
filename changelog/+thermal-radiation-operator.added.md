Add `warp.thermal.VolumetricRadiationOperator`, which selects its forward and adjoint strategies
independently. Forward and tangent applies run once per Krylov iteration, so they can amortize an
assembled `warp.thermal.ExchangeMatrix`; the adjoint runs once per design iteration and always marches
rays. `backend="auto"` assembles only when the exchange factors both fit in memory and cost less work
than re-marching, since an assembled apply is `O(n_elem^2)` regardless of how densely the rays sample.
