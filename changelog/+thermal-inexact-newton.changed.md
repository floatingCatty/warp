Solve conduction-radiation tangent systems inexactly, to a tolerance that tracks the residual's distance
from where it started, and let `warp.thermal.adjoint_solve` warm-start from a previous adjoint. Together with
an unrolled dense exchange apply this roughly halves the cost of a design iteration at identical final
accuracy. Pass `forcing_max` equal to `linear_tol` to recover exact Newton, which converges quadratically
and solves a linear problem in a single step.
