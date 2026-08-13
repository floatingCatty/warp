Report the radiative heat sink example's objective as the temperature integrated over the target region
rather than its mean, matching the formulation of the published result it is compared against, and fix the
objective gradient, which was still weighted for the mean. Optimality criteria now bisects the Lagrange
multiplier logarithmically; bisecting linearly across a bracket spanning many decades resolved small
multipliers too coarsely and let the volume constraint drift.
