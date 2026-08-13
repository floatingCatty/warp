Add `warp.thermal.ConductionOperator2D`, bilinear heat conduction on a uniform 2D grid with element-wise
conductivity, together with `warp.thermal.DirichletMask` and the SIMP interpolation helpers
`warp.thermal.conductivity_from_density` and `warp.thermal.absorptivity_from_density`. The operator also
provides the two transfer operators that couple conduction to radiation: `element_average()` produces the
representative temperature that drives emission, and `scatter_element_source()` returns a volumetric
element source to the nodes.
