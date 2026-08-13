# Differentiable Thermal Design

**Status**: In Progress

## Motivation

Spacecraft reject waste heat by radiation alone. A radiator's performance is set by the
interaction of three mechanisms that cannot be optimized separately: conduction spreads heat
through the structure, the resulting temperature field drives emission as `T^4`, and the
geometry decides both how much area radiates and how much of that area sees itself rather
than space. Designing against all three at once needs gradients of a converged
conduction-radiation solve with respect to geometry and material parameters.

`warp.thermal` provides that gradient. The target workflow is

```
parameterized geometry -> heat transport -> radiation -> objective -> gradient
```

with the physics solved to convergence and differentiated through, rather than approximated
by a surrogate.

Existing topology optimization work on conduction-radiation problems derives design
sensitivities by hand for one fixed formulation. Noguchi et al., *Topology optimization of
conduction-radiation problems based on a ray-tracing approach*
([arXiv:2607.28534](https://arxiv.org/abs/2607.28534)), is the reference implementation this
module is validated against; its radiation discretization is reproduced exactly, its
sensitivity derivation is not. Hand-derived sensitivities have to be re-derived for every new
objective, material model, or design variable. An implicit-function-theorem adjoint composed
with automatic differentiation generalizes for free.

## Requirements

| ID  | Requirement                                                            | Priority | Notes                                              |
| --- | ---------------------------------------------------------------------- | -------- | -------------------------------------------------- |
| R1  | Gradients of a converged nonlinear solve w.r.t. design variables        | Must     | Implicit adjoint, not backprop through the iterates |
| R2  | Surface-to-surface radiation with mutual visibility and occlusion       | Must     | Ray traced, not a boundary-flux approximation       |
| R3  | Gradients verified against directional finite differences               | Must     | Every operator, independently                       |
| R4  | Memory independent of ray count                                         | Must     | Realistic angular discretizations are ~1e7 rays     |
| R5  | Reproduce published conduction-radiation topology optimization results  | Should   | Verification backend, see Regime A                  |
| R6  | Explicit deformed geometry with shape parameters                        | Should   | The design target, see Regime B                     |
| R7  | Gray-diffuse radiation with emissivity as a design variable             | Should   | Needs a radiosity solve                             |
| R8  | Fluid heat transport coupled to solid conduction and radiation          | Could    | Later; FVM                                          |

**Non-goals**: discrete design variables (component counts, connection topology,
combinatorial arrangement); spectral or specular radiation; participating gases.

## Design

### Two regimes, one set of abstractions

The module carries two radiation discretizations. They are not variants of one another and
they do not share assembly code:

|                  | Regime A                             | Regime B                            |
| ---------------- | ------------------------------------ | ----------------------------------- |
| geometry         | density field on a structured grid   | `x = Phi_theta(X)`, deformed mesh   |
| radiation couples as | volumetric source term           | boundary term over `dOmega`         |
| visibility       | DDA traversal, attenuated by density | BVH ray queries against the surface |
| design variables | element densities                    | shape parameters                    |

Regime A is a **verification backend**: it has published results to check against and is
smooth in its design variables by construction, which makes it a clean testbed for the
autodiff, radiation and implicit-adjoint machinery. Regime B is the **product backend**.

It is tempting to describe Regime A as a smooth relaxation of Regime B. That framing is
wrong and leads to a bad architecture. They discretize radiation differently — one deposits
energy into element volumes, the other integrates flux over surfaces — so almost nothing in
their assembly is shared. What *is* shared:

```
Residual(u, theta) -> JVP -> VJP -> implicit solve -> optimizer -> verification
```

A practical consequence: **Regime A's `dJ/drho` must never be used to validate Regime B's
shape gradient.** Regime B needs its own directional finite-difference check, because
visibility is discontinuous in `theta` there and smooth in `rho` here.

### Operator first, matrix optional

Radiative exchange between `N` elements is dense in the worst case. Making the exchange
factor matrix `F` the central data structure imports that density into the architecture.
Instead, radiation is exposed as an operator with three actions — forward, `Jv`, and
`J^T v` — and whether a matrix backs any of them is an implementation detail.

The tangent is **not symmetric**, so the adjoint solve genuinely needs `J^T`, not a second
application of `J`. That is why the transpose is a first-class operation rather than an alias.

### Forward and adjoint use different strategies

Within one design iteration the two run at very different frequencies:

| Stage             | Calls per design iteration | Strategy                              |
| ----------------- | -------------------------- | ------------------------------------- |
| forward / tangent | ~100 (Newton x Krylov)     | assemble exchange factors once, reuse |
| adjoint (VJP)     | 1                          | ray march                             |

Absorptivity is constant across a design iteration, so the exchange factors are too and can
be amortized over every Krylov apply. The adjoint gains nothing from assembly: the
dependence of the exchange factors on absorptivity lives in per-ray transmittance products
that an assembled matrix does not retain.

Selection is gated on **work as well as memory**. An assembled apply costs `O(n_elem^2)`
however the rays fall, while a marching apply costs one step per ray step, so assembly only
pays when the angular discretization is dense enough for rays to revisit the same element
pairs. Measured on CPU:

| `n_elem^2 / ray-steps` | march   | assembled |
| ---------------------- | ------- | --------- |
| 2.13                   | 1069 ms | 1271 ms   |
| 0.88                   | 92 ms   | 40 ms     |
| 0.11                   | 664 ms  | 39 ms     |
| 0.09                   | 1892 ms | 87 ms     |

The reference 2D problem sits at 0.025. A memory-only gate would have chosen assembly for
coarse ray bundles where marching is faster.

### Ray bundles are implicit

Storing each ray's element sequence does not survive a realistic angular discretization: the
reference 2D problem emits 14.4 million rays covering 513 million traversal steps, over 2 GB
of indices that every consumer reads exactly once per design iteration.

`GridRayBundle2D` derives each ray from its index and walks the grid inline, storing nothing.
Only the adjoint buffers per-step state, and only for a bounded chunk of rays at a time. A 2D
grid walk visits at most `res[0] + res[1] + 1` cells, which sizes that buffer exactly. Chunk
size therefore trades memory against launch count without changing results.

`RayPaths`, the explicit representation, is retained: it is the oracle the implicit bundle is
verified against, and it is the natural representation for Regime B, where tracing is
expensive enough to be worth caching across a design iteration.

### Why not hardware ray tracing

Warp has no OptiX or RT-core path; its BVH is a software implementation with `sah`, `median`,
`lbvh` and `cubql` constructors, traversed in ordinary kernels. Independently of that, RT
acceleration would not help Regime A:

- BVH traversal and RT cores accelerate **sparse nearest-hit queries with early termination**.
- Regime A needs **every cell along the ray**, accumulating attenuation. Enumerating all
  intersections is the slow path for RT hardware, and would require manufacturing geometry
  for every cell face.
- A uniform-grid DDA has *no* acceleration structure to traverse. It is O(1) per cell in
  register arithmetic, which is already the standard primitive for volume traversal.

Regime B is the opposite case. "Is patch `j` visible from patch `i`, or occluded?" is exactly
a nearest-hit or any-hit query, and `wp.mesh_query_ray` / `wp.mesh_query_ray_anyhit` are the
right tools there.

Measurement supports this. On the reference problem the adjoint takes 2.8x longer than
assembly for an *identical* traversal; the difference is its scratch traffic, not the walk.
Traversal is not the bottleneck, so a faster ray tracer would not move it.

### The autodiff hazard

Warp does not replay dynamic loops in the backward pass, so a running product whose adjoint
depends on intermediate partial products gets **silently wrong gradients** — see
`docs/user_guide/differentiability.rst`. Ray transmittance, `tau_k = tau_{k-1} (1 - f_k)`, is
exactly that pattern. Measured on the transport kernel:

```
finite difference (reference) : [0.15347 0.27092 0.09344 0.3487  0.45467 0.08214]
warp AD, dynamic loop         : [0.01213 0.06042 0.03784 0.20321 0.3725  0.24444]
warp AD, static/unrolled loop : [0.15347 0.27092 0.09344 0.3487  0.45467 0.08214]
```

The error is O(1), not a tolerance issue, and nothing is raised. Unrolling is exact but
unavailable: real paths run to ~170 cells, far past `max_unroll`. The transport kernels
therefore set `enable_backward=False` and supply a hand-written adjoint built on the
front-to-back compositing recurrence

```
A_{n+1} = E_exit,   A_k = f_k E_k + (1 - f_k) A_{k+1},   dg/df_j = tau_{j-1} (E_j - A_{j+1})
```

which is division-free, so fully opaque cells (`f -> 1`, `tau -> 0`) stay well behaved where
recovering `tau_{j-1}` as `tau_n / tau'_j` would produce NaN.

The operator acts on **absorptivity, not density**. Material interpolation and filtering are
elementwise maps that Warp differentiates correctly on its own, so keeping them outside
confines the hand-written adjoint to the single kernel that needs it, and lets the same
operator carry surface emissivity in Regime B.

### Design-dependent branches are deferred

Terminating rays at `tau < eps`, or skipping void regions with a sparse grid, are both real
optimizations and both introduce a **design-dependent branch**: the objective and its
gradient become discontinuous where the design crosses the threshold. Neither is enabled
until gradients are fully verified, and both must then be accepted on evidence that the
objective and gradient converge as `eps -> 0`.

## Alternatives Considered

**Backpropagate through the Newton iterates.** Memory grows with iteration count and the
gradient is only as converged as the solve. The implicit adjoint costs one linear solve
regardless.

**Cache ray paths across a design iteration.** Strictly dominated in Regime A: it uses ~44x
more memory than the assembled exchange factors (no deduplication of repeated element pairs)
*and* is slower to apply than a matrix-vector product. It becomes the right choice in Regime B,
where re-tracing costs far more than a streaming read.

**Sparse exchange factors.** Worth ~4.5x in 3D, close to nothing in 2D where the fill is high
enough that indices cost as much as the dense values they replace. Deferred until a 3D case
needs it; the marching backend already covers problems too large to assemble.

**`warp.fem` for Regime A conduction.** On a uniform structured grid with cell-wise constant
conductivity the bilinear element operator is analytic and its density derivative is exactly
the element matrix, so a direct implementation is simpler and makes the adjoint trivial.
Regime B, with deformed geometry and curved elements, is where `warp.fem` earns its cost.

## Testing Strategy

Every operator is verified independently before anything is built on it:

- **Independent reference implementations.** Kernels use the compositing recurrence; the
  NumPy references use the explicit attenuated-sum integral. Agreement is evidence about the
  math, not a restatement of the same code. All references run in float64.
- **Directional finite differences** on every gradient, plus elementwise checks to catch what
  a single projection could hide.
- **Adjoint identity** `<Ju, v> == <u, J^T v>`, which finite differences cannot perform. A
  transposed operator can be wrong in a way that passes every forward and FD check while
  quietly corrupting the adjoint solve.
- **Cross-strategy equivalence.** Assembled and marching backends, and implicit and explicit
  ray bundles, must be indistinguishable to callers.
- **Degenerate inputs.** Fully opaque and fully transparent cells, zero-length ray paths,
  repeated cells within a path.
- **Conservation.** Per-element quadrature weights sum to one, converging as `O(1/N_ang^2)`.
- **Mutation testing.** Each verification suite is checked by injecting the defect it exists
  to catch and confirming it fails.

A limit worth stating: the module computes in single precision, so the residual cannot be
evaluated more accurately than roughly `1e-6` relative. Newton converges quadratically down to
that floor and then stalls, which is why convergence is measured relative to the initial
residual and stagnation is reported rather than iterated against. This is ample for
gradient-based design, where the finite-difference checks agree to a few percent, but it rules
out asking the solver for tighter residuals without a double-precision path.

Device coverage follows `get_test_devices()`. Note that all measurements recorded here were
taken on CPU; the assembled apply is bandwidth-bound and the march is register-bound, so the
backend crossover may move on GPU and should be re-measured.

## Milestones

| Milestone | Content                                                        | Status      |
| --------- | -------------------------------------------------------------- | ----------- |
| M2        | Radiative transport primitive and its adjoint                   | Done        |
| M2b       | Forward/adjoint strategy split; implicit grid ray bundles       | Done        |
| M1        | Conduction operator; coupled nonlinear solve; implicit adjoint  | Done        |
| M3        | Reproduce the reference 2D radiative heat sink                  | In progress |
| M4        | Regime B: explicit deformed geometry, surface radiation         | Planned     |
| M5        | Gray radiosity; deployable radiator demonstration               | Planned     |

The reference reproduction (M3) establishes credibility and stops there. The published 3D and
radiation-shield cases are the reference authors' research questions, not this module's, and
are revisited only if a scaling number is wanted. Regime B is the point.

## Current state

Everything M3 needs is built and verified: transport, exchange assembly, implicit ray bundles,
conduction, the coupled residual, the implicit adjoint, the density filter, optimality criteria,
and `warp/examples/thermal/example_radiative_heat_sink.py`. The design loop descends
monotonically and holds the volume constraint exactly.

**The reproduction itself has not been run.** No converged design exists, and neither the
objective nor the structure has been compared with the published result. Running it is the
next action:

```
python -u warp/examples/thermal/example_radiative_heat_sink.py --resolution 60 --iterations 200
```

The example prints the published objective and the percentage difference automatically when the
configuration matches. Two things to check, not one: the converged objective against
`6.9041e-4`, and the structure against the published figure — thick branches from the source to
the upper corners, thin ones to the lower corners, with corners filled. The second is the
substantive check, since a plausible objective can come from a different structure.

Discard any earlier objective trajectory recorded outside this document: values produced before
the objective was switched to its integral form used a gradient weighted for the mean, which is
wrong by the ratio of the two definitions and scaled the descent direction accordingly.

### Known gaps

- **No end-to-end finite-difference check of the design gradient.** The objective's response to
  a design perturbation sits at the single-precision noise floor, where the difference quotient
  varies by more than the quantity it is measuring, and the tolerance needed to fix that is
  unreachable in `float32`. The chain is verified link by link instead: the density VJP against
  finite differences, the filter transpose by adjoint identity, the SIMP derivatives
  analytically, and `dJ/dT` against the objective directly. That is sound but weaker than one
  end-to-end check, and it is how a factor-of-16 error in `dJ/dT` survived until the last link
  got its own test. A `float64` path would close this.
- **CPU only so far.** Every measurement in this document was taken single-threaded, because
  Warp's CPU backend runs a serial loop over the launch dimension. The assemble-versus-march
  crossover, the cost of one design iteration, and the choice of dense over sparse exchange
  factors should all be re-measured on GPU before being treated as settled.
- **Host synchronization in the solver loop.** Newton reads the residual norm back to the host
  every iteration and inside the line search, and optimality criteria does one readback per
  bisection step. Free on CPU, a sync point per call on GPU. If a GPU profile looks
  latency-bound rather than bandwidth-bound, this is why.
- **Preconditioning is weak.** Jacobi uses the conduction diagonal plus the local emission term
  and ignores the radiative coupling entirely, which is what dominates the tangent at
  `N_R = 1`. Around 46 Krylov iterations per Newton step suggests real headroom.
