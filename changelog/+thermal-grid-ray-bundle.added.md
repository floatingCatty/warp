Add `warp.thermal.GridRayBundle2D`, an implicit ray bundle over a 2D structured grid. Rays are derived
from their index and the grid is walked inline, so memory no longer scales with the ray count: a 60x60
domain with 10 launch points and 100 angles per face emits 14.4 million rays covering 513 million
traversal steps, which would need over 2 GB to store as explicit paths. Only the adjoint buffers
per-step state, bounded by `chunk_rays`. `warp.thermal.VolumetricRadiationOperator` accepts either this
or a `warp.thermal.RayPaths`.
