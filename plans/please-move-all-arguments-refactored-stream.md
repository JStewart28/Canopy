# Refactor Solver constructor args into FmmConfig + asymmetric bbox tolerances

## Context

The `Canopy::Solver` constructor takes a long positional list of
configuration values plus the `MPI_Comm`. That list keeps growing (it is now
9 arguments) and existing call sites already rely on `/*comment=*/` markers
to stay readable. This change groups all *FMM-configuration* arguments into
a single struct `FmmConfig`, leaving only the MPI communicator as a true
constructor argument. The `MPI_Comm` stays out of the struct because it is
runtime/program context, not FMM behavior.

In addition, the current bounding-box tolerance is a single
`std::array<double, 3>` applied symmetrically to both the min and the max
side of each axis. Real workloads are often anisotropic at the box
boundaries (e.g. an inflow at `-x` but a free outflow at `+x`), so this
change splits that one factor into six independent per-face tolerances:
`xmin_tol, xmax_tol, ymin_tol, ymax_tol, zmin_tol, zmax_tol`. The
TreeBuilder code that currently does

```
_root_box.min[d] -= tol * width;
_root_box.max[d] += tol * width;
```

will be updated to pad each face independently.

## Files to modify

- [src/Canopy_Solver.hpp](src/Canopy_Solver.hpp) — define `FmmConfig`,
  rewrite `Solver` constructor and `createSolver` to take
  `(MPI_Comm, const FmmConfig&)`, forward the six face tolerances into
  `TreeBuilder`.
- [src/Canopy_TreeBuilder.hpp](src/Canopy_TreeBuilder.hpp) — replace
  `std::array<double, 3> _bb_tf` with a 6-element layout
  (`std::array<double, 6>` ordered `{xmin, xmax, ymin, ymax, zmin, zmax}`),
  update the constructor signature and the padding loop in `build()`
  (currently lines 608–620).
- [examples/02_full_fmm/example_full_fmm.cpp](examples/02_full_fmm/example_full_fmm.cpp)
  — populate an `FmmConfig` from the existing CLI args. Keep a single
  `bbox_tol` CLI option and broadcast it to all six faces (no change in
  user-facing behavior).
- [examples/03_gravity_solve/gravity_solve.cpp](examples/03_gravity_solve/gravity_solve.cpp)
  — same treatment as example 02.
- [tests/tstMultiSolve.hpp](tests/tstMultiSolve.hpp) — update both
  `Solver_t solver(...)` call sites (around lines 322, 763, 997) to build
  an `FmmConfig`.
- [README.md](README.md) — update the "Constructor" section and the
  "Minimal End-to-End Example" to show the `FmmConfig` API. Document the
  six face tolerances and that they default to a small symmetric value.

## FmmConfig shape

Lives in [src/Canopy_Solver.hpp](src/Canopy_Solver.hpp) above the `Solver`
class so callers do not need a second include.

```cpp
struct FmmConfig
{
    int    ncrit;
    int    max_depth;
    double xmin_tol = 0.0;
    double xmax_tol = 0.0;
    double ymin_tol = 0.0;
    double ymax_tol = 0.0;
    double zmin_tol = 0.0;
    double zmax_tol = 0.0;
    double ncrit_tol = 0.1;
    int    replication_depth = 1;
    double imbalance_tolerance = 0.05;
    double mac_theta = 0.5;
    double softening = -1.0; // negative = auto from inter-particle spacing
};
```

The two pieces with no useful default (`ncrit`, `max_depth`) are listed
first; everything else defaults to today's hard-coded defaults so existing
callers shrink rather than grow.

## TreeBuilder change

`TreeBuilder` keeps its current constructor *shape* but takes a
`std::array<double, 6>` ordered `{xmin, xmax, ymin, ymax, zmin, zmax}`
instead of a 3-element symmetric array. The padding loop becomes

```cpp
for ( int d = 0; d < 3; ++d )
{
    double width  = _root_box.max[d] - _root_box.min[d];
    double tmin   = _bb_tf[2 * d + 0];
    double tmax   = _bb_tf[2 * d + 1];
    if ( tmin > 0.0 ) _root_box.min[d] -= tmin * width;
    if ( tmax > 0.0 ) _root_box.max[d] += tmax * width;
}
```

No other TreeBuilder code reads `_bb_tf`, so this is the entire delta.

## Solver constructor

```cpp
Solver( MPI_Comm comm, const FmmConfig& cfg )
    : _comm( comm )
    , _replication_depth( cfg.replication_depth )
    , _builder( comm, cfg.ncrit, cfg.max_depth,
                std::array<double, 6>{ cfg.xmin_tol, cfg.xmax_tol,
                                       cfg.ymin_tol, cfg.ymax_tol,
                                       cfg.zmin_tol, cfg.zmax_tol },
                cfg.ncrit_tol )
    , _partitioner( comm, cfg.replication_depth, cfg.imbalance_tolerance )
    , _comm_plan( comm, cfg.mac_theta )
    , _upward( comm ), _downward( comm ), _p2p( comm )
    , _num_local( 0 )
    , _softening_input( cfg.softening )
    , _softening_initialized( false )
{
    if ( cfg.softening >= 0.0 )
    {
        _p2p.set_softening( static_cast<Scalar>( cfg.softening ) );
        _softening_initialized = true;
    }
}
```

`createSolver` updates correspondingly:
`createSolver(MPI_Comm comm, const FmmConfig& cfg)`.

## Call-site pattern

Every existing call site collapses to roughly:

```cpp
Canopy::FmmConfig cfg;
cfg.ncrit              = ncrit;
cfg.max_depth          = max_depth;
cfg.xmin_tol = cfg.xmax_tol = cfg.ymin_tol = cfg.ymax_tol
             = cfg.zmin_tol = cfg.zmax_tol = bbox_tol;
cfg.ncrit_tol          = ncrit_tol;
cfg.replication_depth  = replication_depth;
cfg.imbalance_tolerance = imbalance_tolerance;
cfg.mac_theta          = mac_theta;
cfg.softening          = 0.0;
Solver_t solver( MPI_COMM_WORLD, cfg );
```

The three test call sites and the two example call sites are mechanical.

## README update

In the "Constructor" section, replace the positional-argument table with an
`FmmConfig` struct listing all fields and their defaults, and update the
minimal example to build an `FmmConfig` and pass it. Mention that the six
bbox tolerances can be set independently per face but typically share the
same value.

## Verification

1. Build:
   `cd build-tuolumne && make -j Canopy_Test_MultiSolve_MPI_SERIAL` after
   activating the spack env per [systems/claude-tuolumne.md](systems/claude-tuolumne.md).
2. Run the minimum test set
   (`Canopy_Test_MultiSolve_MPI_SERIAL` at 1, 2, 3, 4, 5, 6 ranks) via
   `flux run` per `systems/claude-tuolumne.md` §4. All must pass.
3. Compile-build `make -j example_full_fmm gravity_solve` to confirm the
   examples still build. (Running them is optional — they have no
   pass/fail oracle in this refactor.)

## Checkpoint commits

- C1: introduce `FmmConfig`, switch `Solver` + `createSolver`, leave
  `TreeBuilder` taking the still-symmetric 3-element array (forward
  `{xmin,ymin,zmin}` for now, ignoring `*max_tol` temporarily). All tests
  must pass.
- C2: switch `TreeBuilder` to the 6-element array and the asymmetric
  padding loop. All tests must pass.
- C3: README update.
