# Canopy

## Using Canopy for a Far-Field Solve

Canopy provides a parallel Fast Multipole Method (FMM) solver built on top of Kokkos and Cabana. The primary entry point is `Canopy::Solver`, defined in `src/Canopy_Solver.hpp`.

### Template Parameters

```cpp
Canopy::Solver<MemorySpace, ExecutionSpace, Scalar, P_ORDER, NComps>
```

| Parameter | Description | Default |
|---|---|---|
| `MemorySpace` | Kokkos memory space (e.g. `Kokkos::HostSpace`, `Kokkos::CudaSpace`) | — |
| `ExecutionSpace` | Kokkos execution space (e.g. `Kokkos::OpenMP`, `Kokkos::Cuda`) | — |
| `Scalar` | Floating-point type for field values | `double` |
| `P_ORDER` | Multipole expansion order (higher = more accurate, more expensive) | `8` |
| `NComps` | Number of simultaneous charge components | `1` |

### Constructor

```cpp
Canopy::Solver<...> solver(
    MPI_Comm comm,           // MPI communicator
    int      ncrit,          // max particles per leaf cell
    int      max_depth,      // maximum octree depth
    double   tree_tolerance, // admissibility criterion for M2L interactions
    int      replication_depth, // depth to which ghost cells are replicated
    double   imbalance_tolerance = 0.05 // load-balance threshold
);
```

### Typical Usage Pattern

Particles are stored in a Cabana `AoSoA`. Two slices are required: one for 3D positions and one for scalar charges. The template integer arguments `PositionIdx` and `ChargeIdx` identify which AoSoA field holds each quantity.

**1. One-time setup** (call once before the first solve):

```cpp
// particles: Cabana AoSoA with positions at field 0, charges at field 1
// num_local: number of locally-owned particles before any migration
solver.setup<0, 1>(particles, num_local);
```

`setup()` builds the octree, partitions it across MPI ranks, migrates particles to their owning ranks, and prepares all FMM communication structures. After this call, `solver.num_local_particles()` reflects the post-migration local count.

**2. Solve** (call once per timestep):

```cpp
bool compute_gradient = true; // also evaluate gradient of the potential
solver.solve<0, 1>(particles, compute_gradient);

// Retrieve results (Kokkos views, indexed by local particle)
auto phi  = solver.potential(); // shape: (num_local,)
auto grad = solver.gradient();  // shape: (num_local, 3) — only valid if compute_gradient == true
```

`solve()` runs the full FMM pipeline: P2M → M2M → M2L → L2L → L2P → P2P. Output views are zeroed before each call, so results are not accumulated across timesteps.

**3. Between timesteps — maintaining the solver**

After particles move, the solver must be updated before the next `solve()`. Three options are available, in increasing cost:

| Method | When to use |
|---|---|
| `solver.migrate<0>(particles)` | Particles moved but the global octree cell structure is unchanged |
| `solver.rebalance<0>(particles)` | Tree topology changed (cells refined/coarsened) but all particles remain inside the original bounding box |
| `solver.rebuild<0, 1>(particles)` | Particles escaped the bounding box — full reconstruction required |

If you are unsure which applies, use `auto_maintain()`, which selects the cheapest valid path automatically and returns a `MaintenanceAction` enum indicating which path was taken:

```cpp
auto action = solver.auto_maintain<0, 1>(particles);
// action is one of: MaintenanceAction::Migrate, Rebalance, or Rebuild
```

### Minimal End-to-End Example

```cpp
using MemSpace = Kokkos::HostSpace;
using ExecSpace = Kokkos::Serial;

Canopy::Solver<MemSpace, ExecSpace> solver(MPI_COMM_WORLD,
    /*ncrit=*/32, /*max_depth=*/10,
    /*tree_tolerance=*/0.4, /*replication_depth=*/1);

// ... populate particles AoSoA ...

solver.setup<0, 1>(particles, num_local);

for (int step = 0; step < nsteps; ++step)
{
    solver.solve<0, 1>(particles, /*compute_gradient=*/false);

    auto phi = solver.potential(); // use phi in your time integrator

    // ... advance particle positions ...

    solver.auto_maintain<0, 1>(particles);
}
```

---

### Resources used:
1. [Fast multipole info](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Refs/2012_fmm_encyclopedia.pdf)

2. [Lecture 2](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Lectures/lecture02.pdf)

3. [FMM for Vortical Flows](https://repositorio.unesp.br/server/api/core/bitstreams/0e824479-3128-41f7-8cd2-462e9a242c42/content)

4. [1987_greengard_dissertation](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Refs/1987_greengard_dissertation.pdf)

5. [CSCAMM Lecture](https://home.cscamm.umd.edu/programs/fam04/dg_lecture6.pdf)

6. [1999_cheng](https://www.sciencedirect.com/science/article/pii/S0021999199963556)

7. [Rankin, WT: Efficient parallel implementations of multipole based N-body algorithms](https://www.proquest.com/dissertations-theses/efficient-parallel-implementations-multipole/docview/304504480/se-2?accountid=14613)

8. [ExaFmm] (https://github.com/exafmm/exafmm)
