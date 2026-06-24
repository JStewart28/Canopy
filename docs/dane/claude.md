# Dane (LLNL) — Canopy build & run instructions

Dane is an LLNL CTS-2 system with Intel Sapphire Rapids CPUs (2×56 cores =
112 cores/node), MVAPICH2 as the default MPI, and the Slurm job scheduler.
Canopy builds CPU-only here (Kokkos OpenMP + Serial backends, no GPU).

## 0. Required modules (load FIRST, before anything else)

Dane's default compiler is Intel/gcc-10, and gcc 10 **cannot target the
Sapphire Rapids CPU** (`-march=sapphirerapids` fails — spack auto-detects that
arch). You must load gcc 13 before activating spack or building/running:

```bash
module load gcc/13.3.1 openmpi/4.1.2
```

- `gcc/13.3.1` — supports `-march=sapphirerapids`.
- `openmpi/4.1.2` — the spack Trilinos here was built against the external
  `openmpi@4.1.2` (gcc-13.3.1 build, `libmpi.so.40`). Loading it makes Canopy
  link the **same** MPI as Trilinos. If you skip it, the default `mvapich2`
  (`libmpi.so.12`) is used instead and the link emits
  `libmpi.so.40 ... may conflict with libmpi.so.12` — a real ABI mismatch that
  crashes at runtime, not a benign warning.

## 1. Spack environment

After the modules above, the build/run profile (CLAUDE.md "Build & run profile")
records the env; the resolver
[scripts/lib/canopy_env.sh](../../scripts/lib/canopy_env.sh) activates it. The
committed manual-mode default (`CANOPY_SPACK_ENV` in
[scripts/dane/profile.defaults.sh](../../scripts/dane/profile.defaults.sh)) is:

```bash
source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/dane_trilinos
```

Run that by hand for interactive work; the batch scripts activate through the
resolver (after `module load`, which must precede activation — see §0).

**Manual mode** builds Canopy *by hand* (out-of-tree cmake + make) with this
env; it provides Trilinos and the other build dependencies but does not install
Canopy itself. A snapshot of its `spack.yaml` is kept at [spack.yaml](spack.yaml).
**Spack mode** (`spack develop canopy` + `spack install`) uses a dev/prod env
instead; set `CANOPY_SPACK_ENV` (dev) and optionally `CANOPY_SPACK_PROD_ENV`
(prod) in the gitignored `scripts/dane/profile.local.sh`.

**Trilinos must be built `+openmp`.** Kokkos here is `+openmp +serial`, so
Kokkos' default execution space is OpenMP and Tpetra's default `Node` is the
OpenMP node. Trilinos must instantiate that node, or linking Canopy fails with
undefined references like
`Tpetra::Details::FixedHashTable<int, int, Kokkos::Device<Kokkos::OpenMP, ...>>`.
The `trilinos +openmp ^kokkos +openmp +serial` spec in the snapshot is what
fixes this — keep `+openmp` on both.

## 2. CMake args

System-specific args that must be passed to `cmake`:

```
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
-DCMAKE_CXX_COMPILER=mpicxx
-DCMAKE_BUILD_TYPE=RelWithDebInfo
-DCanopy_ENABLE_TESTING=ON
-DCanopy_ENABLE_EXAMPLES=ON
-DCanopy_ENABLE_PROFILING=ON
-DCanopy_PROFILING_LEVEL=0
```

`CMAKE_CXX_COMPILER=mpicxx` resolves (with the modules above) to the
openmpi/4.1.2 wrapper around g++ 13.3.1 — matching Trilinos' compiler and MPI
exactly. The wrapper script [run_cmake_dane.sh](../../run_cmake_dane.sh) is the
canonical source — invoke it from inside an out-of-tree build directory
(e.g. `build-dane/`):

```bash
cd build-dane && bash ../run_cmake_dane.sh
```

If you change MPI or compiler modules, wipe and reconfigure the build dir
(`rm -rf build-dane`) — a stale `CMakeCache.txt` keeps the old MPI/compiler.

## 3. Build command

**Manual mode** (default) builds out-of-tree with `make`. From the build dir
(e.g. `build-dane/`) after configuring with
[run_cmake_dane.sh](../../run_cmake_dane.sh):

```bash
make -j [TARGET]
```

The user specifies the target when appropriate (e.g.
`make -j Canopy_Test_MultiSolve_MPI_SERIAL`); plain `make -j` builds everything.
Binaries land in `$CANOPY_BUILD_DIR` (`build-dane/tests/…`).

**Spack mode** builds and installs via spack — binaries onto `PATH` as
`Canopy_Test_<name>_<DEVICE>`:

```bash
spack install canopy +openmp +testing +examples
```

`+openmp` matches the CPU OpenMP/Serial backend spec (see the Trilinos
`+openmp` note above). The spack recipe drives cmake; `run_cmake_dane.sh` is
manual-mode only.

## 4. Run command for binaries

Dane uses Slurm (`srun`). Binary location depends on the profile's
`CANOPY_BIN_MODE`: `$CANOPY_BUILD_DIR/tests/<exe>` (manual) or the bare on-PATH
name (spack); `canopy_exe <relpath|name>` from the resolver resolves either. A
Dane node has **112 physical cores** (2 sockets × 56). The OpenMP settings below
fill the node without oversubscribing — threads per rank = `112 / ranks_per_node`
— and bind threads to cores:

```bash
export OMP_NUM_THREADS=$(( 112 / N ))   # N = ranks on the node
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
srun -p pdebug -n [N] --cpus-per-task=${OMP_NUM_THREADS} -t [MIN] \
  [EXECUTABLE] [EXTRA_ARGS]
```

`--cpus-per-task` makes Slurm reserve matching cores per rank so the binding
lines up. `pdebug` (1 h limit, 38 nodes) is for short/interactive runs; use
`pbatch` (default, 1-day limit) for longer jobs. Small `srun` jobs can be
launched directly from a login node.

**Note:** the `*_MPI_SERIAL` tests use the Kokkos **Serial** backend, which
ignores `OMP_NUM_THREADS` — threading only affects OpenMP-backend runs and
examples. These env vars are harmless for the Serial tests, so set them
unconditionally.

(Do not copy Tuolumne's `OMP_NUM_THREADS=24` here — that is the per-APU core
count on Tuolumne; on Dane it idles cores at low rank counts and
oversubscribes at 5–6 ranks.)

## 5. Job-scheduler batch template

When not running interactively, submit via `sbatch <script>`. Fill in
`JOB_NAME`, `NODES`, `TIME`, `NTASKS`, the executable, and its args. Save
concrete filled-in scripts under [scripts/dane/](../../scripts/dane/) (create
the directory if it does not exist).

```bash
#!/bin/bash
#SBATCH --job-name=[JOB_NAME]
#SBATCH --nodes=[NODES]
#SBATCH --ntasks=[NTASKS]
#SBATCH --partition=pbatch
#SBATCH --time=[TIME]            # e.g. 00:30:00
#SBATCH --output=%x.%j.log

module load gcc/13.3.1 openmpi/4.1.2
# Profile (env + build dir + binary location) via the resolver. Pin the repo
# root since the scheduler spools this script. Modules load first (§0).
CANOPY_REPO="${CANOPY_REPO:-/g/g20/stewartj/research-bridges/Canopy}"
source "${CANOPY_REPO}/scripts/lib/canopy_env.sh" || exit 1

# Fill the node: 112 cores / ranks-per-node, bound to cores.
RANKS_PER_NODE=$(( SLURM_NTASKS / SLURM_NNODES ))
export OMP_NUM_THREADS=$(( 112 / RANKS_PER_NODE ))
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# [EXECUTABLE] = $(canopy_exe tests/<name>)  (absolute in manual mode, on PATH in spack mode)
srun --cpus-per-task=${OMP_NUM_THREADS} [EXECUTABLE] [EXTRA_ARGS]
```

For the minimum test set (`Canopy_Test_MultiSolve_MPI_SERIAL` at 1–6 ranks),
loop the `srun` line over `-n 1 … 6` (those tests use the Serial backend, so
`OMP_NUM_THREADS` is ignored — the affinity vars are still harmless to set).

### Preferred: drive the suite with CTest

Every unit test is registered with CTest at the required rank counts (1–6),
so the minimum test set is a single command inside an allocation:

```bash
ctest --output-on-failure -R 'Canopy_Test_MultiSolve_MPI_SERIAL'
```

[scripts/dane/run_ctest_minset.slurm](../../scripts/dane/run_ctest_minset.slurm)
is the batch wrapper — it loads the modules, sources the resolver
([scripts/lib/canopy_env.sh](../../scripts/lib/canopy_env.sh): env + profile),
and branches on `CANOPY_BIN_MODE`: in manual mode it runs the `ctest` line above;
in spack mode (no build tree) it launches the on-PATH
`Canopy_Test_MultiSolve_MPI_SERIAL` via `srun` at ranks 1–6. Submit with `sbatch
run_ctest_minset.slurm`. In manual mode, change the `-R` regex to select a
different suite or drop it to run everything; `ctest -N` lists what is registered
without running anything.

Unlike Tuolumne, Dane needs **no** `MPIEXEC_*` overrides: CMake auto-detects
`srun`, which is the native Slurm launcher and nests correctly inside an
`sbatch` allocation. CTest runs each test as `srun -n N <exe>`. (Tuolumne has
to override the launcher to `flux run` because its detected `srun` is a wrapper
that deadlocks unbound at ≥3 ranks; that does not apply here.) The CTest path
on Dane has not yet been validated by a run — confirm it before relying on it.

## 6. Running non-test binaries

When asked to run something other than a test (e.g. one of the
[examples/](../../examples/) problems), ask the user for the example name and
its args, then plug them into the section 4 `srun` template or the section 5
batch template depending on whether an allocation is already held.
