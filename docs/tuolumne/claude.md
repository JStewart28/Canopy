# Tuolumne (LLNL) — Canopy build & run instructions

Tuolumne is an LLNL system with AMD MI300A APUs, Cray MPICH, and the flux
job scheduler. Compile and link via the Cray wrappers (`CC`/`cc`) plus
`amdclang++` for HIP code.

## 1. Spack environment

The build/run profile (CLAUDE.md "Build & run profile") records the env for this
checkout; the resolver
[scripts/lib/canopy_env.sh](../../scripts/lib/canopy_env.sh) activates it. The
committed manual-mode default (`CANOPY_SPACK_ENV` in
[scripts/tuolumne/profile.defaults.sh](../../scripts/tuolumne/profile.defaults.sh))
is:

```bash
source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos
```

Run that by hand for interactive work; the batch scripts activate through the
resolver.

**Manual mode** uses this env to build Canopy *by hand* (out-of-tree cmake +
make) — it provides Trilinos and the other build dependencies but does not
install Canopy itself; use [run_cmake_tuolumne.sh](../../run_cmake_tuolumne.sh)
(not the generic [run_cmake.sh](../../run_cmake.sh)) as the canonical cmake
invocation.

**Spack mode** (`spack develop canopy` + `spack install`) uses a different
dev/prod env. Set `CANOPY_SPACK_ENV` (dev) and optionally `CANOPY_SPACK_PROD_ENV`
(prod) in the gitignored `scripts/tuolumne/profile.local.sh`; confirm the env
names with the user.

## 2. CMake args

System-specific args that must be passed to `cmake`:

```
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
-DCMAKE_CXX_COMPILER=CC
-DCMAKE_C_COMPILER=cc
-DCMAKE_HIP_COMPILER=amdclang++
-DCMAKE_BUILD_TYPE=RelWithDebInfo
-DCanopy_ENABLE_TESTING=ON
-DCanopy_ENABLE_EXAMPLES=ON
-DCanopy_ENABLE_PROFILING=ON
-DCanopy_PROFILING_LEVEL=2
```

The wrapper script [run_cmake_tuolumne.sh](../../run_cmake_tuolumne.sh) is the
canonical source — invoke it from inside an out-of-tree build directory
(e.g. `build-tuolumne/`).

## 3. Build command

**Manual mode** (default) builds out-of-tree with `make`. From the build dir
(e.g. `build-tuolumne/`) after configuring with
[run_cmake_tuolumne.sh](../../run_cmake_tuolumne.sh):

```bash
make -j [TARGET]
```

The user specifies the target when appropriate (e.g.
`make -j Canopy_Test_MultiSolve_MPI_SERIAL`); plain `make -j` builds everything.
Binaries land in `$CANOPY_BUILD_DIR` (`build-tuolumne/tests/…`, `…/examples/…`).

**Spack mode** builds and installs via spack — binaries onto `PATH` as
`Canopy_Test_<name>_<DEVICE>`:

```bash
spack install canopy +rocm +testing +examples +profiling
```

Match the variants to the dev/prod env's spec (`+rocm` for the MI300A device
build). The spack recipe drives cmake; the `run_cmake_*` wrappers are
manual-mode only.

## 4. Run command for binaries

Tuolumne uses flux (not mpirun/srun). Binary location depends on the profile's
`CANOPY_BIN_MODE`: `$CANOPY_BUILD_DIR/tests/<exe>` (manual) or the bare on-PATH
name (spack); `canopy_exe <relpath|name>` from the resolver resolves either. The
basic template for an interactive allocation:

```bash
flux run --ntasks=[N] --nodes=1 --exclusive \
  --gpus-per-task=1 --cores-per-task=8 \
  --setopt=mpibind=verbose:1 \
  [EXECUTABLE] [EXTRA_ARGS]
```

For CPU-only / SERIAL-backend binaries (which is what the minimum test set
uses), drop `--gpus-per-task` and shrink `--cores-per-task` to match:

```bash
flux run --ntasks=[N] --nodes=1 --exclusive \
  --cores-per-task=1 \
  [EXECUTABLE] [EXTRA_ARGS]
```

Before any GPU run, export the Cray-MPICH GPU-aware-comm environment (also
needed for OpenMP):

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
```

**Required for every Canopy binary (CPU *and* GPU): the Cray static-TLS
workaround.** Canopy binaries link several `cray-libsci` libraries whose static
TLS blocks exhaust the dynamic loader's default surplus at startup, so the
binary aborts before `main` with:

```
libsci_cray_mp.so.6: cannot allocate memory in static TLS block
error while loading shared libraries: ... (exit 127)
```

Enlarge the glibc static-TLS surplus so the loader can place them:

```bash
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000
```

Set this in the same environment as the `flux run` (it must reach the launched
task). Do **not** use `LD_PRELOAD` of the libsci `.so` instead — it leaks into
child processes (e.g. `tail`, which then fails to find `libcraymp.so.1`).

## 5. Job-scheduler batch template

When not inside an interactive allocation, submit via `flux batch
<script>`. Use the template below as a starting point — fill in `JOB_NAME`,
`NODES`, `TIME_MIN`, `NTASKS`, the executable, and its args. Save concrete
filled-in scripts under [scripts/tuolumne/](../../scripts/tuolumne/).

```bash
#!/bin/bash
# flux: --job-name=[JOB_NAME]
# flux: --nodes=[NODES]
# flux: --exclusive
# flux: --time=[TIME_MIN]
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug

# Profile (env + build dir + binary location) via the resolver. Pin the repo
# root since the scheduler spools this script (its own path is unreliable).
CANOPY_REPO="${CANOPY_REPO:-/g/g20/stewartj/research-bridges/Canopy}"
source "${CANOPY_REPO}/scripts/lib/canopy_env.sh" || exit 1

export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# [EXECUTABLE] = $(canopy_exe tests/<name>) or $(canopy_exe examples/<dir>/<name>)
flux run --ntasks=[NTASKS] --nodes=[NODES] --exclusive \
  --gpus-per-task=1 --cores-per-task=8 \
  --setopt=mpibind=verbose:1 \
  [EXECUTABLE] [EXTRA_ARGS]
```

For the minimum test set (`Canopy_Test_MultiSolve_MPI_SERIAL` at 1–6
ranks), use the CPU/SERIAL variant of section 4 inside the batch script
rather than the HIP variant above.

### Preferred: drive the suite with CTest

When the build is configured with [run_cmake_tuolumne.sh](../../run_cmake_tuolumne.sh),
every unit test is registered with CTest at the required rank counts, and
`ctest` launches each one via `flux run --ntasks N --nodes=1 --exclusive
--cores-per-task=1` (the `MPIEXEC_*` overrides in that script). So the whole
minimum test set is one command inside an allocation:

```bash
ctest --output-on-failure -R 'Canopy_Test_MultiSolve_MPI_SERIAL'
```

[scripts/tuolumne/run_ctest_minset.flux](../../scripts/tuolumne/run_ctest_minset.flux)
is the batch wrapper for this — it sources the resolver
([scripts/lib/canopy_env.sh](../../scripts/lib/canopy_env.sh): env + profile),
exports the static-TLS workaround, and branches on `CANOPY_BIN_MODE`: in manual
mode it runs the `ctest` line above; in spack mode (no build tree) it launches
the on-PATH `Canopy_Test_MultiSolve_MPI_SERIAL` via `flux run` at ranks 1–6.
Submit with `flux batch run_ctest_minset.flux`. In manual mode, change the `-R`
regex to select a different suite (e.g. `TreePartitioner`) or drop it to run
everything; `ctest -N` lists what is registered without running anything.

**Important:** the `MPIEXEC_*` overrides are what make this work. If CTest is
left to auto-detect the launcher it picks the flux_wrappers `srun`, which runs
with no core binding and deadlocks at ≥3 ranks (the binaries bring up the HIP
backend at `Kokkos::initialize` even for SERIAL tests, and unbound ranks
contend on the single MI300A APU). Keep the overrides in the configure command.

[run_tests.flux](../../run_tests.flux) and
[run_profling.flux](../../run_profling.flux) at the repo root are the
working references this template was distilled from.

## 6. Running non-test binaries

When asked to run something other than a test (e.g. one of the
[examples/](../../examples/) problems), ask the user for the example name and
its args, then plug them into the section 4 `flux run` template or the
section 5 batch template depending on whether an interactive allocation is
already held.
