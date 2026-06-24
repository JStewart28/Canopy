# Tuolumne (LLNL) — Canopy build & run instructions

Tuolumne is an LLNL system with AMD MI300A APUs, Cray MPICH, and the flux
job scheduler. Compile and link via the Cray wrappers (`CC`/`cc`) plus
`amdclang++` for HIP code.

## 1. Spack environment

Before any build or run command, activate the project's spack environment:

```bash
source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos
```

**Note:** this environment is for building Canopy *by hand* on tuolumne
(out-of-tree cmake + make) — it provides Trilinos and the other build
dependencies but does not install Canopy itself. When building by hand on
tuolumne, use [run_cmake_tuolumne.sh](../../run_cmake_tuolumne.sh) (not the
generic [run_cmake.sh](../../run_cmake.sh)) as the canonical cmake
invocation. If you are instead installing Canopy via
`spack install`, the environment to activate is different — confirm the
correct env name with the user before proceeding.

(The flux batch scripts in the repo root keep these as `SPACK_INSTALL` and
`CANOPY_ENV` variables — keep them in sync if those paths move.)

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

Tuolumne builds in-tree with `make`, not via `spack install`. After cmake
configuration:

```bash
make -j [TARGET]
```

The user specifies the target when appropriate (e.g.
`make -j Canopy_Test_MultiSolve_MPI_SERIAL`). For a full build, plain
`make -j` is fine.

## 4. Run command for binaries

Tuolumne uses flux (not mpirun/srun). The basic template for an
interactive allocation:

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

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build-tuolumne

export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

cd ${CANOPY_BUILD}

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
is the batch wrapper for this — it activates the env, exports the static-TLS
workaround, and runs the `ctest` line above. Submit with `flux batch
run_ctest_minset.flux`. Change the `-R` regex to select a different suite
(e.g. `TreePartitioner`) or drop it to run everything. `ctest -N` lists what is
registered without running anything.

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
