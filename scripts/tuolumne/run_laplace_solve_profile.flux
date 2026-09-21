#!/bin/bash
# flux: --job-name=canopy-laplace-solve-prof
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# R3 instrument for tasks/abstract-solver-backend.md T3 (and T4).
#
# The committed build tree build-tuolumne/ is configured with
# Canopy_ENABLE_PROFILING=OFF, so it prints no timing tables and cannot answer
# R3 ("a trait indirection deoptimizes the fused M2L kernel"). Reconfiguring it
# would put the bit-for-bit gate and the timing measurement in different build
# configurations across the before/after pair, so instead a SECOND build tree,
# build-tuolumne-prof/, is configured from the same run_cmake_tuolumne.sh with
# Canopy_ENABLE_PROFILING=ON and Canopy_PROFILING_LEVEL=2. The gate keeps
# build-tuolumne/ untouched; this script only ever runs build-tuolumne-prof/.
#
# np=1 only: R3 is about the per-team fused kernel, which is the same kernel at
# every rank count, and one rank removes MPI variance from the comparison.
# -V so the per-solve "[Canopy Diagnostics]" tables reach the log; the number
# the log records is "M2L kernel (all depths)" from the DownwardSweep::execute()
# table plus "Downward sweep" from the solve() breakdown.
#
# Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_laplace_solve_profile.flux)
#   flux job status "$jobid"

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne-prof
CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see systems/tuolumne/claude.md). Required for
# every Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_laplace_solve_profile.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

ctest -V -R 'Canopy_Test_LaplaceSolve_MPI_SERIAL_np_1$'
exit $?
