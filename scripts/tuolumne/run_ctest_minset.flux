#!/bin/bash
# flux: --job-name=canopy-ctest-minset
# flux: --nodes=1
# flux: --exclusive
# flux: --time=15
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Minimum test set (CLAUDE.md) via CTest. Replaces the hand-rolled per-rank
# loop in run_multisolve_minset.flux: CTest already knows every test and the
# rank counts it must run at (Canopy_TEST_MPI_RANKS, default 1-6), so the whole
# minimum set collapses to a single `ctest -R` invocation that fans out and
# prints its own pass/fail rollup. Submit with:
#   flux batch run_ctest_minset.flux
#
# To run a different suite, change the -R regex (e.g. TreePartitioner) or drop
# it to run everything registered. CTest launches each test through
# MPIEXEC_EXECUTABLE (the flux_wrappers srun), which nests inside this flux
# allocation. These are SERIAL-backend tests, so they run CPU-only.

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build-tuolumne

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see docs/tuolumne/claude.md). Required for every
# Canopy binary on Tuolumne; must reach the srun-launched task, so export it in
# this batch environment before invoking ctest.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

cd ${CANOPY_BUILD}

# --output-on-failure: dump a failing test's stdout/stderr inline.
# -R: select the minimum test set (MultiSolve, SERIAL backend, ranks 1-6).
ctest --output-on-failure -R 'Canopy_Test_MultiSolve_MPI_SERIAL'
exit $?
