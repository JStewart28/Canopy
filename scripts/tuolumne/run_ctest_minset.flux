#!/bin/bash
# flux: --job-name=canopy-ctest-minset
# flux: --nodes=1
# flux: --exclusive
# flux: --time=15
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Required ship gate (CLAUDE.md) via CTest: every `regression`-labeled test on
# the SERIAL backend at ranks 1-6 (SingleSolve, MultiSolve). CTest already
# knows the tests and rank counts (Canopy_TEST_MPI_RANKS, default 1-6), so the
# whole gate collapses to a single `ctest -L regression -R MPI_SERIAL`
# invocation that fans out and prints its own pass/fail rollup. Submit with:
#   flux batch run_ctest_minset.flux
#
# To run the diagnostic suite instead, use `-L unit`; to run a single suite use
# `-R <name>` (e.g. TreePartitioner); drop the selectors to run everything.
# CTest launches each test through MPIEXEC_EXECUTABLE (flux run, per
# run_cmake_tuolumne.sh), which nests inside this flux allocation. SERIAL
# backend => CPU-only.

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build-tuolumne

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see docs/tuolumne/claude.md). Required for every
# Canopy binary on Tuolumne; must reach the launched task, so export it in this
# batch environment before invoking ctest.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

cd ${CANOPY_BUILD}

# --output-on-failure: dump a failing test's stdout/stderr inline.
# -L regression -R MPI_SERIAL: the required gate (SingleSolve + MultiSolve,
# SERIAL backend, ranks 1-6).
ctest --output-on-failure -L regression -R MPI_SERIAL
exit $?
