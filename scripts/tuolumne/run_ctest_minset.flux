#!/bin/bash
# flux: --job-name=canopy-ctest-minset
# flux: --nodes=1
# flux: --exclusive
# flux: --time=15
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Required ship gate (CLAUDE.md): the `regression`-labeled MultiSolve solve on
# the SERIAL backend at ranks 1-6 (tests/CMakeLists.txt: REGRESSION_MPI_TESTS).
# Submit with:  flux batch run_ctest_minset.flux
#
# This script is build-mode aware (scripts/lib/canopy_env.sh):
#   - manual build (CANOPY_BIN_MODE=build-dir): drive the gate with one
#     `ctest -L regression -R MPI_SERIAL` in the build dir. CTest knows the tests
#     and rank counts (Canopy_TEST_MPI_RANKS, default 1-6) and launches each via
#     MPIEXEC_EXECUTABLE (flux run, per run_cmake_tuolumne.sh). Use `-L unit` for
#     the diagnostic suite or `-R <name>` for one suite.
#   - spack install (CANOPY_BIN_MODE=path): no build tree, so run the on-PATH
#     MultiSolve binary via flux at ranks 1-6 (what CTest would launch).
# SERIAL backend => CPU-only.

# Build/run profile: env, build dir, and binary location come from the shared
# resolver via scripts/tuolumne/profile.*.sh. The scheduler copies this batch
# script to a spool dir, so its own path is not reliable under `flux batch` —
# pin the repo root here (update if the checkout moves) and let the resolver
# activate spack + load the profile.
CANOPY_REPO="${CANOPY_REPO:-/g/g20/stewartj/research-bridges/Canopy}"
source "${CANOPY_REPO}/scripts/lib/canopy_env.sh" || exit 1

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see docs/tuolumne/claude.md). Required for every
# Canopy binary on Tuolumne; must reach the launched task, so export it in this
# batch environment before launching.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

if [ "${CANOPY_BIN_MODE}" = build-dir ]; then
  # --output-on-failure: dump a failing test's stdout/stderr inline.
  cd "${CANOPY_BUILD_DIR}"
  ctest --output-on-failure -L regression -R MPI_SERIAL
  exit $?
else
  # Path mode (spack install): enumerate the SERIAL regression binary that CTest
  # would run. Keep this in sync with the `regression` label if it changes.
  EXE="$(canopy_exe Canopy_Test_MultiSolve_MPI_SERIAL)"
  fail=0
  for N in 1 2 3 4 5 6; do
    echo "######## MultiSolve(SERIAL) np=${N} ########"
    flux run --ntasks=${N} --nodes=1 --exclusive --cores-per-task=1 "${EXE}" || fail=1
  done
  exit ${fail}
fi
