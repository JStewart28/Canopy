#!/bin/bash
#SBATCH --job-name=canopy-ctest-minset
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --partition=pdebug
#SBATCH --time=00:15:00
#SBATCH --output=%x.%j.log
#
# Required ship gate (CLAUDE.md) on Dane: the `regression`-labeled MultiSolve
# solve on the SERIAL backend at ranks 1-6 (tests/CMakeLists.txt:
# REGRESSION_MPI_TESTS). Submit with:  sbatch run_ctest_minset.slurm
#
# Build-mode aware (scripts/lib/canopy_env.sh):
#   - manual build (CANOPY_BIN_MODE=build-dir): one `ctest -L regression
#     -R MPI_SERIAL` in the build dir; CTest knows the tests and rank counts
#     (Canopy_TEST_MPI_RANKS, default 1-6) and launches each via srun, which
#     nests inside this sbatch allocation. Use `-L unit` for the diagnostic
#     suite or `-R <name>` for one suite.
#   - spack install (CANOPY_BIN_MODE=path): no build tree, so run the on-PATH
#     MultiSolve binary via srun at ranks 1-6.
# The allocation reserves 6 tasks so the np=6 case has cores to land on.

# Modules must load BEFORE spack activation (docs/dane/claude.md §0); the
# resolver does the spack activation + profile load.
module load gcc/13.3.1 openmpi/4.1.2

# Build/run profile: env, build dir, and binary location from the shared
# resolver (scripts/dane/profile.*.sh). Pin the repo root (the scheduler spools
# this script, so its own path is unreliable under `sbatch`).
CANOPY_REPO="${CANOPY_REPO:-/g/g20/stewartj/research-bridges/Canopy}"
source "${CANOPY_REPO}/scripts/lib/canopy_env.sh" || exit 1

# MultiSolve uses the Kokkos Serial backend, so OMP_NUM_THREADS is ignored; the
# affinity vars are harmless to set unconditionally (see docs/dane/claude.md).
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

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
    srun -n ${N} "${EXE}" || fail=1
  done
  exit ${fail}
fi
