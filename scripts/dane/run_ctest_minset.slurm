#!/bin/bash
#SBATCH --job-name=canopy-ctest-minset
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --partition=pdebug
#SBATCH --time=00:15:00
#SBATCH --output=%x.%j.log
#
# Required ship gate (CLAUDE.md) via CTest on Dane: every `regression`-labeled
# test on the SERIAL backend at ranks 1-6 (SingleSolve, MultiSolve). CTest
# already knows the tests and rank counts (Canopy_TEST_MPI_RANKS, default 1-6),
# so the whole gate collapses to a single `ctest -L regression -R MPI_SERIAL`
# invocation that fans out and prints its own pass/fail rollup. Submit with:
#   sbatch run_ctest_minset.slurm
#
# To run the diagnostic suite instead, use `-L unit`; to run a single suite use
# `-R <name>` (e.g. TreePartitioner); drop the selectors to run everything.
# CTest launches each test through MPIEXEC_EXECUTABLE (srun), which nests inside
# this sbatch allocation. The allocation reserves 6 tasks so the np=6 case has
# cores to land on.

module load gcc/13.3.1 openmpi/4.1.2
source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/dane_trilinos

# MultiSolve uses the Kokkos Serial backend, so OMP_NUM_THREADS is ignored; the
# affinity vars are harmless to set unconditionally (see docs/dane/claude.md).
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build-dane
cd ${CANOPY_BUILD}

# --output-on-failure: dump a failing test's stdout/stderr inline.
# -L regression -R MPI_SERIAL: the required gate (SingleSolve + MultiSolve,
# SERIAL backend, ranks 1-6).
ctest --output-on-failure -L regression -R MPI_SERIAL
exit $?
