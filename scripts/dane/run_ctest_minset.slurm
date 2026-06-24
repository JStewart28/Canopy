#!/bin/bash
#SBATCH --job-name=canopy-ctest-minset
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --partition=pdebug
#SBATCH --time=00:15:00
#SBATCH --output=%x.%j.log
#
# Minimum test set (CLAUDE.md) via CTest on Dane. CTest already knows every
# test and the rank counts it must run at (Canopy_TEST_MPI_RANKS, default 1-6),
# so the whole minimum set collapses to a single `ctest -R` invocation that
# fans out and prints its own pass/fail rollup. Submit with:
#   sbatch run_ctest_minset.slurm
#
# To run a different suite, change the -R regex (e.g. TreePartitioner) or drop
# it to run everything registered. CTest launches each test through
# MPIEXEC_EXECUTABLE (srun), which nests inside this sbatch allocation. The
# allocation reserves 6 tasks so the np=6 case has cores to land on.

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
# -R: select the minimum test set (MultiSolve, SERIAL backend, ranks 1-6).
ctest --output-on-failure -R 'Canopy_Test_MultiSolve_MPI_SERIAL'
exit $?
