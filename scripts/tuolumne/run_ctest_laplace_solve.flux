#!/bin/bash
# flux: --job-name=canopy-laplace-solve
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T1's exit criterion: the Laplace-solve gate at ranks 1-6 on the SERIAL
# backend. Three tests per rank count, each a 12-timestep, 600-particle solve
# — bitForBitArtifacts (np 1-2), crossRankAgreement (np 2-6) and
# matchesDirectSum (np 1-6). CTest already knows the rank counts
# (Canopy_TEST_MPI_RANKS, default 1-6) and launches each test through
# MPIEXEC_EXECUTABLE (`flux run --ntasks N --nodes=1 --exclusive
# --cores-per-task=1`, per run_cmake_tuolumne.sh), which nests inside this
# allocation. Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_ctest_laplace_solve.flux)
#   flux job status "$jobid"
#
# Note: --flags=waitable is rejected here ("only the instance owner can
# submit with FLUX_JOB_WAITABLE"), so `flux job status` is the wait. Note
# too that the walltime option is --time-limit; a bare --time is rejected
# by this flux, which is why run_ctest_minset.flux's `# flux: --time=15`
# will not submit.
#
# CANOPY_LAPLACE_SOLVE_REGENERATE is deliberately NOT set here: this path
# always compares against the committed reference data and never writes it.

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne
CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see systems/tuolumne/claude.md). Required for
# every Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_laplace_solve.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank "[laplace-solve] ..." measurement lines reach this log
# even when every rank count passes: they carry n_unique_ops, the fallback
# count, and the cross-rank and direct-sum deviations the tolerances are
# pinned from.
ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL
exit $?
