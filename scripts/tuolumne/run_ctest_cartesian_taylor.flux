#!/bin/bash
# flux: --job-name=canopy-cartesian-taylor-solve
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T4's exit criterion, in one job: BOTH halves of it.
#
#   Canopy_Test_CartesianTaylorSolve_MPI_SERIAL   np 1-6
#     matchesDirectSumThetaRef     mac_theta = 0.3, the reference treecode's
#                                  admissibility, gated at or below 1e-3
#     matchesDirectSumThetaCanopy  mac_theta = 0.5, Canopy's own default,
#                                  measured and pinned beside it
#
#   Canopy_Test_LaplaceSolve_MPI_SERIAL           np 1-6
#     bitForBitArtifacts (np 1-2), crossRankAgreement (np 2-6),
#     matchesDirectSum (np 1-6), opTableByteBudget
#
# The Laplace half is the guard on T4's Solver edits: this basis adds a header
# and changes no shared arithmetic, so that suite must stay green with NO
# regeneration of tests/data/laplace_solve_P6.txt. CANOPY_LAPLACE_SOLVE_REGENERATE
# is deliberately NOT set here; this path always compares and never writes.
#
# CTest already knows the rank counts (Canopy_TEST_MPI_RANKS, default 1-6) and
# launches each test through MPIEXEC_EXECUTABLE (`flux run --ntasks N
# --nodes=1 --exclusive --cores-per-task=1`, per run_cmake_tuolumne.sh:10-12),
# which nests inside this allocation. These are SERIAL (host) targets: there
# is no rank-to-GPU binding to add and no launcher to add here. Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_ctest_cartesian_taylor.flux)
#   flux job status "$jobid"
#
# --time-limit=20 rather than the 8 the SERIAL-only scripts use: this job runs
# two suites. Note --flags=waitable is rejected here ("only the instance owner
# can submit with FLUX_JOB_WAITABLE"), so `flux job status` is the wait, and
# never `flux job attach` — it forwards signals and cancels the job on two
# SIGINTs. Note too that the walltime option is --time-limit; a bare --time is
# rejected by this flux.

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
echo "submit: flux batch scripts/tuolumne/run_ctest_cartesian_taylor.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank "[ct-solve] ..." and "[laplace-solve] ..." measurement
# lines reach this log even when every rank count passes: they carry
# n_unique_ops, the fallback count, the effective operator cap and the
# direct-sum deviations the tolerances are pinned from.
ctest -V -R Canopy_Test_CartesianTaylorSolve_MPI_SERIAL
rc_cts=$?

ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL
rc_ls=$?

echo "### exit codes: CartesianTaylorSolve=${rc_cts} LaplaceSolve=${rc_ls} ###"
if [ ${rc_cts} -ne 0 ]; then exit ${rc_cts}; fi
exit ${rc_ls}
