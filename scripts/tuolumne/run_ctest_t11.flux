#!/bin/bash
# flux: --job-name=canopy-t11
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=8
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T11's exit criterion, both runnable suites in one allocation:
#
#   LaplaceSolve     the full gate — bit-for-bit at np 1-2, crossRankAgreement
#                    at 2-6, matchesDirectSum at 1-6. T11 adds a defaulted
#                    template template parameter and changes no arithmetic, so
#                    the default instantiation is the same LaplaceKernel it was
#                    and the committed bytes must match with no regeneration.
#   FarFieldContract carries the new compile-only test: Solver named on
#                    MonopoleBasis, forced complete with sizeof so the sweeps'
#                    class-scope guards run against it. That it linked is the
#                    proof; this run is here to show the four existing bodies
#                    did not move.
#
# The other three T11 targets — Canopy_Test_MultiSolve_MPI_SERIAL, example_fmm
# and gravity_solve — are built to COMPILE and are not run (Deliberate
# deviations in tasks/abstract-solver-backend.md).
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_t11.flux)
#   flux job status "$jobid"
#
# Note: --flags=waitable is rejected here ("only the instance owner can submit
# with FLUX_JOB_WAITABLE"), so `flux job status` is the wait. The walltime
# option is --time-limit; a bare --time is rejected by this flux.

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
echo "submit: flux batch scripts/tuolumne/run_ctest_t11.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank "[laplace-solve]" and "[far-field-contract]" measurement
# lines reach this log even when every rank count passes. They carry
# n_unique_ops per rank, fallback_pairs, the extents, the slot-coverage totals
# and the cross-rank and direct-sum deviations.
echo "### LaplaceSolve ###"
ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL
rc_ls=$?

echo "### FarFieldContract ###"
ctest -V -R Canopy_Test_FarFieldContract_MPI_SERIAL
rc_ffc=$?

echo "### exit codes: LaplaceSolve=${rc_ls} FarFieldContract=${rc_ffc} ###"
if [ ${rc_ls} -ne 0 ]; then exit ${rc_ls}; fi
exit ${rc_ffc}
