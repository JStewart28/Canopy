#!/bin/bash
# flux: --job-name=canopy-t10
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=8
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T10's exit criterion, all three suites in one allocation:
#
#   LaplaceSolve     the full gate — bit-for-bit at np 1-2, crossRankAgreement
#                    at 2-6, matchesDirectSum at 1-6. LaplaceKernel declares
#                    sets_per_component = 1, at which the new slot flattening
#                    c * sets_per_component + s collapses to c exactly, so the
#                    committed bytes must match with no regeneration.
#   FarFieldContract MonopoleBasis is now at sets_per_component = 2. Both sets
#                    are compared against the host reference, the two are
#                    required to differ, and the shared-cell slot map is
#                    required to be a bijection onto the full slot range by
#                    all three of its loops (R6, T10 step 5).
#   DownwardSweep    reads downward.locals() directly at four sites and is the
#                    one consumer this task's reshape can break. Known green
#                    at ranks 1-6 as of T9, so a failure here is attributable.
#
# One allocation deliberately, so "the Laplace gate did not move" is checkable
# from the same log as the thing that might have moved it.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_t10.flux)
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
echo "submit: flux batch scripts/tuolumne/run_ctest_t10.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank "[laplace-solve]", "[far-field-contract]" and
# "[downward-caching]" measurement lines reach this log even when every rank
# count passes. They carry n_unique_ops per rank, fallback_pairs, the extents,
# the cross-rank and direct-sum deviations, and the operator-cache counters.
echo "### LaplaceSolve ###"
ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL
rc_ls=$?

echo "### FarFieldContract ###"
ctest -V -R Canopy_Test_FarFieldContract_MPI_SERIAL
rc_ffc=$?

echo "### DownwardSweep ###"
ctest -V -R Canopy_Test_DownwardSweep_MPI_SERIAL
rc_ds=$?

echo "### exit codes: LaplaceSolve=${rc_ls} FarFieldContract=${rc_ffc} DownwardSweep=${rc_ds} ###"
if [ ${rc_ls} -ne 0 ]; then exit ${rc_ls}; fi
if [ ${rc_ffc} -ne 0 ]; then exit ${rc_ffc}; fi
exit ${rc_ds}
