#!/bin/bash
# flux: --job-name=canopy-cartesian-taylor
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=8
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# The SERIAL unit target for the Cartesian-Taylor basis. T1's exit criterion;
# T2 and T3 reuse this script unchanged.
#
#   Canopy_Test_CartesianTaylor_SERIAL
#     index_map_bijection  slot()/inverse_slot() mutually inverse and onto
#                          [0, C(p+3,3)) at orders 0..6
#     closed_forms         the canopy-questions.md §3 recurrence against the
#                          four §2 closed-form tensors, |k| <= 3
#     finite_difference    Richardson-extrapolated central difference of phi,
#                          |k| = 4 .. 2p
#
# The -R regex is ANCHORED. Canopy_add_tests registers a
# Canopy_Test_CartesianTaylor_SERIAL_valgrind variant beside the real test
# because valgrind is found in build-tuolumne/
# (cmake/test_harness/test_harness.cmake:157-162). That variant is not a gate
# and the anchor is what excludes it.
#
# This target is non-MPI and NONMPI_PRECOMMAND is unset, so ctest runs the
# binary directly on the allocated node: no flux run wrapper, no --ntasks.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux)
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
echo "submit: flux batch scripts/tuolumne/run_ctest_cartesian_taylor_serial.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-body "[cartesian-taylor]" measurement lines -- the slot counts,
# the worst closed-form deviation and the worst finite-difference deviation
# with the multi-index and (r, b) they occurred at -- reach this log even when
# every body passes.
ctest -V -R '^Canopy_Test_CartesianTaylor_SERIAL$'
rc_ct=$?

echo "### exit code: CartesianTaylor=${rc_ct} ###"
exit ${rc_ct}
