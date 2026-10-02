#!/bin/bash
# flux: --job-name=canopy-t1
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T1's exit criterion: stems MultiSolve and DownwardSweep at ranks 1-6, and
# nothing else. MultiSolve carries the per-reason report on the existing
# clustered fixture; DownwardSweep carries the new two-scale fixture and its
# two cases.
#
# -V so the [m2l-fallback-reason], [two-scale] and [two-scale-refusals]
# measurement lines reach this log even when every rank count passes. Run it
# TWICE and diff those lines: the tree and partition path is run-to-run
# nondeterministic at np >= 3 (risk R6), so one run is not the number.
#
# KNOWN PRE-EXISTING FAILURE: six MultiSolve tests fail the 1e-8 multi-step
# position/velocity check at every rank count, with errors of 3e-7 to 9e-6.
# See README "Known Issues"; it predates this work and T1 does not touch that
# bound. The DownwardSweep arm is the one T1 adds to.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_t1_step1.flux)
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
echo "submit: flux batch scripts/tuolumne/run_ctest_t1_step1.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# DownwardSweep first, MultiSolve second, and --timeout 300 on both. Not a
# narrowing of the criterion -- the same two stems at the same ranks 1-6 --
# but MultiSolve_np_3 hangs intermittently until the scheduler wall kills it
# (README "Known Issues"), and in one allocation an unbounded hang there takes
# the DownwardSweep results with it. 300 s is ~20x the slowest rank count's
# observed runtime, so it bounds a hang without failing a slow pass.
# DownwardSweep TWICE, back to back: the tree and partition path is
# run-to-run nondeterministic at np >= 3 (risk R6), so the per-(nprocs, rank)
# counters have to be reported from two separate runs with whether they agree
# stated, never as a single draw and never as a mean over ranks.
echo "### DownwardSweep run 1 ###"
ctest -V --timeout 300 -R '^Canopy_Test_DownwardSweep_MPI_SERIAL_np_[1-6]$'
rc_ds=$?

echo "### DownwardSweep run 2 ###"
ctest -V --timeout 300 -R '^Canopy_Test_DownwardSweep_MPI_SERIAL_np_[1-6]$'
rc_ds2=$?

echo "### MultiSolve ###"
ctest -V --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$'
rc_ms=$?

echo "### exit codes: DownwardSweep=${rc_ds} DownwardSweep2=${rc_ds2} MultiSolve=${rc_ms} ###"
if [ ${rc_ds} -ne 0 ]; then exit ${rc_ds}; fi
if [ ${rc_ds2} -ne 0 ]; then exit ${rc_ds2}; fi
exit ${rc_ms}
