#!/bin/bash
# flux: --job-name=canopy-v1
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# V1: re-derive the bounds the tree/key optimization chains are verified
# against. Stems CartesianTaylorSolve and MultiSolve at ranks 1-6, THREE
# passes in succession -- which is both V1's exit criterion and, before the
# bounds move, the measurement the new bounds are derived from. The tree and
# partition path is run-to-run nondeterministic at np >= 3 (risk R6), so one
# draw is not the number; three passes give the spread each bound's margin
# has to clear.
#
# -V so the [multisolve-dev], [fusedm2l-dev] and [ct-solve] deviation lines
# reach this log even when every case PASSES. That is the whole point: risk
# R10's distinguishing measurement is the measured deviation, not the
# pass/fail, and a passing EXPECT_LT prints nothing.
#
# CartesianTaylorSolve runs FIRST and both ctest invocations carry
# --timeout 300: MultiSolve_np_3 hangs intermittently until the scheduler
# wall kills it (README "Known Issues"), and in one allocation an unbounded
# hang there takes the other results with it. 300 s is ~20x the slowest rank
# count's observed runtime, so it bounds a hang without failing a slow pass.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_v1.flux)
#   flux job status "$jobid"; echo "status rc=$?"
#
# pdebug's policy limit is 1 h; a --time-limit above it is rejected at
# submit. Measured: one pass of both stems is ~2.3 min (job f3bmo4JYikKh ran
# all three in 424 s, ctest totals 58-78 s per stem), so three passes plus a
# 300 s watchdog-cancelled hang in every pass is ~22 min; 30 bounds that.
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
echo "submit: flux batch scripts/tuolumne/run_ctest_v1.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "build profiling config:"
grep -E '^Canopy_ENABLE_PROFILING:|^Canopy_PROFILING_LEVEL:' \
    ${CANOPY_BUILD}/CMakeCache.txt
echo "=================="

cd ${CANOPY_BUILD}

# WATCHDOG. ctest --timeout alone does NOT bound the np-3 hang on this
# system: on timeout ctest kills the `flux run` client, but the flux job it
# launched keeps running and keeps the node --exclusive, so every later rank
# count sits in state S behind it and times out too without ever starting
# (measured: flux job f3bn8EK66YaK, np 3 sub-job still R at 19.5 min while
# np 4/5/6 were S). This loop cancels any sub-job in this allocation that has
# run longer than WATCHDOG_S seconds, so the hang costs one rank count, not
# the rest of the pass. WATCHDOG_S matches the ctest --timeout below.
WATCHDOG_S=300
(
    while true; do
        flux jobs --filter=running --no-header -o '{id} {runtime}' |
            while read -r jid rt; do
                if [ "${rt%.*}" -gt ${WATCHDOG_S} ]; then
                    echo "### watchdog: cancelling sub-job ${jid} after ${rt%.*} s ###"
                    flux cancel "${jid}"
                fi
            done
        sleep 15
    done
) &
watchdog_pid=$!

rc_all=0
for i in 1 2 3; do
    echo "### pass ${i}: CartesianTaylorSolve ###"
    ctest -V --timeout 300 \
        -R '^Canopy_Test_CartesianTaylorSolve_MPI_SERIAL_np_[1-6]$'
    rc_cts=$?
    echo "### pass ${i}: MultiSolve ###"
    ctest -V --timeout 300 \
        -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$'
    rc_ms=$?
    echo "### pass ${i} exit codes: CartesianTaylorSolve=${rc_cts} MultiSolve=${rc_ms} ###"
    if [ ${rc_cts} -ne 0 ]; then rc_all=${rc_cts}; fi
    if [ ${rc_ms} -ne 0 ]; then rc_all=${rc_ms}; fi
done

kill ${watchdog_pid} 2>/dev/null
echo "### overall rc=${rc_all} ###"
exit ${rc_all}
