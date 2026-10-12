#!/bin/bash
# flux: --job-name=canopy-cprof-t1
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=40
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Copyright (c) 2025 by the Canopy authors
# SPDX-License-Identifier: BSD-3-Clause
#
# canopy_profiling T1 (beatnik tasks/canopy_profiling.md): the run-time
# reporting switch. Copied from run_ctest_c11.flux's gate mode.
#
#   1. The exit criterion: LaplaceSolve and CartesianTaylorSolve, SERIAL,
#      np 1-6, in build-profiling/ (level 2), through canopy_ctest with -V so
#      the log shows that the existing cases still print [Canopy Diagnostics]
#      and that LaplaceSolve.reportingSwitch's captured output had none.
#   2. The failure direction, once: a LaplaceSolve binary built with the gate
#      in print_solve_breakdown deleted (tests/...SERIAL.nogate, built by hand
#      and never committed), np 1, reportingSwitch only. It must FAIL, with a
#      "solve() phase breakdown" block in its captured stdout.
#
# Usage (from the canopy checkout):
#   jobid=$(flux batch scripts/tuolumne/run_ctest_cprof_t1.flux)
#   flux job status "$jobid"; echo "status rc=$?"

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_SRC=/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy
CANOPY_BUILD=${CANOPY_SRC}/build-profiling

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see systems/tuolumne/claude.md). Required for
# every Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

unset CANOPY_MULTISOLVE_PROBE CANOPY_MULTISOLVE_NPP CANOPY_MAC_THETA CANOPY_BUDGET_CONFIG
unset CANOPY_LAPLACE_SOLVE_REGENERATE

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_cprof_t1.flux $*"
echo "jobid: $(flux job id --to=f58 "$(flux getattr jobid 2>/dev/null)" 2>/dev/null || echo unknown)"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "build: ${CANOPY_BUILD}"
echo "build profiling config:"
grep -E '^Canopy_ENABLE_PROFILING:|^Canopy_PROFILING_LEVEL:' \
    ${CANOPY_BUILD}/CMakeCache.txt
echo "flux: $(flux version | head -1)"
for exe in ${CANOPY_BUILD}/tests/Canopy_Test_{LaplaceSolve,CartesianTaylorSolve}_MPI_SERIAL \
           ${CANOPY_BUILD}/tests/Canopy_Test_LaplaceSolve_MPI_SERIAL.nogate; do
    [ -x "${exe}" ] && echo "binary: $(basename ${exe}) $(stat -c '%y' ${exe})"
done
echo "=================="

cd ${CANOPY_BUILD}

WATCHDOG_S=300
CANOPY_WATCHDOG_PGREP=Canopy_Test_
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
source ${CANOPY_SRC}/scripts/tuolumne/ctest_budget.sh
watchdog_wait_idle || exit 3

rc=0
for stem in LaplaceSolve CartesianTaylorSolve; do
    echo "### T1 ${stem} SERIAL np [1-6] ###"
    canopy_ctest "^Canopy_Test_${stem}_MPI_SERIAL_np_[1-6]\$" -V --output-on-failure
    src=$?
    echo "### T1 ${stem} SERIAL: canopy_ctest rc=${src} ###"
    [ ${src} -eq 0 ] || rc=1
done

echo "### T1 failure direction: gate deleted, np 1, reportingSwitch only ###"
nogate=${CANOPY_BUILD}/tests/Canopy_Test_LaplaceSolve_MPI_SERIAL.nogate
if [ -x "${nogate}" ]; then
    flux run --ntasks=1 --nodes=1 --exclusive --cores-per-task=1 -t 5m \
        "${nogate}" --gtest_filter='LaplaceSolve.reportingSwitch' \
        > nogate.out 2>&1
    ng=$?
    sed 's/^/[nogate] /' nogate.out
    nb=$(grep -c 'solve() phase breakdown' nogate.out)
    echo "### T1 failure direction: rc=${ng} breakdown_blocks_in_failure_output=${nb} ###"
    if [ ${ng} -ne 0 ] && [ ${nb} -gt 0 ]; then
        echo "### T1 failure direction: PASS (the case failed and named the block) ###"
    else
        echo "### T1 failure direction: FAIL (expected a failing case with a breakdown block) ###"
        rc=1
    fi
else
    echo "### T1 failure direction: SKIPPED, ${nogate} missing ###"
    rc=1
fi

watchdog_stop
echo "### overall rc=${rc} ###"
exit ${rc}
