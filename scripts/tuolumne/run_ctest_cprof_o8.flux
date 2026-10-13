#!/bin/bash
# flux: --job-name=canopy-cprof-o8
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=40
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Copyright (c) 2025 by the Canopy authors
# SPDX-License-Identifier: BSD-3-Clause
#
# canopy_profiling O8 (beatnik tasks/canopy_profiling.md): device assembly of
# the M2L operator table. Preamble from run_ctest_cprof_t1.flux, HIP subshell
# from run_ctest_t1_hip.flux.
#
#   1. The exit criterion: DownwardSweep and CartesianTaylorSolve, SERIAL
#      np 1-6 and HIP np 1-4, in build-profiling/ (level 2), through
#      canopy_ctest. DownwardSweep runs -V so the log shows
#      DownwardSweepCaching.deviceOpTableAssemblyBitwise passing per rank.
#   2. The stale-column check, once by hand: two HIP DownwardSweep binaries
#      built from deliberately broken sources and never committed, each run
#      at np 1 and 2 on the bitwise case only. Both must FAIL.
#        .stale_clear  clear_m2l_op_cache() leaves the device copy in place;
#        .stale_grow   a growth of the device copy drops its uploaded prefix.
#
# Usage (from the canopy checkout):
#   jobid=$(flux batch scripts/tuolumne/run_ctest_cprof_o8.flux)
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
echo "submit: flux batch scripts/tuolumne/run_ctest_cprof_o8.flux $*"
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
for exe in ${CANOPY_BUILD}/tests/Canopy_Test_{DownwardSweep,CartesianTaylorSolve}_MPI_{SERIAL,HIP} \
           ${CANOPY_BUILD}/tests/Canopy_Test_DownwardSweep_MPI_HIP.stale_{clear,grow}; do
    [ -x "${exe}" ] && echo "binary: $(basename ${exe}) $(stat -c '%y' ${exe})"
done
echo "=================="

cd ${CANOPY_BUILD}

WATCHDOG_S=300
CANOPY_WATCHDOG_PGREP=Canopy_Test_
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
source ${CANOPY_SRC}/scripts/tuolumne/ctest_budget.sh
watchdog_wait_idle || exit 3

hip_env() {
    export MPICH_GPU_SUPPORT_ENABLED=1 GTL_HSA_VSMSG_CUTOFF_SIZE=4096 \
        FI_CXI_ATS=0 HSA_XNACK=1 MPICH_SMP_SINGLE_COPY_MODE=NONE
}

rc=0
for stem in DownwardSweep CartesianTaylorSolve; do
    if [ ${stem} = DownwardSweep ]; then verb=-V; else verb=; fi
    echo "### O8 ${stem} SERIAL np [1-6] ###"
    canopy_ctest "^Canopy_Test_${stem}_MPI_SERIAL_np_[1-6]\$" ${verb} --output-on-failure
    src=$?
    echo "### O8 ${stem} SERIAL: canopy_ctest rc=${src} ###"
    [ ${src} -eq 0 ] || rc=1

    echo "### O8 ${stem} HIP np [1-4] ###"
    ( hip_env; canopy_ctest "^Canopy_Test_${stem}_MPI_HIP_np_[1-4]\$" ${verb} --output-on-failure )
    src=$?
    echo "### O8 ${stem} HIP: canopy_ctest rc=${src} ###"
    [ ${src} -eq 0 ] || rc=1
done

for mut in stale_clear stale_grow; do
    exe=${CANOPY_BUILD}/tests/Canopy_Test_DownwardSweep_MPI_HIP.${mut}
    for np in 1 2; do
        echo "### O8 stale check: ${mut} HIP np ${np} ###"
        if [ ! -x "${exe}" ]; then
            echo "### O8 stale check: ${mut} np ${np}: SKIPPED, ${exe} missing ###"
            rc=1
            continue
        fi
        (
            hip_env
            flux run --ntasks=${np} --nodes=1 --exclusive --gpus-per-task=1 \
                --cores-per-task=8 -t 5m "${exe}" \
                --gtest_filter='DownwardSweepCaching.deviceOpTableAssemblyBitwise'
        ) > ${mut}_np${np}.out 2>&1
        mr=$?
        sed "s/^/[${mut} np${np}] /" ${mut}_np${np}.out
        nmis=$(grep -c 'countOpTableMismatches' ${mut}_np${np}.out)
        nthrow=$(grep -c 'out of step with the host cache' ${mut}_np${np}.out)
        echo "### O8 stale check: ${mut} np ${np}: rc=${mr} bitwise_failures=${nmis} logic_error_lines=${nthrow} ###"
        if [ ${mr} -ne 0 ]; then
            echo "### O8 stale check: ${mut} np ${np}: PASS (the case failed) ###"
        else
            echo "### O8 stale check: ${mut} np ${np}: FAIL (the broken binary passed) ###"
            rc=1
        fi
    done
done

watchdog_stop
echo "### overall rc=${rc} ###"
exit ${rc}
