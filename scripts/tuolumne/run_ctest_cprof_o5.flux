#!/bin/bash
# flux: --job-name=canopy-cprof-o5
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=40
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Copyright (c) 2025 by the Canopy authors
# SPDX-License-Identifier: BSD-3-Clause
#
# canopy_profiling O5 (beatnik tasks/canopy_profiling.md): the indexed
# communication-plan build and the auto_maintain timers. Copied from
# run_ctest_cprof_o8.flux.
#
#   1. The exit criterion: TreeBuilder, TreePartitioner, CommunicationPlan and
#      MultiSolve, SERIAL np 1-6 and HIP np 1-4, in build-profiling/ (level 2),
#      through canopy_ctest. CommunicationPlan runs -V so the log shows
#      CommunicationPlan.testIndexedMatchesLegacy* passing per rank.
#   2. The failure direction, once by hand: two SERIAL CommunicationPlan
#      binaries built from deliberately broken sources and never committed,
#      each run at np 1 and 2 on the comparison cases only.
#        .drop_child7    the index tables never link an octant-7 child; must
#                        FAIL at np 1 and 2.
#        .no_recv_dedup  the M2L receive buffer is neither sorted nor
#                        de-duplicated; no receives exist on one rank, so it
#                        must PASS at np 1 and FAIL at np 2.
#
# Usage (from the canopy checkout):
#   jobid=$(flux batch scripts/tuolumne/run_ctest_cprof_o5.flux)
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
echo "submit: flux batch scripts/tuolumne/run_ctest_cprof_o5.flux $*"
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
for exe in ${CANOPY_BUILD}/tests/Canopy_Test_{TreeBuilder,TreePartitioner,CommunicationPlan,MultiSolve}_MPI_{SERIAL,HIP} \
           ${CANOPY_BUILD}/tests/Canopy_Test_CommunicationPlan_MPI_SERIAL.{drop_child7,no_recv_dedup}; do
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
for stem in TreeBuilder TreePartitioner CommunicationPlan MultiSolve; do
    if [ ${stem} = CommunicationPlan ]; then verb=-V; else verb=; fi
    echo "### O5 ${stem} SERIAL np [1-6] ###"
    canopy_ctest "^Canopy_Test_${stem}_MPI_SERIAL_np_[1-6]\$" ${verb} --output-on-failure
    src=$?
    echo "### O5 ${stem} SERIAL: canopy_ctest rc=${src} ###"
    [ ${src} -eq 0 ] || rc=1

    echo "### O5 ${stem} HIP np [1-4] ###"
    ( hip_env; canopy_ctest "^Canopy_Test_${stem}_MPI_HIP_np_[1-4]\$" ${verb} --output-on-failure )
    src=$?
    echo "### O5 ${stem} HIP: canopy_ctest rc=${src} ###"
    [ ${src} -eq 0 ] || rc=1
done

# expect: "fail" (the case must fail) or "pass"
for spec in drop_child7:1:fail drop_child7:2:fail no_recv_dedup:1:pass no_recv_dedup:2:fail; do
    IFS=: read -r mut np expect <<< "${spec}"
    exe=${CANOPY_BUILD}/tests/Canopy_Test_CommunicationPlan_MPI_SERIAL.${mut}
    echo "### O5 broken-build check: ${mut} SERIAL np ${np} (expect ${expect}) ###"
    if [ ! -x "${exe}" ]; then
        echo "### O5 broken-build check: ${mut} np ${np}: SKIPPED, ${exe} missing ###"
        rc=1
        continue
    fi
    flux run --ntasks=${np} --nodes=1 --exclusive --cores-per-task=8 -t 5m \
        "${exe}" --gtest_filter='CommunicationPlan.testIndexedMatchesLegacy*' \
        > ${mut}_np${np}.out 2>&1
    mr=$?
    sed "s/^/[${mut} np${np}] /" ${mut}_np${np}.out | grep -v '^\[.*\] *$' | head -60
    nfail=$(grep -c '\[  FAILED  \]' ${mut}_np${np}.out)
    echo "### O5 broken-build check: ${mut} np ${np}: rc=${mr} failed_lines=${nfail} ###"
    if { [ ${expect} = fail ] && [ ${mr} -ne 0 ]; } || \
       { [ ${expect} = pass ] && [ ${mr} -eq 0 ]; }; then
        echo "### O5 broken-build check: ${mut} np ${np}: AS EXPECTED (${expect}) ###"
    else
        echo "### O5 broken-build check: ${mut} np ${np}: UNEXPECTED (expected ${expect}, rc=${mr}) ###"
        rc=1
    fi
done

watchdog_stop
echo "### overall rc=${rc} ###"
exit ${rc}
