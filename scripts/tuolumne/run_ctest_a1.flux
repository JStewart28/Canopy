#!/bin/bash
# flux: --job-name=canopy-a1
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=8
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# tree-opt A1: touching-leaf level differences and the balance cost model
# ([a1-balance]) on the two-scale and graded draws, beside the stem's existing
# lines. Every run goes through canopy_ctest after H0b's watchdog self-test. A
# hip run gets the HIP environment in a subshell, as in fix-hang-rebalance
# H0c/E1. Copied from run_ctest_b0.flux; the noprof mode is dropped.
#   measure <backend>  build-tuolumne (profiling ON): two -V passes of
#                      DownwardSweep, at SERIAL np 1-6 or HIP np 1-4.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_a1.flux measure hip)
#   flux job status "$jobid"; echo "status rc=$?"
#
# --flags=waitable is rejected here; `flux job status` is the wait.

mode=$1
backend=${2:-serial}
case ${mode} in
    measure) ;;
    *) echo "usage: flux batch run_ctest_a1.flux measure serial|hip"; exit 2 ;;
esac
case ${backend} in serial|hip) ;; *) echo "backend: serial|hip"; exit 2 ;; esac

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy
CANOPY_BUILD=${CANOPY_SRC}/build-tuolumne

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see systems/tuolumne/claude.md). Required for
# every Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

unset CANOPY_MULTISOLVE_PROBE CANOPY_MULTISOLVE_NPP CANOPY_MAC_THETA CANOPY_BUDGET_CONFIG

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_a1.flux $*"
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
if [ "${backend}" = hip ]; then exe_dev=HIP; else exe_dev=SERIAL; fi
for exe in ${CANOPY_BUILD}/tests/Canopy_Test_DownwardSweep_MPI_${exe_dev}; do
    [ -x "${exe}" ] && echo "binary: $(basename ${exe}) $(stat -c '%y' ${exe})"
done
echo "=================="

cd ${CANOPY_BUILD}

n56=$(ctest -N -R '_MPI_HIP_np_[56]$' 2>/dev/null | grep -c 'Test *#')
if [ "${n56}" -ne 0 ]; then
    echo "### ABORT: ${n56} HIP entries registered at np 5-6; reconfigure per H0a ###"
    exit 5
fi

echo "### self-test ###"
WATCHDOG_S=20
CANOPY_WATCHDOG_PGREP=sleep
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
flux run --ntasks=3 --nodes=1 --exclusive --cores-per-task=1 sleep 900
watchdog_wait_idle || exit 3
watchdog_stop
echo "self-test cancel record: $(cat ${WATCHDOG_DIR}/cancelled)"
read -r st_jid st_rt st_shell st_matched st_children st_nonempty \
    < ${WATCHDOG_DIR}/cancelled
st_rt=${st_rt#runtime=}
if [ "$(watchdog_cancel_count)" -ne 1 ] ||
    ! awk -v r="${st_rt}" 'BEGIN { exit !(r >= 20 && r <= 25) }' ||
    [ "${st_matched} ${st_children} ${st_nonempty}" != "matched=3 children=3 nonempty=3" ]; then
    echo "### self-test FAILED; nothing else run ###"
    exit 2
fi
echo "### self-test PASSED (cancelled at ${st_rt} s) ###"

WATCHDOG_S=300
CANOPY_WATCHDOG_PGREP=Canopy_Test_
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
source ${CANOPY_SRC}/scripts/tuolumne/ctest_budget.sh
watchdog_wait_idle || exit 3

if [ "${backend}" = hip ]; then DEV=HIP; NPS='[1-4]'; else DEV=SERIAL; NPS='[1-6]'; fi
bctest() {
    if [ "${backend}" = hip ]; then
        (
            export MPICH_GPU_SUPPORT_ENABLED=1 GTL_HSA_VSMSG_CUTOFF_SIZE=4096 \
                FI_CXI_ATS=0 HSA_XNACK=1 MPICH_SMP_SINGLE_COPY_MODE=NONE
            canopy_ctest "$@"
        )
    else
        canopy_ctest "$@"
    fi
}

run_stem() {
    local label=$1 stem=$2
    echo "### ${label} ${stem} ${DEV} np ${NPS} ###"
    bctest "^Canopy_Test_${stem}_MPI_${DEV}_np_${NPS}\$" -V
    echo "### ${label} ${stem} ${DEV}: canopy_ctest rc=$? ###"
}

case ${mode} in
measure)
    run_stem "pass 1" DownwardSweep
    run_stem "pass 2" DownwardSweep ;;
esac
watchdog_stop
echo "### cancelled sub-jobs ###"
cat ${WATCHDOG_DIR}/cancelled
exit 0
