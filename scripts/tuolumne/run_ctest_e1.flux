#!/bin/bash
# flux: --job-name=canopy-e1
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance E1. One mode per job; every run goes through
# canopy_ctest after H0b's watchdog self-test. A hip run gets the HIP
# environment in a subshell, as in H0c.
#   base               Do step 0: two -V passes of MultiSolve HIP np 1-2 on the
#                      unmodified binary (probe unset, config default).
#   inert <backend>    probe unset: two -V passes of MultiSolve at np 1-2, for
#                      the inert-when-off comparison.
#   measure <backend>  CANOPY_MULTISOLVE_PROBE=1 (config probe): two -V passes
#                      of MultiSolve at SERIAL np 1-6 or HIP np 1-4.
#   variants           SERIAL np 1 with the probe, twice each under
#                      CANOPY_MULTISOLVE_NPP=1200 (probe-npp1200) and
#                      CANOPY_MAC_THETA=0.7 (probe-theta0.7).
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_e1.flux measure serial)
#   flux job status "$jobid"; echo "status rc=$?"
#
# --flags=waitable is rejected here; `flux job status` is the wait. Override
# the limit at submit with --time-limit=<min> (pdebug caps at 60).

mode=$1
backend=${2:-serial}
case ${mode} in
    base) backend=hip ;;
    inert|measure)
        case ${backend} in serial|hip) ;; *) echo "backend: serial|hip"; exit 2 ;; esac ;;
    variants) backend=serial ;;
    *) echo "usage: flux batch run_ctest_e1.flux base|inert|measure|variants [serial|hip]"; exit 2 ;;
esac

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

# The switches are set per run below, never inherited from the submit shell.
unset CANOPY_MULTISOLVE_PROBE CANOPY_MULTISOLVE_NPP CANOPY_MAC_THETA CANOPY_BUDGET_CONFIG

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_e1.flux $*"
echo "jobid: $(flux job id --to=f58 "$(flux getattr jobid 2>/dev/null)" 2>/dev/null || echo unknown)"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "build profiling config:"
grep -E '^Canopy_ENABLE_PROFILING:|^Canopy_PROFILING_LEVEL:' \
    ${CANOPY_BUILD}/CMakeCache.txt
echo "flux: $(flux version | head -1)"
for exe in ${CANOPY_BUILD}/tests/Canopy_Test_MultiSolve_MPI_{SERIAL,HIP}; do
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

# run_pass <label> <np-class> [VAR=value ...]: one -V MultiSolve pass with the
# given switches exported for that pass only.
run_pass() {
    local label=$1 nps=$2
    shift 2
    echo "### ${label} ${DEV} np ${nps}: $* ###"
    (
        for kv in "$@"; do export "${kv}"; done
        bctest "^Canopy_Test_MultiSolve_MPI_${DEV}_np_${nps}\$" -V
        echo "### ${label} ${DEV}: canopy_ctest rc=$? ###"
    )
}

case ${mode} in
base|inert)
    for pass in 1 2; do run_pass "${mode} pass ${pass}" '[1-2]'; done ;;
measure)
    for pass in 1 2; do
        run_pass "measure pass ${pass}" "${NPS}" CANOPY_MULTISOLVE_PROBE=1 \
            CANOPY_BUDGET_CONFIG=probe
    done ;;
variants)
    for pass in 1 2; do
        run_pass "npp1200 pass ${pass}" 1 CANOPY_MULTISOLVE_PROBE=1 \
            CANOPY_MULTISOLVE_NPP=1200 CANOPY_BUDGET_CONFIG=probe-npp1200
        run_pass "theta0.7 pass ${pass}" 1 CANOPY_MULTISOLVE_PROBE=1 \
            CANOPY_MAC_THETA=0.7 CANOPY_BUDGET_CONFIG=probe-theta0.7
    done ;;
esac
watchdog_stop
echo "### cancelled sub-jobs ###"
cat ${WATCHDOG_DIR}/cancelled
exit 0
