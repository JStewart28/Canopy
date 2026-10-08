#!/bin/bash
# flux: --job-name=canopy-c11
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# canopy0 C11: analytic far-field gradient in LaplaceKernel::l2p_evaluate.
# Copied from run_ctest_b1.flux. One mode per job; every run after the
# watchdog self-test goes through canopy_ctest, except `calibrate`, which
# measures the budget rows canopy_ctest needs and does not yet have. A hip run
# gets the HIP environment in a subshell, as in run_ctest_b1.flux.
#   calibrate         SERIAL only. Three plain `ctest --timeout 300` passes of
#                     LaplaceKernel and SingleSolve np 1,2,3,5,6 (no budget
#                     rows exist for them), under the watchdog's 300 s default.
#                     Prints one [calibrate] runtime line per entry.
#   kernel            SERIAL only. LaplaceKernel.
#   oldref            SERIAL only. LaplaceKernel, then LaplaceSolve np 1-6
#                     against whatever tests/data/laplace_solve_P6.txt holds.
#   gate <backend>    serial: LaplaceKernel; LaplaceSolve, MultiSolve,
#                     DownwardSweep np 1-6; SingleSolve np 1,2,3,5,6 last
#                     (README Known Issues: its state can poison a later test).
#                     hip: LaplaceSolve, MultiSolve, DownwardSweep np 1-4.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_c11.flux gate serial)
#   flux job status "$jobid"; echo "status rc=$?"
#
# --flags=waitable is rejected here; `flux job status` is the wait.

mode=$1
backend=${2:-serial}
case ${mode} in
    calibrate|kernel|oldref|gate) ;;
    *) echo "usage: flux batch run_ctest_c11.flux calibrate|kernel|oldref|gate serial|hip"; exit 2 ;;
esac
case ${backend} in serial|hip) ;; *) echo "backend: serial|hip"; exit 2 ;; esac
if [ "${mode}" != gate ] && [ "${backend}" != serial ]; then
    echo "mode ${mode} is SERIAL only"; exit 2
fi

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
unset CANOPY_LAPLACE_SOLVE_REGENERATE

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_c11.flux $*"
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
echo "reference data: $(sha256sum ${CANOPY_SRC}/tests/data/laplace_solve_P6.txt)"
for exe in ${CANOPY_BUILD}/tests/Canopy_Test_{LaplaceSolve,MultiSolve,DownwardSweep,SingleSolve}_MPI_{SERIAL,HIP} \
           ${CANOPY_BUILD}/tests/Canopy_Test_LaplaceKernel_SERIAL; do
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
    local label=$1 stem=$2 nps=${3:-${NPS}}
    echo "### ${label} ${stem} ${DEV} np ${nps} ###"
    bctest "^Canopy_Test_${stem}_MPI_${DEV}_np_${nps}\$" -V
    echo "### ${label} ${stem} ${DEV}: canopy_ctest rc=$? ###"
}

run_nonmpi() {
    local label=$1 stem=$2
    echo "### ${label} ${stem} ${DEV} ###"
    bctest "^Canopy_Test_${stem}_${DEV}\$" -V
    echo "### ${label} ${stem} ${DEV}: canopy_ctest rc=$? ###"
}

calibrate_entry() {
    local entry=$1 t0 t1 rc
    t0=$(date +%s.%N)
    ctest --timeout 300 -V -R "^${entry}\$"
    rc=$?
    t1=$(date +%s.%N)
    watchdog_wait_idle
    echo "[calibrate] ${entry} runtime=$(awk -v a="${t0}" -v b="${t1}" 'BEGIN { printf "%.2f", b - a }') rc=${rc}"
}

case ${mode} in
calibrate)
    for pass in 1 2 3; do
        echo "### calibrate pass ${pass} ###"
        calibrate_entry Canopy_Test_LaplaceKernel_SERIAL
        for np in 1 2 3 5 6; do
            calibrate_entry Canopy_Test_SingleSolve_MPI_SERIAL_np_${np}
        done
    done ;;
kernel)
    run_nonmpi "kernel" LaplaceKernel ;;
oldref)
    run_nonmpi "oldref" LaplaceKernel
    run_stem "oldref" LaplaceSolve ;;
gate)
    if [ "${backend}" = serial ]; then
        run_nonmpi "gate" LaplaceKernel
    fi
    for stem in LaplaceSolve MultiSolve DownwardSweep; do
        run_stem "gate" ${stem}
    done
    if [ "${backend}" = serial ]; then
        run_stem "gate" SingleSolve '[12356]'
    fi ;;
esac
watchdog_stop
echo "### cancelled sub-jobs ###"
cat ${WATCHDOG_DIR}/cancelled
exit 0
