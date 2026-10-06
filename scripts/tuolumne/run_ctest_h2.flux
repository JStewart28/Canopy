#!/bin/bash
# flux: --job-name=canopy-h2
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance H2. One mode per job.
#
# SERIAL arm (Zoltan2 MJ on the partitioner's execution space), kept as the
# record of jobs f3bnvGYaSqWP and f3bnvGg3MDo5:
#   fixed  15 consecutive MultiSolve SERIAL np-3 runs under the watchdog. A
#          run is hung only if the watchdog cancelled it (ctest exits 8
#          either way from the pre-existing 1e-8 failures).
#   repro  two -V passes over MultiSolve SERIAL np 1-6.
#
# Partitioner arm (ParMETIS cell partition); backend is serial or hip, and
# every run goes through canopy_ctest after H0b's watchdog self-test. A hip
# run gets the HIP environment in a subshell, as in H0c:
#   pfixed <backend> <np>  15 consecutive MultiSolve <backend> np-<np> runs;
#                          the summary counts over-budget entries.
#   pstems <backend>       TreePartitioner, CommunicationPlan, UpwardSweep,
#                          DownwardSweep, LaplaceSolve and MultiSolve once at
#                          SERIAL np 1-6 or HIP np 1-4, --output-on-failure.
#   prepro <backend>       two -V passes of MultiSolve and TreePartitioner at
#                          SERIAL np 2-6 or HIP np 2-4, for the step-7
#                          reproducibility, imbalance, cut and fallback lines.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_h2.flux pfixed hip 3)
#   flux job status "$jobid"; echo "status rc=$?"
#
# Budget: pfixed is 15 entries at MultiSolve's np budget (17-28 s) plus the
# self-test; pstems ~10 min serial, ~6 min hip; prepro ~2 x 3 min. Override
# the limit at submit with --time-limit=<min> (pdebug caps at 60).
# --flags=waitable is rejected here; `flux job status` is the wait.

mode=$1
backend=${2:-serial}
pnp=$3
case ${mode} in
    fixed|repro) ;;
    pfixed|pstems|prepro)
        case ${backend} in serial|hip) ;; *) echo "backend: serial|hip"; exit 2 ;; esac
        if [ "${mode}" = pfixed ] && ! [[ ${pnp} =~ ^[1-6]$ ]]; then
            echo "usage: run_ctest_h2.flux pfixed serial|hip <np>"; exit 2
        fi ;;
    *) echo "usage: flux batch run_ctest_h2.flux fixed|repro|pfixed|pstems|prepro [serial|hip] [np]"; exit 2 ;;
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

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_h2.flux $*"
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
echo "=================="

cd ${CANOPY_BUILD}

if [[ ${mode} == p* ]]; then
    for exe in ${CANOPY_BUILD}/tests/Canopy_Test_{TreePartitioner,CommunicationPlan,UpwardSweep,DownwardSweep,LaplaceSolve,MultiSolve}_MPI_{SERIAL,HIP}; do
        [ -x "${exe}" ] && echo "binary: $(basename ${exe}) $(stat -c '%y' ${exe})"
    done
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

    if [ "${backend}" = hip ]; then
        DEV=HIP; NPS='[1-4]'; NPS2='[2-4]'
    else
        DEV=SERIAL; NPS='[1-6]'; NPS2='[2-6]'
    fi
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

    case ${mode} in
    pfixed)
        log=${WATCHDOG_DIR}/pfixed.out
        : > ${log}
        for i in $(seq 1 15); do
            echo "### ${DEV} np-${pnp} run ${i} ###"
            bctest "^Canopy_Test_MultiSolve_MPI_${DEV}_np_${pnp}\$" \
                --output-on-failure | tee -a ${log}
        done
        echo "### pfixed summary ${DEV} np ${pnp}: runs=15" \
            "over_budget=$(grep -c 'outcome=over-budget' ${log})" \
            "completed=$(grep -c 'outcome=completed' ${log})" \
            "failed=$(grep -c 'outcome=failed' ${log})" \
            "watchdog_cancellations=$(watchdog_cancel_count) ###"
        ;;
    pstems)
        for stem in TreePartitioner CommunicationPlan UpwardSweep DownwardSweep LaplaceSolve MultiSolve; do
            echo "### ${stem} ${DEV} ###"
            bctest "^Canopy_Test_${stem}_MPI_${DEV}_np_${NPS}\$" --output-on-failure
            echo "### ${stem} ${DEV}: canopy_ctest rc=$? ###"
        done
        ;;
    prepro)
        for pass in 1 2; do
            for stem in MultiSolve TreePartitioner; do
                echo "### prepro pass ${pass} ${stem} ${DEV} ###"
                bctest "^Canopy_Test_${stem}_MPI_${DEV}_np_${NPS2}\$" -V
                echo "### prepro pass ${pass} ${stem} ${DEV}: canopy_ctest rc=$? ###"
            done
        done
        ;;
    esac
    watchdog_stop
    echo "### cancelled sub-jobs ###"
    cat ${WATCHDOG_DIR}/cancelled
    exit 0
fi

WATCHDOG_S=300
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
watchdog_wait_idle || exit 3

if [ "${mode}" = fixed ]; then
    for i in $(seq 1 15); do
        before=$(watchdog_cancel_count)
        echo "### np-3 run ${i} ###"
        ctest --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_3$'
        rc=$?
        # ctest's timeout equals WATCHDOG_S, so it returns before the
        # watchdog has stacked and cancelled a hung sub-job.
        watchdog_wait_idle || exit 3
        if [ "$(watchdog_cancel_count)" -gt "${before}" ]; then
            echo "### np-3 run ${i}: HANG, cancelled by watchdog (ctest rc=${rc}) ###"
        else
            echo "### np-3 run ${i}: completed (ctest rc=${rc}) ###"
        fi
    done
    echo "### np-3 summary: hangs=$(watchdog_cancel_count) runs=15 ###"
else
    for pass in 1 2; do
        echo "### repro pass ${pass} ###"
        ctest -V --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$'
        echo "### repro pass ${pass}: ctest rc=$? ###"
        watchdog_wait_idle || exit 3
    done
    echo "### repro summary: watchdog cancellations=$(watchdog_cancel_count) ###"
fi
watchdog_stop
cat ${WATCHDOG_DIR}/cancelled
exit 0
