#!/bin/bash
# flux: --job-name=canopy-f4-sweep
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# tasks/01_fix-tests.md F4 solver-stem sweep (diagnostic). As
# run_ctest_fix_tests.flux, plus: a regex prefixed RAW: has no budget row and
# runs through plain `ctest --timeout ${RAW_TIMEOUT:-180}` under the watchdog
# instead of canopy_ctest. A regex naming an _MPI_HIP_
# entry runs under the HIP GPU-aware-MPI variables (hip_ctest); anything else
# runs without them. Regexes must be anchored (CLAUDE.md).
#
#   jobid=$(flux batch scripts/tuolumne/run_f4_sweep.flux \
#       '^Canopy_Test_UpwardSweep_MPI_SERIAL_np_[1-6]$' \
#       '^Canopy_Test_UpwardSweep_MPI_HIP_np_[1-4]$')
#   flux job status "$jobid"; echo "status rc=$?"
#
# Override the time limit at submit with --time-limit=<min> (pdebug caps at
# 60). FIX_TESTS_CTEST_ARGS (default --output-on-failure) replaces the ctest
# output flag, e.g. -V to keep a passing entry's output. No set -e: read the
# [canopy_ctest] lines.

if [ $# -eq 0 ]; then
    echo "usage: flux batch $0 '<anchored regex>' ..."
    exit 1
fi

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

JOBID=$(flux job id --to=f58 "$(flux getattr jobid 2>/dev/null)" 2>/dev/null || echo unknown)

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_f4_sweep.flux $(printf "'%s' " "$@")"
echo "jobid: ${JOBID}"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "git diff --stat:"
git -C ${CANOPY_SRC} diff --stat
echo "build profiling config:"
grep -E '^Canopy_ENABLE_PROFILING:|^Canopy_PROFILING_LEVEL:' \
    ${CANOPY_BUILD}/CMakeCache.txt
echo "flux: $(flux version | head -1)"
echo "ctest args: ${FIX_TESTS_CTEST_ARGS:=--output-on-failure}"
for exe in ${CANOPY_BUILD}/tests/Canopy_Test_{CommunicationPlan,DownwardSweep,CartesianTaylorSolve,FarFieldContract,MultiSolve,SingleSolve}_MPI_{SERIAL,HIP}; do
    [ -x "${exe}" ] && echo "binary: $(basename ${exe}) $(stat -c '%y' ${exe})"
done
echo "=================="

cd ${CANOPY_BUILD}

n56=$(ctest -N -R '_MPI_HIP_np_[56]$' 2>/dev/null | grep -c 'Test *#')
if [ "${n56}" -ne 0 ]; then
    echo "### ABORT: ${n56} HIP entries registered at np 5-6; reconfigure per H0a ###"
    exit 5
fi
echo "HIP entries at np 5-6: 0"

echo "### self-test ###"
WATCHDOG_S=20
CANOPY_WATCHDOG_PGREP=sleep
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
flux run --ntasks=3 --nodes=1 --exclusive --cores-per-task=1 sleep 900
echo "self-test: flux run rc=$?"
watchdog_wait_idle || exit 3
watchdog_stop
echo "self-test cancel record: $(cat ${WATCHDOG_DIR}/cancelled)"
read -r st_jid st_rt st_shell st_matched st_children st_nonempty \
    < ${WATCHDOG_DIR}/cancelled
st_rt=${st_rt#runtime=}
selftest_ok=1
[ "$(watchdog_cancel_count)" -eq 1 ] || selftest_ok=0
awk -v r="${st_rt}" 'BEGIN { exit !(r >= 20 && r <= 25) }' || selftest_ok=0
[ "${st_matched}" = matched=3 ] || selftest_ok=0
[ "${st_children}" = children=3 ] || selftest_ok=0
[ "${st_nonempty}" = nonempty=3 ] || selftest_ok=0
if [ ${selftest_ok} -ne 1 ]; then
    echo "### self-test FAILED; nothing else run ###"
    exit 2
fi
echo "### self-test PASSED (cancelled at ${st_rt} s) ###"

WATCHDOG_S=300
CANOPY_WATCHDOG_PGREP=Canopy_Test_
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
source ${CANOPY_SRC}/scripts/tuolumne/ctest_budget.sh
watchdog_wait_idle || exit 3

hip_ctest() {
    (
        export MPICH_GPU_SUPPORT_ENABLED=1 GTL_HSA_VSMSG_CUTOFF_SIZE=4096 \
            FI_CXI_ATS=0 HSA_XNACK=1 MPICH_SMP_SINGLE_COPY_MODE=NONE
        canopy_ctest "$@"
    )
}

for regex in "$@"; do
    if [[ ${regex} == RAW:* ]]; then
        regex=${regex#RAW:}
        echo "### RAW ${regex} ###"
        if [[ ${regex} == *_HIP* ]]; then
            ( export MPICH_GPU_SUPPORT_ENABLED=1 GTL_HSA_VSMSG_CUTOFF_SIZE=4096 \
                FI_CXI_ATS=0 HSA_XNACK=1 MPICH_SMP_SINGLE_COPY_MODE=NONE
              ctest --timeout ${RAW_TIMEOUT:-180} -R "${regex}" ${FIX_TESTS_CTEST_ARGS} )
        else
            ctest --timeout ${RAW_TIMEOUT:-180} -R "${regex}" ${FIX_TESTS_CTEST_ARGS}
        fi
        echo "### RAW ${regex}: ctest rc=$? ###"
        watchdog_wait_idle || exit 3
        continue
    fi
    if [[ ${regex} == *_HIP* ]]; then
        echo "### HIP ${regex} ###"
        hip_ctest "${regex}" ${FIX_TESTS_CTEST_ARGS}
    else
        echo "### ${regex} ###"
        canopy_ctest "${regex}" ${FIX_TESTS_CTEST_ARGS}
    fi
    echo "### ${regex}: canopy_ctest rc=$? ###"
done

watchdog_stop
echo "### cancelled sub-jobs ###"
cat ${WATCHDOG_DIR}/cancelled
exit 0
