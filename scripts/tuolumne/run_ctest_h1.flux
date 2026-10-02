#!/bin/bash
# flux: --job-name=canopy-h1
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance H1: contain the np-3 MultiSolve hang and capture its
# stacks, in one allocation.
#
# 1. Self-test of scripts/tuolumne/flux_watchdog.sh: a 3-rank `sleep 900`
#    sub-job must be cancelled at a runtime of 300-330 s with exactly three
#    `sleep` stacks, each a child of that sub-job's flux-shell, and a 4-rank
#    follow-on must then start.
# 2. Up to 20 runs of MultiSolve np 3 under the watchdog, stopping at the first
#    run the watchdog cancels. A run counts as a hang only when the watchdog
#    cancelled it: the six MultiSolve sites fail the pre-existing 1e-8 check
#    (README "Known Issues"), so ctest's exit code is non-zero either way.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_h1.flux)
#   flux job status "$jobid"; echo "status rc=$?"
#
# Budget: self-test ~330 s; a clean np-3 run ~15 s plus up to one 15 s
# watchdog poll; one hang ~300 s + capture. ~22 min worst case; pdebug caps
# at 60. --flags=waitable is rejected here; `flux job status` is the wait.

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
echo "submit: flux batch scripts/tuolumne/run_ctest_h1.flux"
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
echo "ptrace_scope: $(cat /proc/sys/kernel/yama/ptrace_scope)"
echo "=================="

cd ${CANOPY_BUILD}

WATCHDOG_S=300

echo "### self-test ###"
CANOPY_WATCHDOG_PGREP=sleep
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
t0=$(date +%s)
flux run --ntasks=3 --nodes=1 --exclusive --cores-per-task=1 sleep 900
rc_sleep=$?
t1=$(date +%s)
watchdog_wait_idle || exit 3
watchdog_stop
echo "self-test: flux run rc=${rc_sleep}, wall $((t1 - t0)) s"
echo "self-test cancel record: $(cat ${WATCHDOG_DIR}/cancelled)"

read -r st_jid st_rt st_shell st_matched st_children st_nonempty \
    < ${WATCHDOG_DIR}/cancelled
st_rt=${st_rt#runtime=}; st_rt=${st_rt%.*}
selftest_ok=1
[ "$(watchdog_cancel_count)" -eq 1 ] || selftest_ok=0
[ -n "${st_rt}" ] && [ "${st_rt}" -ge 300 ] && [ "${st_rt}" -le 330 ] || selftest_ok=0
[ "${st_matched}" = matched=3 ] || selftest_ok=0
[ "${st_children}" = children=3 ] || selftest_ok=0
[ "${st_nonempty}" = nonempty=3 ] || selftest_ok=0

t0=$(date +%s)
flux run --ntasks=4 --nodes=1 --exclusive --cores-per-task=1 hostname
rc_follow=$?
echo "self-test follow-on: np-4 hostname rc=${rc_follow}, wall $(( $(date +%s) - t0 )) s"
[ ${rc_follow} -eq 0 ] || selftest_ok=0

if [ ${selftest_ok} -ne 1 ]; then
    echo "### self-test FAILED; np-3 loop not run ###"
    exit 2
fi
echo "### self-test PASSED ###"

echo "### np-3 loop ###"
CANOPY_WATCHDOG_PGREP=Canopy_Test_
source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
watchdog_wait_idle || exit 3
hangs=0
runs=0
for i in $(seq 1 20); do
    echo "### np-3 run ${i} ###"
    ctest -V --timeout 300 -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_3$'
    rc=$?
    # ctest's timeout equals WATCHDOG_S, so it returns before the watchdog
    # has stacked and cancelled a hung sub-job; wait for the verdict.
    watchdog_wait_idle || exit 3
    runs=${i}
    if [ "$(watchdog_cancel_count)" -gt 0 ]; then
        hangs=1
        echo "### np-3 run ${i}: HANG, cancelled by watchdog (ctest rc=${rc}) ###"
        break
    fi
    echo "### np-3 run ${i}: completed (ctest rc=${rc}) ###"
done
watchdog_stop
echo "### np-3 summary: hangs=${hangs} runs=${runs} ###"
cat ${WATCHDOG_DIR}/cancelled
exit 0
