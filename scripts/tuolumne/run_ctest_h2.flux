#!/bin/bash
# flux: --job-name=canopy-h2
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance H2, with Zoltan2 MJ on the partitioner's execution
# space (src/Canopy_TreePartitioner.hpp, partition_leaves). One mode per job:
#
#   fixed  15 consecutive MultiSolve np-3 runs under the watchdog. Does not
#          stop at a hang; a run is hung only if the watchdog cancelled it
#          (ctest exits 8 either way from the pre-existing 1e-8 failures).
#   repro  two -V passes over MultiSolve np 1-6, for the per-(nprocs, case)
#          [multisolve-dev] comparison between passes and against
#          canopy-v1.f3bmo4JYikKh.log at np 1-2.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_h2.flux fixed)
#   flux job status "$jobid"; echo "status rc=$?"
#
# The reverted-build check is run_ctest_h1.flux itself.
#
# Budget: fixed is 15 x ~15 s plus a 15 s watchdog poll each, plus up to a
# 300 s hang per failing run; repro is ~2 x 70 s. --flags=waitable is
# rejected here; `flux job status` is the wait.

mode=$1
case ${mode} in
    fixed|repro) ;;
    *) echo "usage: flux batch run_ctest_h2.flux fixed|repro"; exit 2 ;;
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
echo "submit: flux batch scripts/tuolumne/run_ctest_h2.flux ${mode}"
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
