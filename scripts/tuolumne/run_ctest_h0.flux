#!/bin/bash
# flux: --job-name=canopy-h0
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance H0c: the HIP baseline. H0b's watchdog self-test, then each
# named stem's HIP entries once through canopy_ctest (MPI stems at np 1-4 only,
# one APU per rank as registered by H0a; CartesianTaylor's non-MPI entry), with
# MultiSolve run with -V twice. HIP only ever runs at np 1-4.
#
# Arguments: the stems to run (default: all ten). A stem whose binary was not
# built is reported and skipped.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_h0.flux [Stem ...])
#   flux job status "$jobid"; echo "status rc=$?"
#
# The HIP GPU-aware-MPI variables are exported in a subshell around each HIP
# canopy_ctest call only. No set -e: MultiSolve exits non-zero on its
# pre-existing 1e-8 failures; read the [canopy_ctest] lines.
#
# Budget: every entry cancelled at its budget would cost the sum of the
# np 1-4 budgets, ~10 min, plus a second MultiSolve pass; 30 bounds that.

STEMS=${*:-MultiSolve DownwardSweep UpwardSweep TreeBuilder TreePartitioner CommunicationPlan LaplaceSolve CartesianTaylorSolve FarFieldContract CartesianTaylor}

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
echo "submit: flux batch scripts/tuolumne/run_ctest_h0.flux ${*}"
echo "jobid: ${JOBID}"
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
echo "stems: ${STEMS}"
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

for stem in ${STEMS}; do
    if [ "${stem}" = CartesianTaylor ]; then
        exe=${CANOPY_BUILD}/tests/Canopy_Test_${stem}_HIP
        regex="^Canopy_Test_${stem}_HIP\$"
    else
        exe=${CANOPY_BUILD}/tests/Canopy_Test_${stem}_MPI_HIP
        regex="^Canopy_Test_${stem}_MPI_HIP_np_[1-4]\$"
    fi
    if [ ! -x "${exe}" ]; then
        echo "### ${stem}: NOT BUILT (${exe}); skipped ###"
        continue
    fi
    echo "### ${stem} binary: $(stat -c '%y' ${exe}) ###"
    out=${WATCHDOG_DIR}/${stem}.out
    : > ${out}
    if [ "${stem}" = MultiSolve ]; then
        for pass in 1 2; do
            echo "### ${stem} HIP pass ${pass} (-V) ###"
            hip_ctest "${regex}" -V | tee -a ${out}
            echo "### ${stem} pass ${pass}: canopy_ctest rc=${PIPESTATUS[0]} ###"
        done
    else
        echo "### ${stem} HIP ###"
        hip_ctest "${regex}" --output-on-failure | tee -a ${out}
        echo "### ${stem}: canopy_ctest rc=${PIPESTATUS[0]} ###"
    fi
    # The SERIAL twin of every HIP entry that failed, for H0c's comparison.
    # MultiSolve's six 1e-8 failures are already recorded for SERIAL (H2).
    [ "${stem}" = MultiSolve ] && continue
    for hip_entry in $(sed -n 's/^\[canopy_ctest\] \([^ ]*\) .*outcome=failed$/\1/p' ${out} | sort -u); do
        serial_entry=${hip_entry/_HIP/_SERIAL}
        echo "### SERIAL twin of failed ${hip_entry}: ${serial_entry} ###"
        canopy_ctest "^${serial_entry}\$" --output-on-failure
        echo "### ${serial_entry}: canopy_ctest rc=$? ###"
    done
done

# UpwardSweep's SERIAL entries exit 8 at every np (H0b calibration); record
# which cases fail, for README "Known Issues".
if [[ " ${STEMS} " == *" UpwardSweep "* ]]; then
    echo "### SERIAL UpwardSweep np 1-6 (--output-on-failure) ###"
    canopy_ctest '^Canopy_Test_UpwardSweep_MPI_SERIAL_np_[1-6]$' --output-on-failure
    echo "### SERIAL UpwardSweep: canopy_ctest rc=$? ###"
fi
watchdog_stop
echo "### cancelled sub-jobs ###"
cat ${WATCHDOG_DIR}/cancelled
exit 0
