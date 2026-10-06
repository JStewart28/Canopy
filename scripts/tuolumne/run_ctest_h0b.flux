#!/bin/bash
# flux: --job-name=canopy-h0b
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=55
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance H0b: per-test time budgets from measured SERIAL runtimes.
#
# Phase argument (default all):
#   calibrate  watchdog self-test (WATCHDOG_S=20, 3-rank sleep 900), then
#              every SERIAL entry of H0c's stems three times at --timeout 300
#              under the watchdog; writes serial_runtimes.tsv with t_ref_s the
#              max over the three passes. A run that does not complete is a
#              hang: it is reported and the job stops without writing rows.
#   check      two canopy_ctest MultiSolve SERIAL np 1-6 passes (no entry may
#              go over budget) and the no-row refusal.
#   verify     the self-test, then check; no recalibration.
#   all        calibrate then check, in one allocation.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_h0b.flux [phase])
#   flux job status "$jobid"; echo "status rc=$?"
#
# CANOPY_CAL_REGEX (anchored) restricts calibration to the entries it
# matches; rows for every other entry are kept unchanged, e.g.
#   CANOPY_CAL_REGEX='^Canopy_Test_TreePartitioner_MPI_SERIAL_np_[1-6]$' \
#       flux batch scripts/tuolumne/run_ctest_h0b.flux calibrate
# CANOPY_BUDGET_CONFIG (default `default`) names the rows' config. A switch
# configuration exports its switches at submit, e.g.
#   CANOPY_MULTISOLVE_PROBE=1 CANOPY_BUDGET_CONFIG=probe \
#   CANOPY_CAL_REGEX='^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$' \
#       flux batch scripts/tuolumne/run_ctest_h0b.flux calibrate
#
# Budget: a calibration pass of 55 entries is ~8-10 min, so three are ~30 min;
# check is ~5 min. No set -e: MultiSolve exits 8 on its pre-existing 1e-8
# failures, and the refusal is meant to fail.

PHASE=${1:-all}

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne
CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy
TSV=${CANOPY_SRC}/scripts/tuolumne/serial_runtimes.tsv

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see systems/tuolumne/claude.md). Required for
# every Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

JOBID=$(flux job id --to=f58 "$(flux getattr jobid 2>/dev/null)" 2>/dev/null || echo unknown)

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_h0b.flux ${PHASE}"
echo "CANOPY_* environment:"; env | grep "^CANOPY_" | sort
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
echo "=================="

cd ${CANOPY_BUILD}

if [ "${PHASE}" != check ]; then
    echo "### self-test ###"
    WATCHDOG_S=20
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
fi

if [ "${PHASE}" = calibrate ] || [ "${PHASE}" = all ]; then

    echo "### calibration ###"
    WATCHDOG_S=300
    CANOPY_WATCHDOG_PGREP=Canopy_Test_
    source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
    watchdog_wait_idle || exit 3
    CAL_REGEX=${CANOPY_CAL_REGEX:-'^Canopy_Test_(MultiSolve|DownwardSweep|UpwardSweep|TreeBuilder|TreePartitioner|CommunicationPlan|LaplaceSolve|CartesianTaylorSolve|FarFieldContract)_MPI_SERIAL_np_[1-6]$|^Canopy_Test_CartesianTaylor_SERIAL$'}
    echo "calibration regex: ${CAL_REGEX}"
    CAL_CONFIG=${CANOPY_BUDGET_CONFIG:-default}
    echo "calibration config: ${CAL_CONFIG}"
    entries=$(ctest -N -R "${CAL_REGEX}" 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: //p')
    echo "calibration entries: $(echo ${entries} | wc -w)"
    samples=${WATCHDOG_DIR}/samples
    : > ${samples}
    for pass in 1 2 3; do
        for entry in ${entries}; do
            before=$(watchdog_cancel_count)
            out=$(ctest --timeout 300 -R "^${entry}\$" 2>&1)
            ct_rc=$?
            watchdog_wait_idle || exit 3
            line=$(grep -E "Test +#[0-9]+: ${entry} " <<< "${out}" | tail -1)
            rt=$(sed -nE 's/.* ([0-9]+\.[0-9]+) sec.*/\1/p' <<< "${line}")
            echo "[calibrate] pass=${pass} ${entry} rc=${ct_rc} runtime=${rt} :: ${line}"
            if [ "$(watchdog_cancel_count)" -gt "${before}" ] ||
                grep -q Timeout <<< "${line}" || [ -z "${rt}" ]; then
                echo "${out}"
                echo "### calibration HANG/INCOMPLETE: pass ${pass} ${entry}; no rows written ###"
                watchdog_stop
                exit 4
            fi
            echo -e "${entry}\t${rt}" >> ${samples}
        done
    done
    watchdog_stop

    # Carry each row's extra_s / extra_reason allowance over a recalibration,
    # and keep the rows of entries this run did not calibrate.
    old_tsv=${WATCHDOG_DIR}/old.tsv
    new_rows=${WATCHDOG_DIR}/new_rows.tsv
    cp ${TSV} ${old_tsv} 2>/dev/null || : > ${old_tsv}
    {
        awk -F'\t' '
            { if (!($1 in m) || $2 + 0 > m[$1]) m[$1] = $2 + 0; n[$1]++ }
            END { for (e in m) print e "\t" m[e] "\t" n[e] }' ${samples} |
        while IFS=$'\t' read -r entry tmax count; do
            [ "${count}" = 3 ] || { echo "BAD sample count ${count} for ${entry}" >&2; continue; }
            if [[ ${entry} =~ ^Canopy_Test_([A-Za-z0-9]+)_MPI_SERIAL_np_([0-9]+)$ ]]; then
                echo -e "${BASH_REMATCH[1]}\t${BASH_REMATCH[2]}\t${CAL_CONFIG}\t${tmax}\t${JOBID}"
            elif [[ ${entry} =~ ^Canopy_Test_([A-Za-z0-9]+)_SERIAL$ ]]; then
                echo -e "${BASH_REMATCH[1]}\t1\t${CAL_CONFIG}\t${tmax}\t${JOBID}"
            fi
        done | sort -t$'\t' -k1,1 -k2,2n |
        awk -F'\t' -v OFS='\t' 'NR == FNR { if ($6 != "") x[$1 FS $2 FS $3] = $6 OFS $7; next }
            { k = $1 FS $2 FS $3; print (k in x) ? $0 OFS x[k] : $0 }' ${old_tsv} -
    } > ${new_rows}
    {
        echo -e "stem\tnp\tconfig\tt_ref_s\tjobid\textra_s\textra_reason"
        {
            awk -F'\t' 'NR == FNR { n[$1 FS $2 FS $3] = 1; next }
                FNR > 1 && !(($1 FS $2 FS $3) in n)' ${new_rows} ${old_tsv}
            cat ${new_rows}
        } | sort -t$'\t' -k1,1 -k2,2n
    } > ${TSV}
    echo "### serial_runtimes.tsv ($(($(wc -l < ${TSV}) - 1)) rows) ###"
    cat ${TSV}
fi

if [ "${PHASE}" = check ] || [ "${PHASE}" = verify ] || [ "${PHASE}" = all ]; then
    echo "### check ###"
    WATCHDOG_S=300
    CANOPY_WATCHDOG_PGREP=Canopy_Test_
    source ${CANOPY_SRC}/scripts/tuolumne/flux_watchdog.sh
    source ${CANOPY_SRC}/scripts/tuolumne/ctest_budget.sh
    watchdog_wait_idle || exit 3
    for pass in 1 2; do
        echo "### no-false-positive pass ${pass} ###"
        canopy_ctest '^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$'
        echo "canopy_ctest rc=$?"
    done

    echo "### refusal ###"
    n0=$(flux jobs -a --no-header | wc -l)
    canopy_ctest '^Canopy_Test_SingleSolve_MPI_SERIAL_np_1$'
    echo "refusal: canopy_ctest rc=$?"
    n1=$(flux jobs -a --no-header | wc -l)
    echo "refusal: sub-jobs before=${n0} after=${n1}"

    echo "### name guards (no launch) ###"
    for name in Canopy_Test_MultiSolve_MPI_HIP_np_5 Canopy_Test_CartesianTaylor_SERIAL_valgrind \
        Canopy_Test_CartesianTaylor_OPENMP_nt_2 Canopy_Test_MultiSolve_MPI_SERIAL_np_3 \
        Canopy_Test_CartesianTaylor_SERIAL; do
        b=$(_canopy_ctest_budget "${name}")
        echo "guard: ${name} rc=$? budget=${b}"
    done

    # Both cancel paths, on budgets far below the real runtimes: the watchdog
    # (MPI, via ${WATCHDOG_DIR}/budget) and canopy_ctest's own timer (non-MPI).
    echo "### forced over-budget (throwaway TSV) ###"
    forced_tsv=${WATCHDOG_DIR}/forced.tsv
    printf 'stem\tnp\tconfig\tt_ref_s\tjobid\nMultiSolve\t6\tdefault\t2\tforced\nCartesianTaylor\t1\tdefault\t0.5\tforced\n' \
        > ${forced_tsv}
    CANOPY_BUDGET_TSV=${forced_tsv} canopy_ctest '^Canopy_Test_MultiSolve_MPI_SERIAL_np_6$'
    echo "forced MPI: canopy_ctest rc=$?"
    CANOPY_BUDGET_TSV=${forced_tsv} canopy_ctest '^Canopy_Test_CartesianTaylor_SERIAL$'
    echo "forced non-MPI: canopy_ctest rc=$?"
    watchdog_stop
fi
exit 0
