# Per-entry ctest time budgets. SOURCE this from a flux batch script after
# scripts/tuolumne/flux_watchdog.sh; do not execute it.
#
#   canopy_ctest <anchored-regex> [ctest args...]
#
# Lists the entries the regex matches (ctest -N), checks every one before
# launching any, then runs each alone as
#   ctest --timeout <budget> -R '^<entry>$' [ctest args...]
# followed by watchdog_wait_idle. The budget is ceil(1.75 * t_ref_s) + extra_s
# seconds, t_ref_s and extra_s (optional, default 0) being columns of the
# (stem, np, ${CANOPY_BUDGET_CONFIG:-default}) row of CANOPY_BUDGET_TSV, for
# every backend. extra_s is a measured allowance for work outside the solve
# (e.g. ctest digesting a large failure output); the row's extra_reason column
# says what. The first entry run after flux_watchdog.sh is sourced also gets
# CANOPY_COLD_START_S (6): the first Canopy binary launched in a job runs ~4 s
# slow. One line per entry:
#   [canopy_ctest] <entry> runtime=<s> budget=<s> outcome=<completed|failed|over-budget>
#
# Refused before anything launches (exit 2, naming the entry):
#   - a name other than Canopy_Test_<Stem>_MPI_<DEV>_np_<N> or
#     Canopy_Test_<Stem>_<DEV> (e.g. _valgrind, _nt_<k>);
#   - an _MPI_HIP_ entry above np 4 (a node has four APUs);
#   - an entry with no budget row. There is no default budget.
#
# MPI entries run as flux sub-jobs, which the watchdog stacks and cancels at
# the budget (it reads ${WATCHDOG_DIR}/budget). Non-MPI entries are not
# sub-jobs, so this file stacks and kills them itself at the budget. Either way
# ctest's --timeout is budget + 5 s, so ctest does not kill the launcher first:
# a sub-job that dies with its killed client would leave no stacks.
#
# Returns 0 if every entry completed, 1 if any failed or went over budget.

CANOPY_BUDGET_TSV=${CANOPY_BUDGET_TSV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serial_runtimes.tsv}
: "${CANOPY_COLD_START_S:=6}"
_CANOPY_DEVICES='SERIAL|OPENMP|THREADS|HIP|CUDA|CUDA_UVM|SYCL'

# Sets _cc_stem, _cc_dev, _cc_np, _cc_mpi from an entry name; 1 if unparseable.
_canopy_ctest_parse() {
    local name=$1
    if [[ ${name} =~ ^Canopy_Test_([A-Za-z0-9]+)_MPI_(${_CANOPY_DEVICES})_np_([0-9]+)$ ]]; then
        _cc_stem=${BASH_REMATCH[1]} _cc_dev=${BASH_REMATCH[2]} _cc_np=${BASH_REMATCH[3]} _cc_mpi=1
    elif [[ ${name} =~ ^Canopy_Test_([A-Za-z0-9]+)_(${_CANOPY_DEVICES})$ ]]; then
        _cc_stem=${BASH_REMATCH[1]} _cc_dev=${BASH_REMATCH[2]} _cc_np=1 _cc_mpi=0
    else
        return 1
    fi
}

# Prints the budget in seconds for an entry; on refusal prints the reason to
# stderr and returns 1.
_canopy_ctest_budget() {
    local name=$1 config=${CANOPY_BUDGET_CONFIG:-default} row t_ref extra
    if ! _canopy_ctest_parse "${name}"; then
        echo "canopy_ctest: REFUSED ${name}: not Canopy_Test_<Stem>_MPI_<DEV>_np_<N> or Canopy_Test_<Stem>_<DEV>" >&2
        return 1
    fi
    if [ "${_cc_mpi}" = 1 ] && [ "${_cc_dev}" = HIP ] && [ "${_cc_np}" -gt 4 ]; then
        echo "canopy_ctest: REFUSED ${name}: HIP above np 4 oversubscribes a node's four APUs" >&2
        return 1
    fi
    row=$(awk -F'\t' -v s="${_cc_stem}" -v n="${_cc_np}" -v c="${config}" \
        '$1 == s && $2 == n && $3 == c { print $4, ($6 == "" ? 0 : $6); exit }' \
        "${CANOPY_BUDGET_TSV}" 2>/dev/null)
    read -r t_ref extra <<< "${row}"
    if [ -z "${t_ref}" ]; then
        echo "canopy_ctest: REFUSED ${name}: no budget row (${_cc_stem}, ${_cc_np}, ${config}) in ${CANOPY_BUDGET_TSV}" >&2
        return 1
    fi
    awk -v t="${t_ref}" -v e="${extra}" 'BEGIN { x = 1.75 * t; c = int(x); if (c < x) c++; print c + e }'
}

# Stacks, then kills, the descendants of $1 whose comm matches
# CANOPY_WATCHDOG_PGREP, in the watchdog's capture format.
_canopy_ctest_stack_and_kill() {
    local root=$1 label=$2 rt=$3 pids pid ppid comm matched=0 nonempty=0
    local pattern=${CANOPY_WATCHDOG_PGREP:-Canopy_Test_}
    pids=$(ps -e -o pid=,ppid= | awk -v root="${root}" '
        { parent[$1] = $2; order[NR] = $1 }
        END {
            keep[root] = 1
            do {
                grew = 0
                for (i = 1; i <= NR; i++) {
                    p = order[i]
                    if (!(p in keep) && (parent[p] in keep)) { keep[p] = 1; grew = 1 }
                }
            } while (grew)
            for (i = 1; i <= NR; i++) if (order[i] != root && order[i] in keep) print order[i]
        }')
    echo "### watchdog stacks ${label} ###"
    echo "time: $(date '+%Y-%m-%d %H:%M:%S') runtime: ${rt} s ctest pid: ${root}"
    local victims=
    for pid in ${pids}; do
        read -r ppid comm < <(ps -o ppid=,comm= -p "${pid}")
        [[ -n "${comm}" && "${comm}" =~ ${pattern} ]] || continue
        matched=$((matched + 1))
        echo "--- pid ${pid} ppid ${ppid} comm ${comm} ---"
        _watchdog_stack_pid "${pid}" && nonempty=$((nonempty + 1))
        victims="${victims} ${pid}"
    done
    echo "### end watchdog stacks ${label}: matched=${matched} nonempty=${nonempty} ###"
    echo "### canopy_ctest: killing ${label} after ${rt} s ###"
    [ -n "${victims}" ] && kill -KILL ${victims} 2>/dev/null
}

canopy_ctest() {
    local regex=$1; shift
    local entries entry budget bad=0 rc=0 out ct_rc runtime outcome before t0 elapsed ctpid timed_out
    entries=$(ctest -N -R "${regex}" 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: //p')
    if [ -z "${entries}" ]; then
        echo "canopy_ctest: REFUSED: regex ${regex} matches no ctest entry" >&2
        return 2
    fi
    for entry in ${entries}; do
        _canopy_ctest_budget "${entry}" > /dev/null || bad=1
    done
    [ ${bad} = 0 ] || return 2

    for entry in ${entries}; do
        budget=$(_canopy_ctest_budget "${entry}")
        if [ ! -f "${WATCHDOG_DIR}/warm" ]; then
            budget=$((budget + CANOPY_COLD_START_S))
            : > "${WATCHDOG_DIR}/warm"
        fi
        _canopy_ctest_parse "${entry}"
        out=$(mktemp "${WATCHDOG_DIR}/ctest.XXXXXX")
        before=$(watchdog_cancel_count)
        timed_out=0
        t0=$(date +%s.%N)
        if [ "${_cc_mpi}" = 1 ]; then
            echo "${budget}" > "${WATCHDOG_DIR}/budget"
            ctest --timeout "$((budget + 5))" -R "^${entry}\$" "$@" > "${out}" 2>&1
            ct_rc=$?
        else
            ctest --timeout "$((budget + 5))" -R "^${entry}\$" "$@" > "${out}" 2>&1 &
            ctpid=$!
            while kill -0 "${ctpid}" 2>/dev/null; do
                elapsed=$(awk -v a="${t0}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')
                if awk -v e="${elapsed}" -v b="${budget}" 'BEGIN { exit !(e >= b) }'; then
                    _canopy_ctest_stack_and_kill "${ctpid}" "${entry}" "${elapsed}"
                    timed_out=1
                    break
                fi
                sleep 0.2
            done
            wait "${ctpid}"
            ct_rc=$?
        fi
        cat "${out}"
        watchdog_wait_idle || rc=1
        rm -f "${WATCHDOG_DIR}/budget"
        runtime=$(grep -E "Test +#[0-9]+: ${entry} " "${out}" | tail -1 |
            sed -nE 's/.* ([0-9]+\.[0-9]+) sec.*/\1/p')
        [ -n "${runtime}" ] || runtime=$(awk -v a="${t0}" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f", b - a }')
        if [ "${timed_out}" = 1 ] || [ "$(watchdog_cancel_count)" -gt "${before}" ] ||
            grep -qE "Test +#[0-9]+: ${entry} .*Timeout" "${out}"; then
            outcome=over-budget
        elif [ "${ct_rc}" = 0 ]; then
            outcome=completed
        else
            outcome=failed
        fi
        [ "${outcome}" = completed ] || rc=1
        echo "[canopy_ctest] ${entry} runtime=${runtime} budget=${budget} outcome=${outcome}"
        rm -f "${out}"
    done
    return ${rc}
}
