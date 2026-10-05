# Shared flux sub-job watchdog. SOURCE this from a flux batch script; do not
# execute it. Sourcing starts the loop (watchdog_start).
#
# Why: ctest --timeout does not bound a hung test under flux. ctest kills the
# `flux run` client, but the sub-job keeps running and holds the node
# --exclusive, so every later sub-job waits in state S behind it (flux job
# f3bn8EK66YaK). Every WATCHDOG_POLL_S seconds this loop lists the running
# sub-jobs of the enclosing instance; any one older than its threshold is first
# stack-sampled, then `flux cancel`ed. The threshold is the integer in
# ${WATCHDOG_DIR}/budget while that file exists (written by canopy_ctest in
# ctest_budget.sh for the entry it is running), and WATCHDOG_S otherwise.
#
# Only the sub-job's own processes are stacked: descendants of the flux-shell
# whose last argument is that sub-job's id (`flux-shell [OPTIONS] JOBID`),
# filtered by CANOPY_WATCHDOG_PGREP as an extended regex on the process name
# (comm, truncated to 15 chars). Never a node-wide or full-command-line match:
# that would catch this loop's own `sleep`, ctest (its -R regex) and the
# `flux run` client.
#
# Inputs (read when the loop starts):
#   WATCHDOG_S            cancel threshold, seconds of sub-job runtime, for
#                         sub-jobs canopy_ctest did not launch (300)
#   WATCHDOG_POLL_S       poll period, seconds (2)
#   CANOPY_WATCHDOG_PGREP process-name regex of the processes to stack
#                         (Canopy_Test_)
#
# Functions:
#   watchdog_start          start the loop (called on source)
#   watchdog_stop           kill the loop
#   watchdog_wait_idle [s]  block until a poll that began after the call sees no
#                           running sub-job; 1 after s seconds (180)
#   watchdog_cancel_count   number of sub-jobs cancelled since sourcing
#
# Each cancel appends "<jobid> runtime=<s> shell=<pid> matched=<n>
# children=<n> nonempty=<n>" to ${WATCHDOG_DIR}/cancelled. matched counts
# stacked processes, children those whose parent is the shell itself, nonempty
# those whose capture holds at least one "#0" frame.

: "${WATCHDOG_S:=300}"
: "${WATCHDOG_POLL_S:=2}"
WATCHDOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/canopy-watchdog.XXXXXX")
: > "${WATCHDOG_DIR}/cancelled"

# Stack one PID, falling back per risk R1: gstack, then eu-stack, then gdb.
_watchdog_stack_pid() {
    local pid=$1 out tool
    for tool in gstack eu-stack gdb; do
        case ${tool} in
            gstack)   out=$(gstack "${pid}" 2>&1) ;;
            eu-stack) out=$(eu-stack -p "${pid}" 2>&1) ;;
            gdb)      out=$(gdb -batch -ex 'thread apply all bt' -p "${pid}" 2>&1) ;;
        esac
        if grep -q '^#0 ' <<< "${out}"; then
            echo "tool: ${tool}"
            echo "${out}"
            return 0
        fi
        echo "tool: ${tool} FAILED:"
        echo "${out}" | head -20
    done
    return 1
}

_watchdog_capture_and_cancel() {
    local jid=$1 rt=$2 dec shell snap pids pid ppid comm
    local matched=0 children=0 nonempty=0
    dec=$(flux job id --to=dec "${jid}")
    snap=$(ps -e -o pid=,ppid=,comm=)
    shell=$(ps -e -o pid=,comm=,args= |
        awk -v d="${dec}" -v f="${jid}" \
            '$2 == "flux-shell" && ($NF == d || $NF == f) { print $1; exit }')
    echo "### watchdog stacks ${jid} ###"
    echo "time: $(date '+%Y-%m-%d %H:%M:%S') runtime: ${rt} s jobid(dec): ${dec}"
    if [ -z "${shell}" ]; then
        echo "WATCHDOG ERROR: no flux-shell with JOBID ${dec} or ${jid}; flux-shells present:"
        ps -e -o pid=,ppid=,args= | awk '$3 ~ /flux-shell/'
    else
        echo "shell: $(ps -o pid=,args= -p "${shell}")"
        # Descendants of the shell, breadth first, from one ps snapshot.
        pids=$(awk -v root="${shell}" '
            { parent[$1] = $2; name[$1] = $3; order[NR] = $1 }
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
            }' <<< "${snap}")
        for pid in ${pids}; do
            read -r ppid comm < <(ps -o ppid=,comm= -p "${pid}")
            [[ -n "${comm}" && "${comm}" =~ ${CANOPY_WATCHDOG_PGREP} ]] || continue
            matched=$((matched + 1))
            [ "${ppid}" = "${shell}" ] && children=$((children + 1))
            echo "--- pid ${pid} ppid ${ppid} comm ${comm} ---"
            _watchdog_stack_pid "${pid}" && nonempty=$((nonempty + 1))
        done
    fi
    echo "### end watchdog stacks ${jid}: matched=${matched} children=${children} nonempty=${nonempty} ###"
    echo "### watchdog: cancelling sub-job ${jid} after ${rt} s ###"
    flux cancel "${jid}"
    echo "${jid} runtime=${rt} shell=${shell:-none} matched=${matched}" \
        "children=${children} nonempty=${nonempty}" >> "${WATCHDOG_DIR}/cancelled"
}

watchdog_start() {
    local pattern=${CANOPY_WATCHDOG_PGREP:-Canopy_Test_}
    echo "### watchdog start: WATCHDOG_S=${WATCHDOG_S} poll=${WATCHDOG_POLL_S} s pattern=${pattern} dir=${WATCHDOG_DIR} ###"
    (
        CANOPY_WATCHDOG_PGREP=${pattern}
        while true; do
            begin=$(date +%s%N)
            listing=$(flux jobs --filter=running --no-header -o '{id} {runtime}')
            running=0
            threshold=${WATCHDOG_S}
            [ -f "${WATCHDOG_DIR}/budget" ] && read -r threshold < "${WATCHDOG_DIR}/budget"
            while read -r jid rt; do
                [ -z "${jid}" ] && continue
                running=$((running + 1))
                grep -q "^${jid} " "${WATCHDOG_DIR}/cancelled" && continue
                if awk -v r="${rt}" -v t="${threshold}" 'BEGIN { exit !(r > t) }'; then
                    _watchdog_capture_and_cancel "${jid}" "${rt}"
                fi
            done <<< "${listing}"
            echo "${begin} ${running}" > "${WATCHDOG_DIR}/tick.tmp"
            mv "${WATCHDOG_DIR}/tick.tmp" "${WATCHDOG_DIR}/tick"
            sleep "${WATCHDOG_POLL_S}"
        done
    ) &
    watchdog_pid=$!
}

watchdog_stop() {
    [ -n "${watchdog_pid}" ] || return 0
    pkill -P "${watchdog_pid}" 2>/dev/null
    kill "${watchdog_pid}" 2>/dev/null
    wait "${watchdog_pid}" 2>/dev/null
    watchdog_pid=
}

watchdog_wait_idle() {
    local limit=${1:-180} t0 begin running
    t0=$(date +%s%N)
    local deadline=$(( $(date +%s) + limit ))
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if [ -f "${WATCHDOG_DIR}/tick" ] &&
            read -r begin running < "${WATCHDOG_DIR}/tick" &&
            [ "${begin}" -ge "${t0}" ] && [ "${running}" -eq 0 ]; then
            return 0
        fi
        sleep 1
    done
    echo "WATCHDOG ERROR: sub-jobs still running ${limit} s after watchdog_wait_idle:"
    flux jobs --filter=active
    return 1
}

watchdog_cancel_count() {
    wc -l < "${WATCHDOG_DIR}/cancelled"
}

watchdog_start
