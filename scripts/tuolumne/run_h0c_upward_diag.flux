#!/bin/bash
# flux: --job-name=canopy-h0c-upward-diag
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=10
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance H0c diagnostic: UpwardSweep HIP np 3-4 ran every case,
# then outlived its budget with no watchdog cancellation (job f3cM4ghTjtiT).
# Reruns each entry three times under ctest --timeout 300 in the background
# and, once an entry passes its budget, snapshots the flux sub-job states and
# the node's Canopy/flux processes and stacks every Canopy_Test_ process, to
# see what is still alive and why the watchdog does not list it as running.
#
#   jobid=$(flux batch scripts/tuolumne/run_h0c_upward_diag.flux)
#   flux job status "$jobid"; echo "status rc=$?"

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne
CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_h0c_upward_diag.flux"
echo "jobid: $(flux job id --to=f58 "$(flux getattr jobid 2>/dev/null)" 2>/dev/null || echo unknown)"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "flux: $(flux version | head -1)"
echo "=================="

cd ${CANOPY_BUILD}
export MPICH_GPU_SUPPORT_ENABLED=1 GTL_HSA_VSMSG_CUTOFF_SIZE=4096 FI_CXI_ATS=0 \
    HSA_XNACK=1 MPICH_SMP_SINGLE_COPY_MODE=NONE

snapshot() {
    echo "--- flux jobs (all, this instance) ---"
    flux jobs -a --no-header -o '{id.f58} {state} {status} {runtime} {name}' | head -5
    echo "--- processes ---"
    ps -eo pid,ppid,stat,etime,comm,args --sort=pid |
        awk 'NR == 1 || /Canopy_Test_|flux-shell|flux run|ctest/' | grep -v awk | cut -c1-200
    for pid in $(ps -eo pid=,comm= | awk '$2 ~ /^Canopy_Test_/ { print $1 }'); do
        echo "--- gstack ${pid} ($(ps -o stat=,etime= -p ${pid})) ---"
        gstack ${pid} 2>&1 | grep -E '^Thread|^#' | grep -vE 'testing::|ioctl|hsakmt|AsyncEventsLoop|ThreadTrampoline|start_thread|clone' | cut -c1-220
    done
}

for np in 3 4; do
    budget=$(( np == 3 ? 10 : 11 ))
    for run in 1 2 3; do
        entry=Canopy_Test_UpwardSweep_MPI_HIP_np_${np}
        echo "### ${entry} run ${run} (budget ${budget}) ###"
        t0=$(date +%s)
        ctest --timeout 300 -R "^${entry}\$" > /tmp/diag.$$.out 2>&1 &
        ctpid=$!
        snapped=0
        while kill -0 ${ctpid} 2>/dev/null; do
            if [ ${snapped} = 0 ] && [ $(( $(date +%s) - t0 )) -ge ${budget} ]; then
                echo "### past budget at $(( $(date +%s) - t0 )) s ###"
                snapshot
                snapped=1
            fi
            if [ $(( $(date +%s) - t0 )) -ge 60 ]; then
                echo "### still running at 60 s: snapshot, then cancel all ###"
                snapshot
                flux cancel --all 2>&1
                break
            fi
            sleep 1
        done
        wait ${ctpid}
        echo "ctest rc=$? wall $(( $(date +%s) - t0 )) s"
        grep -E "Test +#[0-9]+:|tests ran|PASSED|FAILED  \] [0-9]" /tmp/diag.$$.out
    done
done
rm -f /tmp/diag.$$.out
exit 0
