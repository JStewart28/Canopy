#!/bin/bash
# flux: --job-name=canopy-treepart-hip
# flux: --nodes=1
# flux: --exclusive
# flux: --time=15
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Device-path validation of the registration-coalesced particle migration
# (issue #22): run the TreePartitioner suite on the HIP backend — including the
# new testCoalescedMigrateIntegrity (multi-peer pack/unpack, exact id
# bijection, payload integrity, ownership) — at 1 and 4 ranks. This is the
# GPU-aware-MPI path that Dane (CPU-only) cannot exercise. Submit with:
#   flux batch run_treepartitioner_hip.flux
#
# HIP/GPU variant (section 4 of docs/tuolumne/claude.md): one APU per rank.

# Build/run profile: env + binary location from the shared resolver
# (scripts/tuolumne/profile.*.sh). Pin the repo root (the scheduler spools this
# script, so its own path is unreliable under `flux batch`).
CANOPY_REPO="${CANOPY_REPO:-/g/g20/stewartj/research-bridges/Canopy}"
source "${CANOPY_REPO}/scripts/lib/canopy_env.sh" || exit 1
TEST=$(canopy_exe tests/Canopy_Test_TreePartitioner_MPI_HIP)
# Runtime env (Cray-MPICH GPU-aware comm, HIP/HMM, OMP_*, static-TLS workaround)
# is exported by the resolver from scripts/tuolumne/runtime_env.sh.

fail=0
for N in 1 4; do
    echo "###################### TreePartitioner(HIP) np=${N} ######################"
    flux run --ntasks=${N} --nodes=1 --exclusive \
        --gpus-per-task=1 --cores-per-task=8 \
        --setopt=mpibind=verbose:1 "${TEST}"
    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "RESULT: np=${N} FAILED (exit ${rc})"
        fail=1
    else
        echo "RESULT: np=${N} PASSED"
    fi
done

echo "===================================================================="
if [ ${fail} -ne 0 ]; then
    echo "TreePartitioner HIP: FAILED."
else
    echo "TreePartitioner HIP: ALL RANKS PASSED."
fi
exit ${fail}
