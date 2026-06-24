#!/bin/bash
# flux: --job-name=canopy-multisolve-minset
# flux: --nodes=1
# flux: --exclusive
# flux: --time=15
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Minimum test set (CLAUDE.md): Canopy_Test_MultiSolve_MPI_SERIAL at
# 1,2,3,4,5,6 ranks. Run this before resubmitting the big Beatnik rollup to
# confirm the registration-coalesced migrate_particles change (issue #22) is
# correct on Tuolumne. Submit with:  flux batch run_multisolve_minset.flux
#
# These tests use the Kokkos SERIAL backend, so they run CPU-only — no
# --gpus-per-task, --cores-per-task=1 (the CPU/SERIAL variant of section 4 in
# docs/tuolumne/claude.md). The OMP_* vars are harmless for Serial.

# Build/run profile: env + binary location from the shared resolver
# (scripts/tuolumne/profile.*.sh). Pin the repo root (the scheduler spools this
# script, so its own path is unreliable under `flux batch`).
CANOPY_REPO="${CANOPY_REPO:-/g/g20/stewartj/research-bridges/Canopy}"
source "${CANOPY_REPO}/scripts/lib/canopy_env.sh" || exit 1
TEST=$(canopy_exe tests/Canopy_Test_MultiSolve_MPI_SERIAL)
# Runtime env (OMP_*, Cray-MPICH/HIP, static-TLS workaround) is exported by the
# resolver from scripts/tuolumne/runtime_env.sh.

fail=0
for N in 1 2 3 4 5 6; do
    echo "######################## MultiSolve np=${N} ########################"
    flux run --ntasks=${N} --nodes=1 --exclusive --cores-per-task=1 "${TEST}"
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
    echo "MINIMUM TEST SET: FAILED — do NOT resubmit the big run."
else
    echo "MINIMUM TEST SET: ALL RANKS PASSED."
fi
exit ${fail}
