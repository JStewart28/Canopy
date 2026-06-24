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

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build-tuolumne
TEST=${CANOPY_BUILD}/tests/Canopy_Test_MultiSolve_MPI_SERIAL

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround. Canopy binaries link multiple cray-libsci
# libraries whose static TLS blocks exhaust the default loader surplus at
# startup ("libsci_cray_mp.so.6: cannot allocate memory in static TLS block",
# exit 127). Enlarging the glibc static-TLS surplus lets the loader place them.
# Required for every Canopy binary on Tuolumne (CPU and GPU).
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

cd ${CANOPY_BUILD}

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
