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

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build-tuolumne
TEST=${CANOPY_BUILD}/tests/Canopy_Test_TreePartitioner_MPI_HIP

# Cray-MPICH GPU-aware comm (required for device pointers in MPI calls).
export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

cd ${CANOPY_BUILD}

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
