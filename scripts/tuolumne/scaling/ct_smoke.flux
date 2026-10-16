#!/bin/bash
# flux: --job-name=canopy-scaling-ct-smoke
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=30m
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Smoke test + walltime calibration for scaling_breakdown_ct (CartesianTaylorBasis)
# on one node. Driver logs go to /p/lustre5/stewartj/Canopy/scaling_ct_smoke/.
#   flux batch scripts/tuolumne/scaling/ct_smoke.flux

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy
CANOPY_BUILD=/p/lustre5/stewartj/Canopy/scaling_build_ct/build
EXE=${CANOPY_BUILD}/examples/05_scaling_breakdown/scaling_breakdown_ct
OUTDIR=/p/lustre5/stewartj/Canopy/scaling_ct_smoke
mkdir -p ${OUTDIR}

export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

JOBID=$(flux job id --to=f58 "$(flux getattr jobid 2>/dev/null)" 2>/dev/null || echo unknown)
echo "jobid: ${JOBID} host: $(hostname) git: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "binary: $(stat -c '%y' ${EXE})"

cd ${OUTDIR}
rc=0
for spec in "1 40000 shared" "4 40000 strong" "4 160000 weak" "4 2000000 weak -r 1 -w 0"; do
    set -- ${spec}
    np=$1 n=$2 label=$3; shift 3; extra="$*"
    LOG=${OUTDIR}/driver.np${np}_N${n}.${JOBID}.log
    echo "### smoke np=${np} N=${n} label=${label} -> ${LOG}"
    t0=$(date +%s)
    flux run --ntasks=${np} --nodes=1 --exclusive \
        --gpus-per-task=1 --cores-per-task=8 \
        --setopt=mpibind=verbose:1 \
        ${EXE} -N ${n} -l ${label} ${extra} > ${LOG} 2>&1
    r=$?
    echo "### smoke np=${np} N=${n}: rc=${r} elapsed=$(( $(date +%s) - t0 ))s"
    grep -E '^### CANOPY_SCALING (end|done)' ${LOG} | tail -3
    grep -cE 'WARNING|[Ee]rror' ${LOG}
    [ ${r} -eq 0 ] || rc=1
done
exit ${rc}
