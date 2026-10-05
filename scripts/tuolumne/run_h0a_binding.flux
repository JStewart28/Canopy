#!/bin/bash
# flux: --job-name=canopy-h0a-binding
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=5
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# fix-hang-rebalance H0a: show that the HIP ctest binding gives one APU per
# rank at np 4, and that flux refuses np 5 as unsatisfiable. That refusal is
# why HIP MPI tests register at np 1-4 only.
#
#   jobid=$(flux batch scripts/tuolumne/run_h0a_binding.flux)
#   flux job status "$jobid"; echo "status rc=$?"
#
# No set -e: the np-5 commands are meant to fail.

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne
CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy

export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_h0a_binding.flux"
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

show='echo rank=$FLUX_TASK_RANK ROCR=${ROCR_VISIBLE_DEVICES-unset} HIP=${HIP_VISIBLE_DEVICES-unset} CUDA=${CUDA_VISIBLE_DEVICES-unset}'

for np in 4 5; do
    echo "### np=${np} verbatim: printenv ROCR_VISIBLE_DEVICES ###"
    flux run --ntasks=${np} --nodes=1 --exclusive --gpus-per-task=1 \
        printenv ROCR_VISIBLE_DEVICES
    echo "rc=$?"
    echo "### np=${np} verbatim binding, all device variables ###"
    flux run --ntasks=${np} --nodes=1 --exclusive --gpus-per-task=1 \
        --label-io sh -c "${show}"
    echo "rc=$?"
    echo "### np=${np} registered HIP preflags (--cores-per-task=8) ###"
    flux run --ntasks=${np} --nodes=1 --exclusive --gpus-per-task=1 \
        --cores-per-task=8 --setopt=mpibind=verbose:1 --label-io sh -c "${show}"
    echo "rc=$?"
done
exit 0
