#!/bin/bash
# flux: --job-name=canopy-ctest-minset-dev
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=15
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# The required ship gate (CLAUDE.md "Minimum test set") for the canopy-dev
# checkout: every `regression`-labeled test on the SERIAL backend at ranks
# 1-6. Identical to run_ctest_minset.flux except for CANOPY_BUILD, which that
# script points at a different checkout. Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_ctest_minset.canopy-dev.flux)
#   flux job status "$jobid"
#
# Note: --flags=waitable is rejected here ("only the instance owner can
# submit with FLUX_JOB_WAITABLE"), so `flux job status` is the wait. Note
# too that the walltime option is --time-limit; a bare --time is rejected
# by this flux, which is why run_ctest_minset.flux's `# flux: --time=15`
# will not submit.

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne
CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see systems/tuolumne/claude.md). Required for
# every Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_ctest_minset.canopy-dev.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

ctest --output-on-failure -L regression -R MPI_SERIAL
exit $?
