#!/bin/bash
# flux: --job-name=canopy-golden
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=10
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T1's exit criterion: the golden bit-for-bit harness at ranks 1-6 on the
# SERIAL backend. CTest already knows the rank counts (Canopy_TEST_MPI_RANKS,
# default 1-6) and launches each test through MPIEXEC_EXECUTABLE
# (`flux run --ntasks N --nodes=1 --exclusive --cores-per-task=1`, per
# run_cmake_tuolumne.sh), which nests inside this allocation. Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_ctest_golden.flux)
#   flux job status "$jobid"
#
# Note: --flags=waitable is rejected here ("only the instance owner can
# submit with FLUX_JOB_WAITABLE"), so `flux job status` is the wait. Note
# too that the walltime option is --time-limit; a bare --time is rejected
# by this flux, which is why run_ctest_minset.flux's `# flux: --time=15`
# will not submit.
#
# CANOPY_GOLDEN_REGENERATE is deliberately NOT set here: this path always
# compares against the committed reference data and never writes it.

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
echo "submit: flux batch scripts/tuolumne/run_ctest_golden.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank "[golden] ..." measurement lines reach this log even when
# every rank count passes.
ctest -V -R Canopy_Test_Golden_MPI_SERIAL
exit $?
