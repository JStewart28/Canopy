#!/bin/bash
# flux: --job-name=canopy-far-field-contract
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T6's exit criterion: the far-field conformance gate at ranks 1-6 on the
# SERIAL backend, plus the Laplace-solve gate unchanged. Three tests per rank
# count in FarFieldContract (two locals()-against-host-reference bodies at two
# problem sizes, plus the L2P check), and the three Laplace-solve bodies.
# CTest already knows the rank counts (Canopy_TEST_MPI_RANKS, default 1-6) and
# launches each test through MPIEXEC_EXECUTABLE (`flux run --ntasks N
# --nodes=1 --exclusive --cores-per-task=1`, per run_cmake_tuolumne.sh), which
# nests inside this allocation. Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_ctest_far_field_contract.flux)
#   flux job status "$jobid"
#
# Note: --flags=waitable is rejected here ("only the instance owner can
# submit with FLUX_JOB_WAITABLE"), so `flux job status` is the wait. The
# walltime option is --time-limit; a bare --time is rejected by this flux.
#
# This runs both suites in one allocation deliberately: T6 adds a basis and a
# test and touches no shared code, so the Laplace-solve gate's 22 measured
# deviations must reproduce T5's table, and having both in one log makes that
# checkable without correlating two jobs.

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
echo "submit: flux batch scripts/tuolumne/run_ctest_far_field_contract.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank "[far-field-contract] ..." and "[laplace-solve] ..."
# measurement lines reach this log even when every rank count passes. The
# far-field lines carry the checked-slot count, the bit-identity count and the
# fallback count; the Laplace ones carry n_unique_ops, the fallback count and
# the cross-rank and direct-sum deviations the tolerances are pinned from.
echo "### FarFieldContract ###"
ctest -V -R Canopy_Test_FarFieldContract_MPI_SERIAL
rc_ffc=$?

echo "### LaplaceSolve ###"
ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL
rc_ls=$?

echo "### exit codes: FarFieldContract=${rc_ffc} LaplaceSolve=${rc_ls} ###"
if [ ${rc_ffc} -ne 0 ]; then exit ${rc_ffc}; fi
exit ${rc_ls}
