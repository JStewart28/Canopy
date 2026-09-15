#!/bin/bash
# flux: --job-name=canopy-op-table-budget-prof
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T8's instrumentation readout for tasks/abstract-solver-backend.md — the
# realized M2L key count and the operator table's size, which risk R7 turns on.
#
# The emission is guarded by CANOPY_ENABLE_PROFILING and the committed
# build-tuolumne/ is configured OFF, so it prints nothing there. This runs the
# SECOND build tree, build-tuolumne-prof/ (ON, PROFILING_LEVEL=2), the same one
# scripts/tuolumne/run_laplace_solve_profile.flux uses for R3.
#
# np=1 of both suites: that is all three configurations this document has —
# the Laplace-solve gate's 600-particle P=6 tree, and FarFieldContract's Basic
# (1000/rank, ncrit 32) and Small (200/rank, ncrit 16), plus the ncrit-4 tree
# levelReachesTheKey stands up. The key count is per rank and unreduced, so a
# single-rank run is the cleanest read; the multi-rank per-rank counts are
# already in the gate log's [laplace-solve] lines.
#
# Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_op_table_budget_profile.flux)
#   flux job status "$jobid"

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/canopy-dev/Canopy/build-tuolumne-prof
CANOPY_SRC=/g/g20/stewartj/research-bridges/canopy-dev/Canopy

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see systems/tuolumne/claude.md). Required for
# every Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_op_table_budget_profile.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

echo "### LaplaceSolve np=1 ###"
ctest -V -R 'Canopy_Test_LaplaceSolve_MPI_SERIAL_np_1$'
rc_ls=$?

echo "### FarFieldContract np=1 ###"
ctest -V -R 'Canopy_Test_FarFieldContract_MPI_SERIAL_np_1$'
rc_ffc=$?

echo "### exit codes: LaplaceSolve=${rc_ls} FarFieldContract=${rc_ffc} ###"
if [ ${rc_ls} -ne 0 ]; then exit ${rc_ls}; fi
exit ${rc_ffc}
