#!/bin/bash
# flux: --job-name=canopy_t1_demand
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=40
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T1's exit criterion (tasks/add-canopy-t6.md in the Beatnik repo): run the
# LaplaceSolve suite at ranks 1-6 on SERIAL in BOTH of T1's cmake trees --
# one configured -DCanopy_ENABLE_PROFILING=ON and one =OFF -- so the two new
# key-demand cases are exercised in each, and so the constrained-budget
# case's REALIZED count, fallback pair count and realized key list can be
# diffed between them. They must be identical; that is what proves the
# demand counter changed no answer (risk R1).
#
# One job, two ctest invocations. CTest launches each test through
# MPIEXEC_EXECUTABLE (`flux run --ntasks N --nodes=1 --exclusive
# --cores-per-task=1`), which nests inside this allocation, so there is no
# second layer of flux run here.
#
# Note: the walltime option on this flux is --time-limit=<minutes>; a bare
# --time is rejected (see scripts/tuolumne/run_ctest_laplace_solve.flux).
# --flags=waitable is also rejected, so `flux job status <jobid>` is the wait.

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_SRC=/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy
BUILD_ON=${CANOPY_SRC}/build-t1-prof-on
BUILD_OFF=${CANOPY_SRC}/build-t1-prof-off

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (systems/tuolumne/claude.md). Required for every
# Canopy binary on Tuolumne.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_t1_demand.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
for b in ${BUILD_ON} ${BUILD_OFF}; do
  echo "tree ${b}"
  echo "  Canopy_ENABLE_PROFILING = $(grep '^Canopy_ENABLE_PROFILING' ${b}/CMakeCache.txt)"
  echo "  Canopy_PROFILING_LEVEL  = $(grep '^Canopy_PROFILING_LEVEL:' ${b}/CMakeCache.txt)"
  echo "  CMAKE_BUILD_TYPE        = $(grep '^CMAKE_BUILD_TYPE' ${b}/CMakeCache.txt)"
done
echo "=================="

rc_total=0

# -V so the per-rank "[laplace-solve] ..." measurement lines reach this log
# even when every rank count passes: they carry n_unique_ops, the demanded
# count, the fallback count, the effective cap and the realized key list,
# which are the exit criterion's evidence.
for tag in ON OFF; do
  if [ "${tag}" = "ON" ]; then b=${BUILD_ON}; else b=${BUILD_OFF}; fi
  echo ""
  echo "########## ctest in the PROFILING=${tag} tree: ${b} ##########"
  cd ${b}
  ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL
  rc=$?
  echo "########## PROFILING=${tag} ctest rc=${rc} ##########"
  rc_total=$(( rc_total + rc ))
done

echo ""
echo "=== combined rc = ${rc_total} ==="
exit ${rc_total}
