#!/bin/bash
# flux: --job-name=canopy_t8b_fallback_reasons
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=40
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T8b's FAILURE DIRECTION (tasks/add-canopy-t6.md in the Beatnik repo): in a
# ~profiling build every per-reason fallback counter must read -1 and NOT 0,
# and the sum identity must be SKIPPED rather than passing vacuously on a row
# of sentinels (risk R7).
#
# WHY THIS RUNS IN CANOPY AND NOT IN BEATNIK. The Beatnik env concretizes
# `canopy +profiling` and has since T4, so no Beatnik binary in it can observe
# the sentinel at all -- the direction is unobservable from the probe. It is
# therefore taken the way T1's and T6's were: one case in this suite, built in
# BOTH of T1's cmake trees, run at ranks 1-6 on SERIAL in one job.
#
#   build-t1-prof-on   -DCanopy_ENABLE_PROFILING=ON  -> the three counters
#                      carry real pair counts, fb_range_guard + fb_count_cap
#                      == total_fallback_pair_count() is ASSERTED, and
#                      fb_dropped must be 0.
#   build-t1-prof-off  -DCanopy_ENABLE_PROFILING=OFF -> all three read the -1
#                      sentinel, asserted explicitly, and the identity is not
#                      evaluated.
#
# The new case is m2lFallbackReasonBreakdown. It reuses
# testM2LOpCountCapConstrained's configuration -- a count cap of
# LS_COUNT_CAP_KEYS columns against the default 2 GB byte budget -- which
# already realizes exactly that many columns with non-zero fallback at every
# rank count, so the COUNT CAP reason is driven and the identity has a
# non-trivial right-hand side. The two ungated preconditions (eff_cap and
# fallback > 0) are checked in both trees.
#
# The suite is now NINE gtest cases per rank count; the six ctest "tests" are
# the six rank counts.
#
# One job, two ctest invocations. CTest launches each test through
# MPIEXEC_EXECUTABLE (`flux run --ntasks N --nodes=1 --exclusive
# --cores-per-task=1`), which nests inside this allocation, so there is no
# second layer of flux run here.
#
# Note: the walltime option on this flux is --time-limit=<minutes>; a bare
# --time is rejected. --flags=waitable is also rejected on this instance, so
# `flux job status <jobid>` is the wait.
#
# Usage:  flux batch scripts/tuolumne/run_t8b_fallback_reasons.flux
# Then read canopy_t8b_fallback_reasons.<jobid>.log.

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
echo "submit: flux batch scripts/tuolumne/run_t8b_fallback_reasons.flux"
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

# -V so the per-rank "[laplace-solve] ... m2l_fallback_reasons" lines reach
# this log even when every rank count passes: they carry the three counters,
# the fallback total they must sum to, and breakdown_available, which are the
# exit criterion's evidence.
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
echo "Read, per tree, the 21 (nprocs, rank) m2l_fallback_reasons lines:"
echo "  ON  tree: fb_range_guard + fb_count_cap == fallback at every pair,"
echo "            fb_dropped=0, fb_count_cap>0, breakdown_available=1."
echo "  OFF tree: fb_range_guard=fb_count_cap=fb_dropped=-1 and NOT 0, with"
echo "            breakdown_available=0 -- the identity is not evaluated."
exit ${rc_total}
