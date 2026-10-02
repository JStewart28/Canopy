#!/bin/bash
# flux: --job-name=canopy-t1-step1
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T1 step 1: report the per-reason fallback breakdown on the EXISTING
# clustered fixture, unchanged, at its own ncrit = 8, max_depth = 8,
# mac_theta = 0.3. The answer decides whether T1 step 2 is a relocation of
# this distribution or a new two-scale one, so this runs before any new
# fixture is written.
#
# -V so the [m2l-fallback-reason] lines reach this log even though every
# rank count passes. One line per (nprocs, rank, step): range_guard,
# count_cap, depth_dropped, the fallback total, unique_ops and the
# per-depth occupancy.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_t1_step1.flux)
#   flux job status "$jobid"
#
# Note: --flags=waitable is rejected here ("only the instance owner can submit
# with FLUX_JOB_WAITABLE"), so `flux job status` is the wait. The walltime
# option is --time-limit; a bare --time is rejected by this flux.

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
echo "submit: flux batch scripts/tuolumne/run_ctest_t1_step1.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

echo "### MultiSolve ###"
ctest -V -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$'
rc_ms=$?

echo "### exit codes: MultiSolve=${rc_ms} ###"
exit ${rc_ms}
