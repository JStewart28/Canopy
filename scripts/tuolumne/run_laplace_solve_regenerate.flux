#!/bin/bash
# flux: --job-name=canopy-laplace-solve-regen
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Regenerate the T1 Laplace-solve reference data.
# CANOPY_LAPLACE_SOLVE_REGENERATE makes LaplaceSolve.bitForBitArtifacts write
# its record to <dir>/laplace_solve_np<N>_rank<R>.part and skip the
# comparison; crossRankAgreement and matchesDirectSum skip outright, having
# nothing to compare against yet. bitForBitArtifacts runs at np 1-2 only, so
# one `ctest` invocation over ranks 1-6 produces exactly three parts —
# (1,0), (2,0), (2,1). The np=1 rank-0 part also carries the initial-set hash
# and the np=1 field record. Merge them into
# tests/data/laplace_solve_P6.txt afterwards, under the existing header:
#   cat laplace_solve_np1_rank0.part laplace_solve_np2_rank0.part \
#       laplace_solve_np2_rank1.part
#
# This is a deliberate, manual step: nothing in the default test path can
# write reference data, so a later refactor cannot silently re-baseline
# itself. Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_laplace_solve_regenerate.flux)
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
# every Canopy binary on Tuolumne; must reach the launched task, so export it
# in this batch environment before invoking ctest.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

export CANOPY_LAPLACE_SOLVE_REGENERATE=${CANOPY_BUILD}/laplace_solve_regen
mkdir -p ${CANOPY_LAPLACE_SOLVE_REGENERATE}
rm -f ${CANOPY_LAPLACE_SOLVE_REGENERATE}/*.part

# Provenance. The reference data is only meaningful with the commit it was
# generated at, so record it here rather than reconstructing it later.
echo "=== provenance ==="
echo "submit: flux batch scripts/tuolumne/run_laplace_solve_regenerate.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "regen dir: ${CANOPY_LAPLACE_SOLVE_REGENERATE}"
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank "[laplace-solve] ..." measurement lines reach this log;
# they are the source for the n_unique_ops and fallback counts recorded in the
# progress log.
ctest -V -R Canopy_Test_LaplaceSolve_MPI_SERIAL
rc=$?

echo "=== parts written ==="
ls -1 ${CANOPY_LAPLACE_SOLVE_REGENERATE}
exit $rc
