#!/bin/bash
# flux: --job-name=canopy-t9-baseline
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T9 baseline: Canopy_Test_DownwardSweep_MPI_SERIAL at unmodified HEAD.
# No task in tasks/abstract-solver-backend.md has ever RUN this target — it
# has only been compiled — so a pre-existing failure there would otherwise be
# charged to T9. This job records the failure set before any T9 edit.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_t9_baseline.flux)
#   flux job status "$jobid"

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
echo "submit: flux batch scripts/tuolumne/run_ctest_t9_baseline.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

echo "### DownwardSweep (baseline, unmodified HEAD) ###"
ctest -V -R Canopy_Test_DownwardSweep_MPI_SERIAL
rc_ds=$?

echo "### exit codes: DownwardSweep=${rc_ds} ###"
exit ${rc_ds}
