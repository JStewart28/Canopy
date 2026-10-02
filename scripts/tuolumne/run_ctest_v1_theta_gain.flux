#!/bin/bash
# flux: --job-name=canopy-v1-gain
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=40
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# V1 step 1, the derivation check -- and V1's failure direction in the same
# job.
#
# QUESTION. The six fmm_tolerance sites compare positions and velocities
# after a short integration, not a field. The bound each one gets has to be
# read against the far-field truncation floor at MultiSolveTest::P_ORDER = 8
# and theta = 0.5, near theta^(P+1) = 1.95e-3, damped by the integrator's
# v += dt * g then r += dt * drift * v. If a measured trajectory deviation is
# LARGER than that floor and that damping account for, the far field is wrong
# and the bound is not the defect (V1 step 1, "stop and report").
#
# MEASUREMENT. get_test_mac_theta() (tests/tstMultiSolve.hpp:38-43) reads
# CANOPY_MAC_THETA, so the far-field truncation error can be moved WITHOUT a
# rebuild and without editing a bound. theta^(P+1) at P_ORDER = 8:
#   theta 0.4 -> 2.62e-04   (0.134x the 0.5 floor)
#   theta 0.5 -> 1.95e-03   (the default; the baseline passes measured this)
#   theta 0.7 -> 4.04e-02   (20.7x the 0.5 floor)
# If each site's deviation moves with that floor, the deviation IS far-field
# truncation error times a per-site gain fixed by the configuration. If a
# deviation is insensitive to theta, it is not truncation, and the
# stop-and-report branch is the right one.
#
# MultiSolve only: the CartesianTaylorSolve arms pin their own theta
# constants and do not read this env var. M2L_BinEdge_Fallback passes
# mac_theta_override = 0.3 and ignores it too.
#
#   jobid=$(flux batch scripts/tuolumne/run_ctest_v1_theta_gain.flux)
#   flux job status "$jobid"; echo "status rc=$?"
#
# pdebug's policy limit is 1 h; a --time-limit above it is rejected at
# submit. --flags=waitable is rejected here, so `flux job status` is the
# wait, and the walltime option is --time-limit, not --time.

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
echo "submit: flux batch scripts/tuolumne/run_ctest_v1_theta_gain.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "build profiling config:"
grep -E '^Canopy_ENABLE_PROFILING:|^Canopy_PROFILING_LEVEL:' \
    ${CANOPY_BUILD}/CMakeCache.txt
echo "=================="

cd ${CANOPY_BUILD}

for theta in 0.4 0.7; do
    echo "### CANOPY_MAC_THETA=${theta} ###"
    CANOPY_MAC_THETA=${theta} ctest -V --timeout 300 \
        -R '^Canopy_Test_MultiSolve_MPI_SERIAL_np_[1-6]$'
    echo "### CANOPY_MAC_THETA=${theta} rc=$? ###"
done
