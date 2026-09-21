#!/bin/bash
# flux: --job-name=canopy-cartesian-taylor-solve-t5
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# T5's exit criterion, in one job:
#
#   Canopy_Test_CartesianTaylorSolve_MPI_SERIAL   np 1-6
#     matchesDirectSumThetaRef              mac_theta = 0.3, p = 3, gated at
#                                           or below the 1e-3 reference bar
#     matchesDirectSumThetaCanopy           mac_theta = 0.5, p = 2, pinned
#     operatorCacheAcrossDriftThetaCanopy   mac_theta = 0.5, p = 2, the long
#                                           trajectory — the R6 measurement
#     operatorCacheAcrossDriftThetaRef      mac_theta = 0.3, p = 2, likewise
#
# The two gating arms are here because they share the harness the measurement
# changed and must come through unmoved at their pinned tolerances.
#
# NO LaplaceSolve HALF, unlike run_ctest_cartesian_taylor.flux. T5 changes no
# file under src/ — the whole change is in tests/tstCartesianTaylorSolve.hpp —
# so the Laplace-solve gate has nothing to guard here, and the two added arms
# need the walltime the second suite was using.
#
# CTest already knows the rank counts (Canopy_TEST_MPI_RANKS, default 1-6) and
# launches each test through MPIEXEC_EXECUTABLE (`flux run --ntasks N
# --nodes=1 --exclusive --cores-per-task=1`, per run_cmake_tuolumne.sh:10-12),
# which nests inside this allocation. These are SERIAL (host) targets: there
# is no rank-to-GPU binding to add and no launcher to add here. Submit with:
#   jobid=$(flux batch scripts/tuolumne/run_ctest_cartesian_taylor_t5.flux)
#   flux job status "$jobid"
#
# --time-limit=20 and not the 8 the SERIAL-only scripts use: four arms at
# np 1-6 do not fit 8 minutes with any margin. Note --flags=waitable is
# rejected here ("only the instance owner can submit with FLUX_JOB_WAITABLE"),
# so `flux job status` is the wait, and never `flux job attach` — it forwards
# signals and cancels the job on two SIGINTs. Note too that the walltime
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
echo "submit: flux batch scripts/tuolumne/run_ctest_cartesian_taylor_t5.flux"
echo "host: $(hostname)"
spack env status
echo "compiler: $(CC --version 2>&1 | head -2 | tr '\n' ' ')"
echo "git HEAD: $(git -C ${CANOPY_SRC} rev-parse HEAD)"
echo "git status (porcelain):"
git -C ${CANOPY_SRC} status --porcelain
echo "=================="

cd ${CANOPY_BUILD}

# -V so the per-rank, PER-STEP "[ct-cache] ..." lines reach this log: they are
# the measurement T5 exists to record (m2l_op_keys_built_count(),
# m2l_op_cache_size(), interaction_list_build_count() and both root
# half-widths, after every solve), and the once-per-run "[ct-solve] ..." lines
# beside them carry the configuration and the direct-sum deviations.
ctest -V -R Canopy_Test_CartesianTaylorSolve_MPI_SERIAL
rc_cts=$?

echo "### exit code: CartesianTaylorSolve=${rc_cts} ###"
exit ${rc_cts}
