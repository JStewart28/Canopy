#!/bin/bash
# flux: --job-name=canopy-rollup-sweep
# flux: --nodes=1
# flux: --exclusive
# flux: --time=20
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Phase-1 near-field cost diagnostic (tasks/near-field-softening.md). Runs the
# 05_rollup_nearfield cold-collapse miniapp at near_softening_factor k = 0, 2, 4
# with an IDENTICAL initial condition and a fixed softening length, writing one
# CSV per k. The gap between the n_p2p_particle_pairs columns across the three
# CSVs is the softening-attributable near-field cost as the ball collapses.
# Submit with:  flux batch rollup_sweep.flux
#
# Single rank => the per-rank near-field counters are exact global counts and
# the run is cheapest. SERIAL backend => CPU-only.

source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos

CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build-tuolumne
EXE=${CANOPY_BUILD}/examples/05_rollup_nearfield/rollup_nearfield
OUTDIR=${CANOPY_BUILD}/rollup_sweep
mkdir -p ${OUTDIR}

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Cray static-TLS workaround (see docs/tuolumne/claude.md); must reach the task.
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000

# Shared IC / integration parameters (identical across the sweep). Tune -t/-g/-s
# to reach deeper collapse; keep them the same for every k so the curves are
# comparable.
COMMON="-p 20000 -t 60 -s 5e-3 -g 50 -e 0.02 -R 1.0 -M 1.0 -d 18 -n 32"

for K in 0 2 4; do
  echo "=== near_softening_factor k=${K} ==="
  flux run --ntasks=1 --nodes=1 --exclusive --cores-per-task=8 \
    ${EXE} ${COMMON} -k ${K} -o ${OUTDIR}/rollup_k${K}.csv
done

echo "CSVs written to ${OUTDIR}/rollup_k{0,2,4}.csv"
