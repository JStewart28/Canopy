#!/bin/bash
# flux: --job-name=canopy_t1_repro
# flux: --nodes=1
# flux: --exclusive
# flux: --time-limit=30
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# DIAGNOSTIC, not an exit-criterion run. The T1 exit criterion compares the
# realized output of the +profiling and ~profiling trees; the first comparison
# showed the two agreeing exactly on every figure of the two new cases, but
# disagreeing on n_unique_ops for the FIRST solve in the process at some rank
# counts. This job asks whether that figure is reproducible WITHIN a single
# tree: three identical ctest passes per tree. If a tree disagrees with itself,
# the between-tree difference is not evidence about the demand counter.
source /usr/workspace/stewartj/spack/share/spack/setup-env.sh
spack env activate ${HOME}/spack_envs/tuolumne_trilinos
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_WAIT_POLICY=PASSIVE
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2000000
C=/g/g20/stewartj/spack_envs/tuolumne_beatnik/canopy
for tag in ON OFF; do
  if [ "$tag" = ON ]; then b=$C/build-t1-prof-on; else b=$C/build-t1-prof-off; fi
  cd $b
  for pass in 1 2 3; do
    echo "########## tree=$tag pass=$pass ##########"
    ctest -V -R 'Canopy_Test_LaplaceSolve_MPI_SERIAL_np_(3|6)$' 2>&1 \
      | grep -E '^[0-9]+: \[laplace-solve\]|tests passed|tests failed'
    echo "########## tree=$tag pass=$pass rc=$? ##########"
  done
done
