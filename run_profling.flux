#!/bin/bash
# flux: --job-name=tstSolver-canopy
# flux: --nodes=2
# flux: --exclusive 
# flux: --time=10
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug 

module load rocm

SPACK_INSTALL=/usr/workspace/stewartj/spack
CANOPY_ENV=${HOME}/spack_envs/tuolumne_canopy/
CANOPY_SCRATCH=/p/lustre5/${USER}/Canopy-opt100/
CANOPY_BUILD=/g/g20/stewartj/research-bridges/Canopy/build

echo "Loading spack environment"
# source ${SPACK_INSTALL}/share/spack/setup-env.sh
# spack env activate ${BEATNIK_ENV} 

# echo "Creating output directory"
rm -rf ${CANOPY_SCRATCH}
mkdir -p ${CANOPY_SCRATCH}
cd ${CANOPY_SCRATCH}
cp -a ${CANOPY_BUILD}/. .

# Make sure cray mpich supports GPU-aware communication
export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
echo "Starting MPI run with 8 processes"

flux run --ntasks=8 --nodes=2 --exclusive --gpus-per-task=1 --cores-per-task=8 --setopt=mpibind=verbose:1 rocprofv3 --kokkos-trace --output-format csv -- tests/Canopy_Test_Solver_MPI_HIP

echo "Finished tests"
