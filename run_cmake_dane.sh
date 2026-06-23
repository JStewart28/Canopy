# Dane (LLNL CTS-2, Sapphire Rapids). Run from an out-of-tree build dir
# (e.g. build-dane/) after:
#   module load gcc/13.3.1 openmpi/4.1.2
#   spack env activate ${HOME}/spack_envs/dane_trilinos
#
# CMAKE_CXX_COMPILER is the openmpi mpicxx wrapper (wraps g++ 13.3.1) so the
# compiler and MPI match the spack-built Trilinos (gcc@13.3.1 ^openmpi@4.1.2).
# gcc 13 is required: Dane's CPU is sapphirerapids and gcc 10 cannot target it.
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER=mpicxx \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCanopy_ENABLE_TESTING=ON \
      -DCanopy_ENABLE_EXAMPLES=ON \
      -DCanopy_ENABLE_PROFILING=ON \
      -DCanopy_PROFILING_LEVEL=0 ..

# for i in {1..6}; do srun -n $i ./tests/Canopy_Test_MultiSolve_MPI_SERIAL; done
