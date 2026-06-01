cmake -DCMAKE_CXX_COMPILER=$(spack location -i kokkos)/bin/nvcc_wrapper \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCanopy_ENABLE_TESTING=ON \
      -DCanopy_ENABLE_EXAMPLES=ON \
      -DCanopy_ENABLE_PROFILING=ON \
      -DCanopy_PROFILING_LEVEL=3 ..

# for i in {1..6}; do mpirun -np $i ./tests/Canopy_Test_MultiSolve_MPI_SERIAL; done
# for i in {1..6}; do mpirun -np $i ./tests/Canopy_Test_MultiSolve_MPI_CUDA; done