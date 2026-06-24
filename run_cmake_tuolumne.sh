cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER=CC \
      -DCMAKE_C_COMPILER=cc \
      -DCMAKE_HIP_COMPILER=amdclang++ \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCanopy_ENABLE_TESTING=ON \
      -DCanopy_ENABLE_EXAMPLES=ON \
      -DCanopy_ENABLE_PROFILING=ON \
      -DCanopy_PROFILING_LEVEL=2 \
      -DMPIEXEC_EXECUTABLE=$(which flux) \
      "-DMPIEXEC_NUMPROC_FLAG=run;--ntasks" \
      "-DMPIEXEC_PREFLAGS=--nodes=1;--exclusive;--cores-per-task=1" ..

# MPIEXEC_* overrides make CTest launch each MPI test via `flux run` with the
# same resource binding the by-hand scripts use:
#   flux run --ntasks N --nodes=1 --exclusive --cores-per-task=1 <exe>
# Without them CMake's FindMPI auto-detects the flux_wrappers `srun`, which runs
# with no core binding and deadlocks at >=3 ranks (the binaries init the HIP
# backend at Kokkos::initialize even for SERIAL tests, and unbound ranks contend
# on the single MI300A APU). With the overrides, `ctest` is the single entry
# point for the test suite. GPU/backend env (HSA_XNACK, MPICH_GPU_*, the
# static-TLS workaround) still belongs in the batch script that invokes ctest.
