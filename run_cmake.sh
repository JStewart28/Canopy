cmake -DCMAKE_CXX_COMPILER=$(spack location -i kokkos)/bin/nvcc_wrapper \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCanopy_ENABLE_TESTING=ON \
      -DCanopy_ENABLE_EXAMPLES=ON ..