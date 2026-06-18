############################################################################
# TEMPORARY (debug-nan branch): offline replay harness for the premature    #
# full-rollup FMM NaN. Build out-of-tree against the installed Canopy:       #
#                                                                            #
#   spack env activate ~/spack_envs/tuolumne_beatnik                         #
#   cmake -B build -S examples/04_nan_replay \                               #
#         -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.2/bin/hipcc                      #
#   cmake --build build                                                      #
#                                                                            #
# Uses the INSTALLED Canopy headers (Canopy::Canopy carries all transitive   #
# includes/links). For Step-3 iteration on Canopy headers, `spack install`   #
# canopy first (~11 s), then rebuild this harness.                           #
#                                                                            #
# Remove this example when the NaN is resolved.                              #
############################################################################

cmake_minimum_required(VERSION 3.16)
project(Canopy_NanReplay CXX)

find_package(Canopy REQUIRED)
find_package(Cabana REQUIRED)   # pulls ArborX include dirs the Canopy export misses
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(nan_replay nan_replay.cpp)
# Canopy::Canopy carries the installed Canopy headers + Trilinos/Zoltan2;
# Cabana::Core completes the ArborX include dirs; Kokkos for the backend.
target_link_libraries(nan_replay Canopy::Canopy Cabana::Core Kokkos::kokkos MPI::MPI_CXX)
target_include_directories(nan_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
# The spack env view symlink farm can lag newly-added Canopy headers (e.g.
# Canopy_RegisteredBufferPool.hpp), which breaks quote-includes resolved via
# the view. Add the canopy store include dir directly so they resolve. Pass:
#   -DCANOPY_INC=$(spack location -i canopy)/include
if(DEFINED CANOPY_INC)
  target_include_directories(nan_replay BEFORE PRIVATE ${CANOPY_INC})
endif()
