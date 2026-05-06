configure_file(Canopy_Config.hpp.cmakein Canopy_Config.hpp)

set(HEADERS_PUBLIC
  Canopy_BatchedGemm.hpp
  Canopy_CommunicationPlan.hpp
  Canopy_DownwardSweep.hpp
  Canopy_Helpers.hpp
  Canopy_LaplaceKernel.hpp
  Canopy_P2P.hpp
  Canopy_Profiling.hpp
  Canopy_Solver.hpp
  Canopy_SphericalCoefficients.hpp
  Canopy_TreeBuilder.hpp
  Canopy_TreePartitioner.hpp
  Canopy_UpwardSweep.hpp
  )

# Currently no impl headers
# set(HEADERS_IMPL
#   impl/Canopy_CartesianGrid.hpp
#   )

add_library(Canopy INTERFACE)

if(CANOPY_ENABLE_PROFILING)
  target_compile_definitions(Canopy INTERFACE CANOPY_ENABLE_PROFILING)
endif()

# Require minimum of C++17
set_target_properties(Canopy PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_17)

target_link_libraries(Canopy INTERFACE Kokkos::kokkos)

target_link_libraries(Canopy INTERFACE MPI::MPI_CXX)

target_link_libraries(Canopy INTERFACE Cabana::Core Cabana::Grid)

target_link_libraries(Canopy INTERFACE ${Trilinos_LIBRARIES})

if(Canopy_ENABLE_CUBLAS)
  target_link_libraries(Canopy INTERFACE CUDA::cublas)
  target_compile_definitions(Canopy INTERFACE CANOPY_ENABLE_CUBLAS)
endif()

if(Canopy_ENABLE_HIPBLAS)
  target_link_libraries(Canopy INTERFACE roc::hipblas)
  target_compile_definitions(Canopy INTERFACE CANOPY_ENABLE_HIPBLAS)
endif()

if(Canopy_ENABLE_KOKKOSKERNELS)
  target_link_libraries(Canopy INTERFACE Kokkos::kokkoskernels)
  target_compile_definitions(Canopy INTERFACE CANOPY_ENABLE_KOKKOSKERNELS)
endif()

target_include_directories(Canopy INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${Trilinos_INCLUDE_DIRS}
  )

install(TARGETS Canopy
  EXPORT Canopy_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT Canopy_Targets
  FILE Canopy_Targets.cmake
  NAMESPACE Canopy::
  DESTINATION ${Canopy_INSTALL_PACKAGEDIR})

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${HEADERS_IMPL} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/impl/)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Canopy_Config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)