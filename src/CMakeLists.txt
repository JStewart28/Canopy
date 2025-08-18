configure_file(Canopy_Config.hpp.cmakein Canopy_Config.hpp)

set(HEADERS_PUBLIC
  Canopy_AoSoA.hpp
  )


if(Canopy_ENABLE_SILO)
  list(APPEND HEADERS_PUBLIC Canopy_SiloParticleOutput.hpp)
endif()

if(Canopy_ENABLE_HDF5)
  list(APPEND HEADERS_PUBLIC Canopy_HDF5ParticleOutput.hpp)
endif()

set(HEADERS_IMPL
  impl/Canopy_CartesianGrid.hpp
  impl/Canopy_Index.hpp
  impl/Canopy_PerformanceTraits.hpp
  impl/Canopy_TypeTraits.hpp
  )

add_library(Canopy INTERFACE)

# Require minimum of C++17
set_target_properties(Canopy PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_17)

target_link_libraries(Canopy INTERFACE Kokkos::kokkos)

if(Canopy_ENABLE_ARBORX)
  target_link_libraries(Canopy INTERFACE ArborX::ArborX)
endif()

if(Canopy_ENABLE_SILO)
  target_link_libraries(Canopy INTERFACE SILO::silo)
endif()

if(Canopy_ENABLE_HDF5)
  # FIXME: remove when requiring newer CMake
  if(CMAKE_VERSION VERSION_LESS 3.26)
    target_link_libraries(Canopy INTERFACE ${HDF5_LIBRARIES})
  else()
    target_link_libraries(Canopy INTERFACE HDF5::HDF5)
  endif()
endif()

if(Canopy_ENABLE_MPI)
  target_link_libraries(Canopy INTERFACE MPI::MPI_CXX)
endif()

target_include_directories(Canopy INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
if(Canopy_ENABLE_HDF5)
  target_include_directories(Canopy INTERFACE
    ${HDF5_INCLUDE_DIRS} # FIXME: remove when requiring newer CMake
  )
endif()

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