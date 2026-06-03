#------------------------------------------------------------------------------#
# Project settings
#
# This cmake organization, macros, and setup is copied from the
# Cabana library: https://github.com/ECP-copa/Cabana/
#------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.16)

project(Canopy LANGUAGES CXX)
set(PROJECT_VERSION "0.1.0-dev")

# If the user doesn't provide a build type default to release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

# Disable verbose makefiles
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)

# use gnu standard install directories
include(GNUInstallDirs)
set(Canopy_INSTALL_PACKAGEDIR "${CMAKE_INSTALL_DATADIR}/cmake/Canopy" CACHE PATH "Install location of CMake target files")

include(FeatureSummary)

# add local cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# find kokkos
find_package(Kokkos 4 REQUIRED)

# set supported kokkos devices
set(CANOPY_SUPPORTED_DEVICES SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET)

# check user required kokkos device types
foreach(_device ${CANOPY_SUPPORTED_DEVICES})
  option(Canopy_REQUIRE_${_device} "Build Canopy with required Kokkos ${device} support" OFF)
  if(Canopy_REQUIRE_${_device})
    kokkos_check( DEVICES ${_device} )
  endif()
endforeach()

# ensure that we can use lambdas
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

# find cabana
find_package(Cabana REQUIRED COMPONENTS Cabana::Grid Cabana::Core)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_GRID )
  message( FATAL_ERROR "Cabana must be compiled with Grid support" )
endif()

# standard dependency macro
macro(Canopy_add_dependency)
  cmake_parse_arguments(CANOPY_DEPENDENCY "" "PACKAGE;VERSION" "COMPONENTS" ${ARGN})
  find_package( ${CANOPY_DEPENDENCY_PACKAGE} ${CANOPY_DEPENDENCY_VERSION} QUIET COMPONENTS ${CANOPY_DEPENDENCY_COMPONENTS} )
  string(TOUPPER "${CANOPY_DEPENDENCY_PACKAGE}" CANOPY_DEPENDENCY_OPTION )
  option(
    Canopy_REQUIRE_${CANOPY_DEPENDENCY_OPTION}
    "Require Canopy to build with ${CANOPY_DEPENDENCY_PACKAGE} support" ${CANOPY_DEPENDENCY_PACKAGE}_FOUND)
  if(Canopy_REQUIRE_${CANOPY_DEPENDENCY_OPTION})
    find_package( ${CANOPY_DEPENDENCY_PACKAGE} ${CANOPY_DEPENDENCY_VERSION} REQUIRED COMPONENTS ${CANOPY_DEPENDENCY_COMPONENTS} )
  endif()
  set(Canopy_ENABLE_${CANOPY_DEPENDENCY_OPTION} ${${CANOPY_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

# find trilinos
Canopy_add_dependency( PACKAGE Trilinos )
set_package_properties(Trilinos PROPERTIES TYPE REQUIRED PURPOSE "Used for load balancing")

# find MPI
Canopy_add_dependency( PACKAGE MPI COMPONENTS CXX )
set_package_properties(MPI PROPERTIES TYPE REQUIRED PURPOSE "Used for distributed parallelization")

# find Clang Format
find_package( CLANG_FORMAT 14 )

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Canopy_BUILD_MARCH "" CACHE STRING "Arch to use with -march= (if empty CMake will try to use 'native') in release build and only release build")

  # Try -march first. On platforms that don't support it, GCC will issue
  # a hard error, so we'll know not to use it.
  if(Canopy_BUILD_MARCH)
    set(INTERNAL_Canopy_BUILD_MARCH ${Canopy_BUILD_MARCH})
  else()
    set(INTERNAL_Canopy_BUILD_MARCH "native")
  endif()

  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=${INTERNAL_Canopy_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
  if(COMPILER_SUPPORTS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_Canopy_BUILD_MARCH}")
  elseif(Canopy_BUILD_MARCH)
    message(FATAL_ERROR "The flag -march=${INTERNAL_Canopy_BUILD_MARCH} is not supported by the compiler")
  else()
    unset(INTERNAL_Canopy_BUILD_MARCH)
  endif()
endif()

##---------------------------------------------------------------------------##
## DevFast: minimal optimization for fast compile during active development.
##
## Targets -O0 to skip nvcc optimization passes (the dominant per-TU cost
## with nvcc_wrapper) while keeping -g so backtraces remain useful. Not for
## profiling or performance work — use RelWithDebInfo for that.
##---------------------------------------------------------------------------##
set(CMAKE_CXX_FLAGS_DEVFAST "-O0 -g" CACHE STRING
    "Flags used by the CXX compiler during DevFast builds." FORCE)
set(CMAKE_C_FLAGS_DEVFAST "-O0 -g" CACHE STRING
    "Flags used by the C compiler during DevFast builds." FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEVFAST "" CACHE STRING
    "Flags used for linking binaries during DevFast builds." FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEVFAST "" CACHE STRING
    "Flags used by the shared libraries linker during DevFast builds." FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_DEVFAST
  CMAKE_C_FLAGS_DEVFAST
  CMAKE_EXE_LINKER_FLAGS_DEVFAST
  CMAKE_SHARED_LINKER_FLAGS_DEVFAST)
if(CMAKE_BUILD_TYPE STREQUAL "DevFast")
  message(STATUS "DevFast CXX flags: ${CMAKE_CXX_FLAGS_DEVFAST}")
endif()

##---------------------------------------------------------------------------##
## RelWithDebInfo: profiling-friendly flags
##
## Targets -O3 (keep optimized code paths) with host -g (already in the
## CMake default) and -lineinfo on device code so Nsight Compute can map
## hardware counters back to source lines.
##---------------------------------------------------------------------------##
if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  string(REPLACE "-O2" "-O3"
    CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
  if(Kokkos_ENABLE_CUDA)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
      "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -lineinfo")
  endif()
  message(STATUS "RelWithDebInfo CXX flags: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

##---------------------------------------------------------------------------##
## Code coverage testing
##---------------------------------------------------------------------------##
option(Canopy_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(Canopy_ENABLE_COVERAGE_BUILD)
  message(STATUS "Enabling coverage build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  option(WITH_ASAN "Build with address sanitizer" OFF)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
  option(WITH_MSAN "Build with memory sanitizer (experimental; requires a memory-sanitized Python interpreter)" OFF)
endif()

if(WITH_ASAN AND WITH_MSAN)
  message( FATAL_ERROR "Address sanitizer and memory sanitizer cannot be enabled simultaneously")
endif()
if(WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()

if(WITH_MSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=memory")
endif()

##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  EXECUTE_PROCESS(
    COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE   Canopy_GIT_COMMIT_HASH
    )
ELSE()
  SET(Canopy_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "Canopy Revision = '${Canopy_GIT_COMMIT_HASH}'")

#------------------------------------------------------------------------------#
# Tests and Documentation
#------------------------------------------------------------------------------#
# enable unit tests
option(Canopy_ENABLE_TESTING "Build tests" OFF)
add_feature_info(Tests Canopy_ENABLE_TESTING "Build unit tests (requires GTest)")

# Set installing tests to off by default
option(Canopy_INSTALL_TEST_EXECUTABLES "Install Canopy test executables" OFF)

if(Canopy_ENABLE_TESTING)
  find_package(GTest 1.10 REQUIRED)
  # Workaround for FindGTest module in CMake older than 3.20
  if(TARGET GTest::gtest)
    set(gtest_target GTest::gtest)
  elseif(TARGET GTest::GTest)
    set(gtest_target GTest::GTest)
  else()
    message(FATAL_ERROR "bug in GTest find module workaround")
  endif()
  set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/cmake/test_harness)
  include(cmake/test_harness/test_harness.cmake)
  add_subdirectory(tests)
  enable_testing()
endif()

# enable doxygen
option(Canopy_ENABLE_DOXYGEN "Build documentation" OFF)
add_feature_info(Documentation Canopy_ENABLE_DOXYGEN "Build documentation (requires Doxygen)")
if(Canopy_ENABLE_DOXYGEN)
  find_package(Doxygen REQUIRED)
  doxygen_add_docs(doxygen src src)
endif()

##---------------------------------------------------------------------------##
## Diagnostics / profiling timers
##---------------------------------------------------------------------------##
option(Canopy_ENABLE_PROFILING "Profiling" OFF)
set(Canopy_PROFILING_LEVEL "" CACHE STRING
    "Profiling detail level: 0=off, 1=basic phases, 2=detailed sub-phases, 3=verbose (reserved)")
set_property(CACHE Canopy_PROFILING_LEVEL PROPERTY STRINGS "" 0 1 2 3)

# Resolve the effective profiling level. If Canopy_ENABLE_PROFILING is ON and
# no explicit level was given, default to level 1 (the original behavior).
# Canopy_ENABLE_PROFILING=OFF is an authoritative kill switch: it forces the
# level to 0 regardless of any Canopy_PROFILING_LEVEL the user supplied.
if(NOT Canopy_ENABLE_PROFILING AND NOT Canopy_PROFILING_LEVEL STREQUAL "")
  set(_canopy_profiling_level 0)
elseif(Canopy_PROFILING_LEVEL STREQUAL "")
  if(Canopy_ENABLE_PROFILING)
    set(_canopy_profiling_level 1)
  else()
    set(_canopy_profiling_level 0)
  endif()
else()
  set(_canopy_profiling_level ${Canopy_PROFILING_LEVEL})
endif()

if(_canopy_profiling_level GREATER 0)
  set(Canopy_ENABLE_PROFILING ON CACHE BOOL "Profiling" FORCE)
  add_compile_definitions(CANOPY_ENABLE_PROFILING)
  add_compile_definitions(CANOPY_PROFILING_LEVEL=${_canopy_profiling_level})
endif()

add_feature_info(Profiling Canopy_ENABLE_PROFILING
    "Enable FMM phase timing via MPI_Wtime (level=${_canopy_profiling_level})")

##---------------------------------------------------------------------------##
## Debug-only correctness checks
##
## Enables in-code side-by-side verification passes (e.g. the incremental
## ilist rebuild's full-rebuild cross-check) that double the work in the
## checked path. Off by default and on production / at-scale builds.
##---------------------------------------------------------------------------##
option(Canopy_ENABLE_DEBUG "Debug correctness checks" OFF)
if(Canopy_ENABLE_DEBUG)
  add_compile_definitions(CANOPY_ENABLE_DEBUG)
endif()
add_feature_info(Debug Canopy_ENABLE_DEBUG
    "Enable in-code side-by-side correctness checks (significant overhead)")

##---------------------------------------------------------------------------##
## Libraries and Examples
##---------------------------------------------------------------------------##

if(MPI_FOUND)
  add_subdirectory(src)
else()
  message(FATAL_ERROR "Canopy requires MPI")
endif()

option(Canopy_ENABLE_EXAMPLES "Build tutorial examples" OFF)
# Set installing examples to off by default
option(Canopy_INSTALL_EXAMPLES "Install Canopy example executables" OFF)
add_feature_info(Examples Canopy_ENABLE_EXAMPLES "Build tutorial examples")
if(Canopy_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

##---------------------------------------------------------------------------##
## Package Configuration
##---------------------------------------------------------------------------##
write_basic_package_version_file("CanopyConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CanopyConfig.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/CanopyConfig.cmake @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CanopyConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/CanopyConfigVersion.cmake"
  DESTINATION ${Canopy_INSTALL_PACKAGEDIR})

##---------------------------------------------------------------------------##
## Clang Format
##---------------------------------------------------------------------------##
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp examples/*cpp examples/*hpp cmake/*cpp cmake/*hpp)
  add_custom_target(cabana-format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()

feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)
