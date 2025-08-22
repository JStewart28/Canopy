############################################################################
# Copyright (c) 2025 by the Canopy authors                            #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Canopy library. Canopy is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

include(FindPackageHandleStandardArgs)
find_program(VALGRIND_EXECUTABLE valgrind)
find_package_handle_standard_args(VALGRIND REQUIRED_VARS VALGRIND_EXECUTABLE)
if(VALGRIND_FOUND)
  set(VALGRIND_ARGS --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes --error-exitcode=1)
endif()

##--------------------------------------------------------------------------##
## On-node tests with and without MPI.
##--------------------------------------------------------------------------##
set(CANOPY_TEST_DEVICES)
foreach(_device ${CANOPY_SUPPORTED_DEVICES})
  if(Kokkos_ENABLE_${_device})
    list(APPEND CANOPY_TEST_DEVICES ${_device})
    if(_device STREQUAL CUDA)
      list(APPEND CANOPY_TEST_DEVICES CUDA_UVM)
    endif()
  endif()
endforeach()

macro(Canopy_add_tests)
  cmake_parse_arguments(CANOPY_UNIT_TEST "MPI" "PACKAGE" "NAMES" ${ARGN})
  set(CANOPY_UNIT_TEST_MPIEXEC_NUMPROCS 1)
  foreach( _np 2 4 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_np})
      list(APPEND CANOPY_UNIT_TEST_MPIEXEC_NUMPROCS ${_np})
    endif()
  endforeach()
  if(MPIEXEC_MAX_NUMPROCS GREATER 4)
    list(APPEND CANOPY_UNIT_TEST_MPIEXEC_NUMPROCS ${MPIEXEC_MAX_NUMPROCS})
  endif()
  set(CANOPY_UNIT_TEST_NUMTHREADS 1)
  foreach( _nt 2 4 )
    if(MPIEXEC_MAX_NUMPROCS GREATER_EQUAL ${_nt})
      list(APPEND CANOPY_UNIT_TEST_NUMTHREADS ${_nt})
    endif()
  endforeach()
  if(CANOPY_UNIT_TEST_MPI)
    set(CANOPY_UNIT_TEST_MAIN ${TEST_HARNESS_DIR}/mpi_unit_test_main.cpp)
  else()
    set(CANOPY_UNIT_TEST_MAIN ${TEST_HARNESS_DIR}/unit_test_main.cpp)
  endif()
  foreach(_device ${CANOPY_TEST_DEVICES})
    set(_dir ${CMAKE_CURRENT_BINARY_DIR}/${_device})
    file(MAKE_DIRECTORY ${_dir})
    foreach(_test ${CANOPY_UNIT_TEST_NAMES})
      set(_file ${_dir}/tst${_test}_${_device}.cpp)
      file(WRITE ${_file}
        "#include <Test${_device}_Category.hpp>\n"
        "#include <tst${_test}.hpp>\n"
      )
      if(CANOPY_UNIT_TEST_MPI)
        set(_target Canopy_${CANOPY_UNIT_TEST_PACKAGE}_Test_${_test}_MPI_${_device})
      else()
        set(_target Canopy_${CANOPY_UNIT_TEST_PACKAGE}_Test_${_test}_${_device})
      endif()
      add_executable(${_target} ${_file} ${CANOPY_UNIT_TEST_MAIN})
      target_include_directories(${_target} PRIVATE ${_dir}
        ${TEST_HARNESS_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
      target_link_libraries(${_target} PRIVATE ${CANOPY_UNIT_TEST_PACKAGE} ${gtest_target})
      if(CANOPY_UNIT_TEST_MPI)
        foreach(_np ${CANOPY_UNIT_TEST_MPIEXEC_NUMPROCS})
          add_test(NAME ${_target}_np_${_np} COMMAND
            ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${_target}> ${MPIEXEC_POSTFLAGS} ${gtest_args})
          set_property(TEST ${_target}_np_${_np} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
        endforeach()
      else()
        if(_device STREQUAL THREADS OR _device STREQUAL OPENMP)
          foreach(_thread ${CANOPY_UNIT_TEST_NUMTHREADS})
            add_test(NAME ${_target}_nt_${_thread} COMMAND
                    ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args} --kokkos-num-threads=${_thread})
            if(_device STREQUAL OPENMP)
              set_property(TEST ${_target}_nt_${_thread} PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_thread})
            endif()
            if(VALGRIND_FOUND)
              add_test(NAME ${_target}_nt_${_thread}_valgrind COMMAND
                ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args} --kokkos-num-threads=${_thread})
              if(_device STREQUAL OPENMP)
                set_property(TEST ${_target}_nt_${_thread}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=${_thread})
              endif()
            endif()
          endforeach()
        else()
          add_test(NAME ${_target} COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:${_target}> ${gtest_args})
          set_property(TEST ${_target} PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
          if(VALGRIND_FOUND)
            add_test(NAME ${_target}_valgrind COMMAND ${NONMPI_PRECOMMAND} ${VALGRIND_EXECUTABLE} ${VALGRIND_ARGS} $<TARGET_FILE:${_target}> ${gtest_args})
            set_property(TEST ${_target}_valgrind PROPERTY ENVIRONMENT OMP_NUM_THREADS=1)
          endif()
        endif()
      endif()
      if(Canopy_INSTALL_TEST_EXECUTABLES)
        install(TARGETS ${_target}
                RUNTIME DESTINATION ${CMAKE_INSTALL_DATADIR}/Canopy/tests)
      endif()
    endforeach()
  endforeach()
endmacro()
