############################################################################
# Copyright (c) 2025 by the Canopy authors                                 #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Canopy library. Canopy is distributed under a   #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

# the next for lines are just, so that we can use this as
# a standalone CMakeLists.txt for testing of the Target export
cmake_minimum_required(VERSION 3.16)
project(Canopy_Development)
if(NOT TARGET Canopy)
  find_package(Canopy)
endif()

add_executable(example_fmm example_full_fmm.cpp)
target_link_libraries(example_fmm Canopy)
target_include_directories(example_fmm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
add_test(NAME Canopy_Tutorial_02_full_fmm COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:example_fmm> ${MPIEXEC_POSTFLAGS})
set_tests_properties(Canopy_Tutorial_02_full_fmm PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
if(Canopy_INSTALL_EXAMPLES)
      install(TARGETS example_fmm
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
