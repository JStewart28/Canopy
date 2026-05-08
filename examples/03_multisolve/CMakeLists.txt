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

cmake_minimum_required(VERSION 3.16)
project(Canopy_MultiSolve)
if(NOT TARGET Canopy)
  find_package(Canopy)
endif()

add_executable(multisolve example_multisolve.cpp)
target_link_libraries(multisolve Canopy)
add_test(NAME Canopy_Tutorial_03_multisolve COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:multisolve> ${MPIEXEC_POSTFLAGS})
set_tests_properties(Canopy_Tutorial_03_multisolve PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
if(Canopy_INSTALL_EXAMPLES)
      install(TARGETS multisolve
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
