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
project(Canopy_RollupNearfield)
if(NOT TARGET Canopy)
  find_package(Canopy)
endif()

add_executable(rollup_nearfield rollup_nearfield.cpp)
target_link_libraries(rollup_nearfield Canopy)
target_include_directories(rollup_nearfield PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
# Cheap smoke run: small N, few steps — just exercises the code path. The real
# measurement is a -k sweep driven by scripts/tuolumne/rollup_sweep.flux.
add_test(NAME Canopy_Tutorial_05_rollup_nearfield COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:rollup_nearfield> ${MPIEXEC_POSTFLAGS}
-p 2000 -t 8)
set_tests_properties(Canopy_Tutorial_05_rollup_nearfield PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
if(Canopy_INSTALL_EXAMPLES)
      install(TARGETS rollup_nearfield
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
