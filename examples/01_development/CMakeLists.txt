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

# the next for lines are just, so that we can use this as
# a standalone CMakeLists.txt for testing of the Target export
cmake_minimum_required(VERSION 3.16)
project(Canopy_Development)
if(NOT TARGET Canopy)
  find_package(Canopy)
endif()

add_executable(Development development.cpp)
target_link_libraries(Development Canopy)
add_test(NAME Canopy_Tutorial_01_development COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:Development> ${MPIEXEC_POSTFLAGS})
set_tests_properties(Canopy_Tutorial_01_development PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
