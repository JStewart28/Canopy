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

# Profiling-only drivers (scripts/tuolumne/scaling/). Not CTest tests.
# scaling_breakdown uses LaplaceKernel; scaling_breakdown_ct the same source
# with CartesianTaylorBasis.
if(Canopy_ENABLE_PROFILING)
  add_executable(scaling_breakdown scaling_breakdown.cpp)
  target_link_libraries(scaling_breakdown Canopy)
  target_include_directories(scaling_breakdown PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

  add_executable(scaling_breakdown_ct scaling_breakdown.cpp)
  target_link_libraries(scaling_breakdown_ct Canopy)
  target_include_directories(scaling_breakdown_ct PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
  target_compile_definitions(scaling_breakdown_ct PRIVATE CANOPY_SCALING_CT)
endif()
