/****************************************************************************
 * Copyright (c) 2018-2023 by the Canopy authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Canopy library. Canopy is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Canopy_Tree.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

/**
 * Test that data is correctly inserted into the tree at the leaf layer 
 * when there is only one particle per cell
 */
template <class DataAoSoA>
void checkOnePerCellLeaf(DataAoSoA input_data)
{

}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
// TEST( AoSoA, Test ) { testAoSoA(); }

//---------------------------------------------------------------------------//

} // end namespace Test