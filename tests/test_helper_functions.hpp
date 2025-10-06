/****************************************************************************
 * Copyright (c) 2025 by the Canopy authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Canopy library. Canopy is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

/**
 * Fill a view with random (x, y, z) coordinates within the specified bounds,
 * where bounds is (x_min, y_min, z_min, x_max, y_max, z_max)
 */
template <class PosView>
void fillRandomCoordinates(PosView& cart_coords, Kokkos::Array<double, 6> bounds)
{
    using RandomPool = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    RandomPool rand_pool( 12345 ); // Seed the random number generator
    Kokkos::parallel_for(
        "populate_cart_coords",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, cart_coords.extent(0) ),
        KOKKOS_LAMBDA( const int i ) {
            auto rand_gen = rand_pool.get_state();
            
            // X-coordinate
            double min_x = bounds[0];
            double max_x = bounds[3];
            cart_coords( i, 0 ) = (max_x - min_x) * rand_gen.drand() + min_x;

            // Y-coordinate
            double min_y = bounds[1];
            double max_y = bounds[4];
            cart_coords( i, 1 ) = (max_y - min_y) * rand_gen.drand() + min_y;
            
            // Z-coordinate
            double min_z = bounds[2];
            double max_z = bounds[5];
            cart_coords( i, 2 ) = (max_z - min_z) * rand_gen.drand() + min_z;
            
            rand_pool.free_state( rand_gen );
        } );
    Kokkos::fence();
}

/**
 * Fill a view with random scalar values within the specified (min, max) bound.
 */
template <class View>
void fillRandomScalar(View& q, Kokkos::Array<double, 2> bounds)
{
    using RandomPool = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    RandomPool rand_pool( 12345 ); // Seed the random number generator
    Kokkos::parallel_for(
        "populate_cart_coords",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, q.extent(0) ),
        KOKKOS_LAMBDA( const int i ) {
            auto rand_gen = rand_pool.get_state();
            // X-coordinate
            double min = bounds[0];
            double max = bounds[1];
            q( i ) = (max - min) * rand_gen.drand() + min;
            rand_pool.free_state( rand_gen );
        } );
    Kokkos::fence();
}

} // end namepace Test
