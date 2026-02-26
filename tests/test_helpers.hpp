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

#include <Canopy_Solver.hpp>

namespace Test
{
//---------------------------------------------------------------------------//

// Define complex double type
using cdouble = Kokkos::complex<double>;

// Define input aosoa data
// pos/charge/potential/global particle id
using particle_tuple_type = Cabana::MemberTypes<double[3], double, double, int>;
using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
using particle_aosoa_type_h = Cabana::AoSoA<particle_tuple_type, Kokkos::HostSpace, 4>;
using MD = Canopy::ParticleMetadata<particle_aosoa_type, 0, 1, 2>; 

// Define input aosoa data including force
// pos/force/charge/potential/global particle id
using particle_tuple_type_f = Cabana::MemberTypes<double[3], double[3], double, double, int>;
using particle_aosoa_type_f = Cabana::AoSoA<particle_tuple_type_f, TEST_MEMSPACE, 4>;
using particle_aosoa_type_f_h = Cabana::AoSoA<particle_tuple_type_f, Kokkos::HostSpace, 4>;
using MD_f = Canopy::ParticleMetadata<particle_aosoa_type_f, 0, 2, 3, 1>; 

double distance(const Kokkos::Array<double,3>& a,
                const Kokkos::Array<double,3>& b)
{
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
}

/**
 * Fill a view with random (x, y, z) coordinates within the specified bounds,
 * where bounds is (x_min, y_min, z_min, x_max, y_max, z_max)
 */
template <class PosView>
void fillRandomCoordinates(PosView& cart_coords, Kokkos::Array<double, 6> bounds, int seed)
{
    using RandomPool = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    RandomPool rand_pool( seed );
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
void fillRandomScalar(View& q, Kokkos::Array<double, 2> bounds, int seed)
{
    using RandomPool = Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE>;
    RandomPool rand_pool( seed );
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
