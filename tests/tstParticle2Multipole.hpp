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

#include <Canopy_Tree.hpp>

#include <test_helper_functions.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//

/**
 * Rank 0 creates all the data. Each rank gets two particles.
 * All ranks insert into the tree.
 * Tests that data was correctly distributed to the rank that owns it
 * and correctly inserted into the tree and converted into multipole coefficients
 * at the leaf layer.
 */
template <std::size_t p_val>
void testParticle2Multipole(bool balanced)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create a tree of at least depth 3 for any number of processes
    using particle_tuple_type = Cabana::MemberTypes<double[3], double>;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
    std::array<double, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<double, 3> global_high_corner = { 3.0, 3.0, 3.0 };
    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 2;
    static constexpr std::size_t p = p_val;
    std::size_t leaf_tiles, red_factor;
    red_factor = comm_size / 2, leaf_tiles = comm_size * 4;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, Cabana::Grid::Cell,
        num_dim, cells_per_tile, p>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, MPI_COMM_WORLD);
    
    // The tree depth should always be at least three, but this check is here just in case.
    // If the depth is less than 3, this test may not work correctly.
    // if (rank == 0) printf("R%d: num tree layers: %d\n", rank, tree->numLayers());
    // ASSERT_GE(tree->numLayers(), 3) << "testUpwardsAggregation: Error: Tree depth must be at least 3.\n";
    
    // Create the data
    int num_points = (rank == 0) ? (comm_size * 500) : 0;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );
    
    double bound_val = 3.0;
    Kokkos::Array<double, 6> coord_bounds = {-bound_val, -bound_val, -bound_val, bound_val, bound_val, bound_val};
    // If not balanced, fill domain unevenly
    if (!balanced)
    {
        coord_bounds = {-2.8, 0.3, -0.2, -0.5, 3.0, 1.3};
    }

    fillRandomCoordinates(cart_coords, coord_bounds, 123);
    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    fillRandomScalar(q, charge_bounds, 123);

    Cabana::AoSoA<particle_tuple_type, Kokkos::HostSpace, 4> particle_aosoa_host("particle_aosoa", num_points);
    auto pos_slice_host = Cabana::slice<0>(particle_aosoa_host);
    auto scalar_slice_host = Cabana::slice<1>(particle_aosoa_host);

    // Fill the particles into the AoSoA
    auto cart_coords_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cart_coords);
    auto q_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);
    for (int i = 0; i < num_points; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            pos_slice_host(i, j) = cart_coords_h(i, j);
        }
        scalar_slice_host(i) = q_h(i);
    }

    // Calculate direct potential.
    // Target point far away from domain so multipole approximation holds.
    double Px = 15.1, Py = -20.3, Pz = 16.2;
    double r, theta, phi;
    Canopy::Kernel::cart2sph( Px, Py, Pz, r, theta, phi );
    double potential_direct = 0.0;
    for (std::size_t i = 0; i < num_points; ++i)
    {
        double dx = Px - pos_slice_host( i, 0 );
        double dy = Py - pos_slice_host( i, 1 );
        double dz = Pz - pos_slice_host( i, 2 );
        double dist = std::sqrt( dx * dx + dy * dy + dz * dz );
        potential_direct += scalar_slice_host( i ) / dist;
    }

    // Broadcast direct potential to all other ranks.
    MPI_Bcast(&potential_direct, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Copy to device
    auto particle_aosoa =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // Fill the tree
    bool run_load_balance = !balanced;
    tree->create_multipoles(particle_aosoa, run_load_balance);

    /***********************************************
     * Check the data in the root layer
     **********************************************/
    auto m_root_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), tree->M_root() );

    // Compute potential at P using M
    Kokkos::complex<double> potential_M = 0.0;
    for ( int j = 0; j <= p; ++j )
    {
        for ( int k = -j; k <= j; ++k )
        {
            int idx = Canopy::Kernel::Scalar::index( j, k );
            potential_M +=
                m_root_h( idx ) / Kokkos::pow( r, j + 1 ) *
                Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
        }
    }

    // Check error between translated multipole and direct potentials
    // The error is already mathematically checked in testM2MKernel0,
    // so here we just make sure they are close to each other.
    int p_int = p;
    double error = Kokkos::pow(10, -p_int);
    EXPECT_NEAR(potential_direct, potential_M.real(), error) << "p="
        << p << ": Potentials do not match. Tree depth " << tree->numLayers();
    // printf("R%d: potential: %0.8lf, M: %0.8lf\n", rank, potential_direct, potential_M.real());
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

// Test accuracy with increasing truncation cutoffs of multipole coefficients.
// Test with a balanced particle distribution.
TEST( Tree, testParticle2Multipole1_balanced ) { testParticle2Multipole<1>(true); }
TEST( Tree, testParticle2Multipole2_balanced ) { testParticle2Multipole<2>(true); }
TEST( Tree, testParticle2Multipole3_balanced ) { testParticle2Multipole<3>(true); }
TEST( Tree, testParticle2Multipole4_balanced ) { testParticle2Multipole<4>(true); }

// Test with an unbalanced particle distribution.
TEST( Tree, testParticle2Multipole1_unbalanced ) { testParticle2Multipole<1>(false); }
TEST( Tree, testParticle2Multipole2_unbalanced ) { testParticle2Multipole<2>(false); }
TEST( Tree, testParticle2Multipole3_unbalanced ) { testParticle2Multipole<3>(false); }
TEST( Tree, testParticle2Multipole4_unbalanced ) { testParticle2Multipole<4>(false); }

//---------------------------------------------------------------------------//

} // end namespace Test