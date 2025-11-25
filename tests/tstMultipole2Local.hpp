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

using cdouble = Kokkos::complex<double>;

/**
 * Tests that on a single layer, multipole are correctly converted to locals.
 * Process:
 *  1. Generate particles and create multipoles on a three-layer tree. Three layers
 *     smallest possible because the leaf layer has X tiles per dimension, layer 1
 *     has one tile per dimension, and layer 2 (root layer) has one cell per dimension.
 *  2. At layer 3, choose a position (root) at which to calculate potential directly,
 *     but only consider particles that are in cells in the root cell's interaction list.
 *  3. Calculate the potential using local approximation and compare values.
 */
template <std::size_t p_val>
void testMultipole2Local(bool balanced)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create a tree of depth 3.
    using particle_tuple_type = Cabana::MemberTypes<double[3], double>;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
    std::array<double, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<double, 3> global_high_corner = { 3.0, 3.0, 3.0 };

    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 2;
    static constexpr std::size_t p = p_val;
    std::size_t leaf_tiles, red_factor;
    red_factor = comm_size * 8, leaf_tiles = comm_size * 8;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, Cabana::Grid::Cell,
        num_dim, cells_per_tile, p>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, MPI_COMM_WORLD);
    
    // The tree depth should always be at least three, but this check is here just in case.
    // If the depth is less than 3, this test may not work correctly.
    // if (rank == 0) printf("R%d: num tree layers: %d\n", rank, tree->numLayers());
    ASSERT_EQ(tree->numLayers(), 3) << "testMultipole2Local: Error: Tree depth must be depth 3.";

    // Check mesh information for leaf layer (layer 0)
    auto layer = tree->layer(0);
    int cells_per_leaf_dimension = cells_per_tile * leaf_tiles;
    Kokkos::Array<double, 3> cell_size;
    for (int i = 0; i < 3; ++i)
    {
        cell_size[i] = (global_high_corner[i] - global_low_corner[i]) / cells_per_leaf_dimension;
    }
    ASSERT_EQ(layer->cellsPerDim(), cells_per_leaf_dimension) << "testMultipole2Local: Error: Unexpected cells_per_leaf_dimension";
    ASSERT_EQ(layer->tilesPerDim(), leaf_tiles) << "testMultipole2Local: Error: Unexpected leaf_tiles";
    ASSERT_EQ(layer->cellSize(), cell_size) << "testMultipole2Local: Error: Unexpected cell_size";

    // Create the data on rank 0. It will automatically be distributed correctly when
    // filled into the tree. There must be enough particles so that the target point resides
    // in a cell that has been activated in the mesh. This won't be a problem in the
    // "real" code because we only evaluate locals where cells are activated.
    int points_per_proc = 100;
    int num_points = (rank == 0) ? (comm_size * points_per_proc) : 0;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          num_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", num_points );
    
    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    double bound_val = 3.0;
    Kokkos::Array<double, 6> coord_bounds = {-bound_val, -bound_val, -bound_val, bound_val, bound_val, bound_val};
    // If not balanced, fill domain unevenly
    if (!balanced)
    {
        coord_bounds = {-2.8, 0.3, -0.2, -0.5, 3.0, 1.3};
    }

    fillRandomCoordinates(cart_coords, coord_bounds, 123);
    fillRandomScalar(q, charge_bounds, 321);

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
        // printf("R%d: initial particle: p(%0.3lf, %0.3lf, %0.3lf), q(%0.3lf)\n", rank,
        //     pos_slice_host(i, 0), pos_slice_host(i, 1), pos_slice_host(i, 2), scalar_slice_host(i));
    }

    // // Copy to device
    auto particle_aosoa =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // Fill the tree
    bool run_load_balance = !balanced;
    tree->create_multipoles(particle_aosoa, run_load_balance);

    // Consider all cells when creating locals because we are only testing one layer
    layer->multipole_to_local(cells_per_leaf_dimension);

    // Create target points at which to calculate potential directly, omitting
    // nearest and second-nearest neighbor cells.
    int num_target_points = 120;
    Kokkos::View<double*[3], TEST_MEMSPACE> target_points( "target_points",
                                                          num_target_points );
    fillRandomCoordinates(target_points, coord_bounds, 456);
    
    auto target_points_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), target_points);

    // Iterate over target points and calculate potential
    Kokkos::View<double*, Kokkos::HostSpace> direct_potentials( "direct_potentials",
                                                          num_target_points );
    Kokkos::deep_copy(direct_potentials, 0.0);
    for (int tpi = 0; tpi < num_target_points; ++tpi)
    {
        // Get the cell this point falls into
        Kokkos::Array<std::size_t, 3> target_cell_ijk;
        for (int dim = 0; dim < 3; ++dim)
        {
            target_cell_ijk[dim] = static_cast<std::size_t>(
                Kokkos::floor((target_points_host(tpi, dim) - global_low_corner[dim]) / cell_size[dim]) );
        }
        // printf("target_cell_ijk(%d): (%d, %d, %d)\n", tpi, target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2]);

        // Set inner bound - where cells are too close for the local
        // approximation to be accurate. Inclusive on lower end,
        // exclusive on upper end
        Kokkos::Array<int, 3> inner_lower_bound;
        Kokkos::Array<int, 3> inner_upper_bound;
        for (int dim = 0; dim < 3; ++dim)
        {
            inner_upper_bound[dim] = Kokkos::min(static_cast<int>(target_cell_ijk[dim]) + 3, cells_per_leaf_dimension);
            inner_lower_bound[dim] = Kokkos::max(static_cast<int>(target_cell_ijk[dim]) - 2, 0);
        }

        // Iterate over all particles inserted into the mesh. If it falls into a cell
        // within 2 cells of the target point's cell, skip it. If not, add its contribution
        // to the potential at the target point.
        for (int op = 0; op < num_points; ++op)
        {
            Kokkos::Array<int, 3> cell_ijk;
            for (int dim = 0; dim < 3; ++dim)
            {
                cell_ijk[dim] = static_cast<int>(
                    Kokkos::floor((cart_coords_h(op, dim) - global_low_corner[dim]) / cell_size[dim]) );
            }

            // Only consider cells between our outer lower and inner lower
            // or inner upper and outer upper bounds. If inside these bounds,
            // skip.
            if ((cell_ijk[0] >= inner_lower_bound[0] && cell_ijk[0] < inner_upper_bound[0]) &&
            (cell_ijk[1] >= inner_lower_bound[1] && cell_ijk[1] < inner_upper_bound[1]) &&
            (cell_ijk[2] >= inner_lower_bound[2] && cell_ijk[2] < inner_upper_bound[2]))
            {
                continue;
            }

            // printf("p%d: Far cell (%d, %d, %d) p(%.2lf, %.2lf, %.2lf)\n", op,
            //     cell_ijk[0], cell_ijk[1], cell_ijk[2], cart_coords_h( op, 0 ), cart_coords_h( op, 1 ), cart_coords_h( op, 2 ));

            // Add this particle's contribution to the potential
            double dx = target_points_host(tpi, 0) - cart_coords_h( op, 0 );
            double dy = target_points_host(tpi, 1) - cart_coords_h( op, 1 );
            double dz = target_points_host(tpi, 2) - cart_coords_h( op, 2 );
            double dist = Kokkos::sqrt( dx * dx + dy * dy + dz * dz );
            direct_potentials(tpi) += q_h( op ) / dist;
        
        }
        // printf("t%d, cell(%lu, %lu, %lu), coord(%.2lf, %.2lf, %.2lf), dp: %.3lf\n", tpi,
        //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
        //     target_points_host(tpi, 0), target_points_host(tpi, 1), target_points_host(tpi, 2),
        //     direct_potentials(tpi));
    }
    
    // Get locals
    auto locals = layer->locals();
    auto ijk2l = layer->cellijk2l();

    // Device-friendly version of global low corner
    Kokkos::Array<double, 3> global_low_corner_k;
    for (int i = 0; i < 3; i++)
        global_low_corner_k[i] = global_low_corner[i];

    // Iterate over target points and use locals to calculate potential
    Kokkos::View<cdouble*, TEST_MEMSPACE> local_potential("local_potential", num_target_points);
    Kokkos::deep_copy(local_potential, cdouble(0.0, 0.0));
    Kokkos::parallel_for(
        "populate_local_potential",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, num_target_points ),
        KOKKOS_LAMBDA( const int tpi ) {

            // Get the cell this point falls into
            Kokkos::Array<std::size_t, 3> target_cell_ijk;
            for (int dim = 0; dim < 3; ++dim)
            {
                target_cell_ijk[dim] = static_cast<std::size_t>(
                    Kokkos::floor((target_points(tpi, dim) - global_low_corner_k[dim]) / cell_size[dim]) );
            }

            // Only continue if this cell exists in the mesh
            auto cell_exists = ijk2l.exists(target_cell_ijk);
            if (!cell_exists)
                return;

             // Center of local expansion is the cell center
            Kokkos::Array<double, 3> l_center;
            for (int i = 0; i < 3; i++)
                l_center[i] = global_low_corner_k[i] + (static_cast<double>(target_cell_ijk[i]) + 0.5) * cell_size[i];

            // Convert target point to spherical coordinates relative to local center
            double r, theta, phi;
            Canopy::Kernel::cart2sph( target_points(tpi, 0) - l_center[0],
                                      target_points(tpi, 1) - l_center[1],
                                      target_points(tpi, 2) - l_center[2],
                                      r, theta, phi );

            auto ijk2l_index = ijk2l.find(target_cell_ijk);
            // printf("tpi: %d, target_cell_ijk: (%lu, %lu, %lu) ijk2l index: %u\n", tpi,
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2], ijk2l_index);
            auto local_index = ijk2l.value_at(ijk2l_index);
            // printf("tpi: %d, local index: %lu\n", tpi, local_index);
            // printf("tcell(%d, %d, %d): l(%d): (%.2lf, %.2lf, %.2lf, %.2lf)\n",
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //         local_index, locals(local_index, 0).real(), locals(local_index, 1).real(),
            //         locals(local_index, 2).real(), locals(local_index, 3).real());
            // Calculate potential using locals
            for ( int j = 0; j <= p_val; ++j )
            {
                for ( int k = -j; k <= j; ++k )
                {
                    int idx = Canopy::Kernel::Scalar::index( j, k );

                    /* Target point 1 calculations */
                    // Greengard eq. 3.59
                    local_potential(tpi) +=
                        locals( local_index, idx ) * Kokkos::pow( r, j ) *
                        Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
                }
            }
            // printf("t%d, cell(%d, %d, %d), center(%.2lf, %.2lf, %.2lf), lp: %.3lf\n", tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2], local_potential(tpi).real());
            printf("t%d, cell(%d, %d, %d), dp: %.3lf, lp: %.3lf\n", tpi,
                target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
                l_center[0], l_center[1], l_center[2], direct_potentials(tpi), local_potential(tpi).real());
        } );
    Kokkos::fence();

    // // Broadcast direct potential to all other ranks.
    // MPI_Bcast(&potential_direct, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // // Copy to device
    // auto particle_aosoa =
    //     Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // // Fill the tree
    // bool run_load_balance = !balanced;
    // tree->create_multipoles(particle_aosoa, run_load_balance);

    // domains_host = tree->layer(0)->domains();
    // // for (std::size_t i = 0; i < domains_host.size(); ++i)
    // // {
    // //     if (rank == 0) printf("After: L0: R%d: [%0.3lf, %0.3lf, %0.3lf] to [%0.3lf, %0.3lf, %0.3lf]\n",
    // //         i, domains_host[i][0], domains_host[i][1], domains_host[i][2], domains_host[i][3],
    // //         domains_host[i][4], domains_host[i][5]);
    // // }

    // /***********************************************
    //  * Check the data in the root layer
    //  **********************************************/
    // auto m_root_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), tree->M_root() );
    // // if (rank == 0)
    // //     for (std::size_t i = 0; i < m_root_h.size(); ++i)
    // //     {
    // //         printf("m_root(%d): (%.4lf, %.4lf)\n", i, m_root_h(i).real(), m_root_h(i).imag());
    // //     }

    // // Compute potential at P using M
    // Kokkos::complex<double> potential_M = 0.0;
    // for ( int j = 0; j <= p; ++j )
    // {
    //     for ( int k = -j; k <= j; ++k )
    //     {
    //         int idx = Canopy::Kernel::Scalar::index( j, k );
    //         potential_M +=
    //             m_root_h( idx ) / Kokkos::pow( r, j + 1 ) *
    //             Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
    //     }
    // }

    // // Check error between translated multipole and direct potentials
    // // The error is already mathematically checked in testM2MKernel0,
    // // so here we just make sure they are close to each other.
    // int p_int = p;
    // double error = Kokkos::pow(10, -p_int+1);
    // EXPECT_NEAR(potential_direct, potential_M.real(), error) << "p="
    //     << p << ": Potentials do not match. Tree depth " << tree->numLayers();
    // printf("R%d: potential: %0.8lf, M: %0.8lf\n", rank, potential_direct, potential_M.real());
    

    // Each rank should own two particles
    // EXPECT_EQ(2, data_host.size());

    // Check that the correct rank owns the particle
    // rank_slice_host = Cabana::slice<2>(data_host);
    // for (std::size_t i = 0; i < data_host.size(); i++)
    // {
    //     EXPECT_EQ(rank_slice_host(i), rank) << "Rank " << rank << std::endl;
    // }

    // XXX - At some point separate this out into a new test?
    // tree->multipole_to_local();

}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

// Test accuracy with increasing truncation cutoffs of multipole coefficients.
// Test with a balanced particle distribution.
TEST( Tree, testMultipole2Local_balanced ) { testMultipole2Local<1>(true); }

//---------------------------------------------------------------------------//

} // end namespace Test