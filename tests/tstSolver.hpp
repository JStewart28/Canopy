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
 * Tests that particle-to-particle potentials are calculated correctly at the leaf layer.
 */
template <std::size_t p>
void testSolver(int points_per_proc_in, bool balanced)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create a tree of depth 3. pos/charge/potential/global particle id
    using particle_tuple_type = Cabana::MemberTypes<double[3], double, double, int>;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
    using particle_aosoa_type_h = Cabana::AoSoA<particle_tuple_type, Kokkos::HostSpace, 4>;
    std::array<double, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<double, 3> global_high_corner = { 3.0, 3.0, 3.0 };

    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 2;
    std::size_t leaf_tiles, red_factor;
    red_factor = 2, leaf_tiles = 64;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, particle_aosoa_type, 0, 1, 2,
        num_dim, cells_per_tile, p>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, MPI_COMM_WORLD);
    
    // The tree depth should always be at least three, but this check is here just in case.
    // If the depth is less than 3, this test may not work correctly.
    // if (rank == 0) printf("R%d: num tree layers: %d\n", rank, tree->numLayers());
    // ASSERT_EQ(tree->numLayers(), 3) << "testMultipole2Local: Error: Tree depth must be depth 3.";

    // Check mesh information for leaf layer
    int cells_per_dimension_leaf = cells_per_tile * leaf_tiles;
    Kokkos::Array<double, 3> cell_size;
    for (int i = 0; i < 3; ++i)
    {
        cell_size[i] = (global_high_corner[i] - global_low_corner[i]) / cells_per_dimension_leaf;
    }
    ASSERT_EQ(tree->layer(0)->cellsPerDim(), cells_per_dimension_leaf) << "testMultipole2Local: Error: Unexpected cells_per_leaf_dimension";
    ASSERT_EQ(tree->layer(0)->tilesPerDim(), leaf_tiles) << "testMultipole2Local: Error: Unexpected leaf_tiles";
    ASSERT_EQ(tree->layer(0)->cellSize(), cell_size) << "testMultipole2Local: Error: Unexpected cell_size";

    // Create the data on rank 0. It will automatically be distributed correctly when
    // filled into the tree. There must be enough particles so that the target point resides
    // in a cell that has been activated in the mesh. This won't be a problem in the
    // "real" code because we only evaluate locals where cells are activated.
    int total_points = points_per_proc_in;
    int owned_points = (rank == 0) ? (total_points) : 0;
    Kokkos::View<double* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          owned_points );
    Kokkos::View<double*, TEST_MEMSPACE> q( "q", owned_points );
    
    Kokkos::Array<double, 2> charge_bounds = {-10.0, 10.0};
    double bound_val = 3.0;
    Kokkos::Array<double, 6> coord_bounds = {-bound_val, -bound_val, -bound_val, bound_val, bound_val, bound_val};
    // If not balanced, fill domain unevenly
    if (!balanced)
    {
        coord_bounds = {-2.8, 0.3, -0.2, -0.5, 3.0, 1.3};
    }
    
    // Kokkos::Array<double, 6> coord_bounds1 = {2.3, 2.3, 2.3, bound_val, bound_val, bound_val};
    fillRandomCoordinates(cart_coords, coord_bounds, 123);
    fillRandomScalar(q, charge_bounds, 321);

    Cabana::AoSoA<particle_tuple_type, Kokkos::HostSpace, 4> particle_aosoa_host("particle_aosoa", owned_points);
    auto pos_slice_host = Cabana::slice<0>(particle_aosoa_host);
    auto scalar_slice_host = Cabana::slice<1>(particle_aosoa_host);
    auto potential_slice_host = Cabana::slice<2>(particle_aosoa_host);
    auto id_slice_host = Cabana::slice<3>(particle_aosoa_host);
    Cabana::deep_copy(potential_slice_host, 0.0);

    // Fill the particles into the AoSoA
    auto cart_coords_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cart_coords);
    auto q_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), q);

    for (int i = 0; i < owned_points; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            pos_slice_host(i, j) = cart_coords_h(i, j);
        }
        scalar_slice_host(i) = q_h(i);
        id_slice_host(i) = i;
        // printf("R%d: initial particle: p(%0.3lf, %0.3lf, %0.3lf), q(%0.3lf)\n", rank,
        //     pos_slice_host(i, 0), pos_slice_host(i, 1), pos_slice_host(i, 2), scalar_slice_host(i));
    }

    // Iterate over particles and calculate potential
    Kokkos::View<double*, Kokkos::HostSpace> direct_potentials( "direct_potentials",
                                                          owned_points );
    Kokkos::deep_copy(direct_potentials, 0.0);
    for (int this_pid = 0; this_pid < owned_points; this_pid++)
    {
        // Get the cell this point falls into
        // Kokkos::Array<std::size_t, 3> this_cell_ijk;
        // for (int dim = 0; dim < 3; ++dim)
        // {
        //     this_cell_ijk[dim] = static_cast<std::size_t>(
        //         Kokkos::floor((pos_slice_host(this_pid, dim) - global_low_corner[dim]) / cell_size[dim]) );
        // }
        // printf("this_pid(%d): (%d, %d, %d)\n", this_pid, this_cell_ijk[0], this_cell_ijk[1], this_cell_ijk[2]);

        // Iterate over all particles inserted into the mesh. If it falls into a cell
        // within 2 cells of the target point's cell, skip it. If not, add its contribution
        // to the potential at the target point.
        for (int other_pid = 0; other_pid < owned_points; other_pid++)
        {
            if (this_pid == other_pid)
                continue;

            // Kokkos::Array<int, 3> cell_ijk;
            // for (int dim = 0; dim < 3; ++dim)
            // {
            //     cell_ijk[dim] = static_cast<int>(
            //         Kokkos::floor((pos_slice_host(other_pid, dim) - global_low_corner[dim]) / cell_size[dim]) );
            // }
            
            // printf("this_pid(%d): other_pid(%d): (%d, %d, %d)\n", this_pid, other_pid, cell_ijk[0], cell_ijk[1], cell_ijk[2]);
            double dx = pos_slice_host(other_pid, 0) - pos_slice_host( this_pid, 0 );
            double dy = pos_slice_host(other_pid, 1) - pos_slice_host( this_pid, 1 );
            double dz = pos_slice_host(other_pid, 2) - pos_slice_host( this_pid, 2 );
            double dist = Kokkos::sqrt( dx * dx + dy * dy + dz * dz );
            // printf("dp(%d) += other(%d): dx/y/z: %.2lf, %.2lf, %.2lf\n", this_pid, other_pid, dx, dy, dz);
            direct_potentials(this_pid) += q_h( other_pid ) / dist;        
        }
    }

    // Copy to device
    auto particle_aosoa =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // Fill the tree. This migrates particles to their correct rank.
    bool run_load_balance = !balanced;
    tree->create_multipoles(particle_aosoa, run_load_balance);

    tree->multipole_to_local();
    
    tree->computeL2P();

    tree->computeP2P();

    // Gather all particles from the tree back to rank 0 for testing
    auto tmp = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), tree->particles());
    particle_aosoa_type_h tree_particles("tree_particles", tmp.size());
    Cabana::deep_copy(tree_particles, tmp);

    // Remove ghost particles
    tree_particles.resize(tree->numOwnedParticles());
    
    // Send particles back to rank 0 for testing
    Kokkos::View<int*, Kokkos::HostSpace> send_to("send_to", tree->numOwnedParticles());
    Kokkos::deep_copy(send_to, 0);
    Cabana::Distributor<Kokkos::HostSpace> distributor(MPI_COMM_WORLD, send_to);
    Cabana::migrate( distributor, tree_particles );

    // Sort the particles by increasing cell_id
    auto tree_id_slice = Cabana::slice<3>(tree_particles);
    auto sort_data = Cabana::sortByKey( tree_id_slice );
    Cabana::permute( sort_data, tree_particles );
    tree_id_slice = Cabana::slice<3>(tree_particles);
    auto tree_potentials = Cabana::slice<2>(tree_particles);

    int p_int = static_cast<int>(p);
    for (int i = 0; i < owned_points; i++)
    {
        auto particle_id = tree_id_slice(i);
        auto direct_potential = direct_potentials(particle_id);
        auto mesh_potential = tree_potentials(i);
        double allowed_error = Kokkos::pow(10, -p_int+3);
        EXPECT_NEAR(mesh_potential, direct_potential, allowed_error) << " at particle " << particle_id;
        // printf("i%d, pid %d: direct: %.6lf, mesh: %.6lf\n", i, particle_id, direct_potential, mesh_potential);
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( Tree, testSolver_balanced )
{ 
    testSolver<5>(50, true);
}

//---------------------------------------------------------------------------//

} // end namespace Test