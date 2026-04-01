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

#include <Canopy_Solver.hpp>

#include <test_helpers.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//


/**
 * Tests that particle-to-particle potentials are calculated correctly at the leaf layer.
 */
template <int p>
void testSolver(int points_per_proc_in, bool balanced)
{
    static_assert(p > 0);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create a tree of depth 3. pos/charge/potential/global particle id
    std::array<scalar_type, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<scalar_type, 3> global_high_corner = { 3.0, 3.0, 3.0 };

    static constexpr std::size_t cells_per_tile = 2;
    std::size_t leaf_tiles, red_factor;
    red_factor = 2, leaf_tiles = 16;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createSolver<TEST_MEMSPACE, TEST_EXECSPACE, MD_f, cells_per_tile, p>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, MPI_COMM_WORLD);
    
    // The tree depth should always be at least three, but this check is here just in case.
    // If the depth is less than 3, this test may not work correctly.
    // if (rank == 0) printf("R%d: num tree layers: %d\n", rank, tree->numLayers());
    // ASSERT_EQ(tree->numLayers(), 3) << "testMultipole2Local: Error: Solver depth must be depth 3.";

    // Create the data on rank 0. It will automatically be distributed correctly when
    // filled into the tree. There must be enough particles so that the target point resides
    // in a cell that has been activated in the mesh. This won't be a problem in the
    // "real" code because we only evaluate locals where cells are activated.
    int total_points = points_per_proc_in;
    int owned_points = (rank == 0) ? (total_points) : 0;
    Kokkos::View<scalar_type* [3], TEST_MEMSPACE> cart_coords( "cart_coords",
                                                          owned_points );
    Kokkos::View<scalar_type*, TEST_MEMSPACE> q( "q", owned_points );
    
    Kokkos::Array<scalar_type, 2> charge_bounds = {-10.0, 10.0};
    scalar_type bound_val = 3.0;
    Kokkos::Array<scalar_type, 6> coord_bounds = {-bound_val, -bound_val, -bound_val, bound_val, bound_val, bound_val};
    // If not balanced, fill domain unevenly
    if (!balanced)
    {
        coord_bounds = {-3.0, -3.0, -0.05, 3.0, 3.0, 0.05};
    }
    
    // Kokkos::Array<scalar_type, 6> coord_bounds1 = {2.3, 2.3, 2.3, bound_val, bound_val, bound_val};
    fillRandomCoordinates(cart_coords, coord_bounds, 123);
    fillRandomScalar(q, charge_bounds, 321);

    particle_aosoa_type_f_h particle_aosoa_host("particle_aosoa", owned_points);
    auto pos_slice_host = Cabana::slice<MD_f::pos>(particle_aosoa_host);
    auto scalar_slice_host = Cabana::slice<MD_f::in>(particle_aosoa_host);
    auto potential_slice_host = Cabana::slice<MD_f::out>(particle_aosoa_host);
    auto force_slice_host = Cabana::slice<MD_f::force>(particle_aosoa_host);
    auto id_slice_host = Cabana::slice<4>(particle_aosoa_host);
    Cabana::deep_copy(potential_slice_host, 0.0);
    Cabana::deep_copy(force_slice_host, 0.0);

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
    Kokkos::View<scalar_type*, Kokkos::HostSpace> direct_potentials( "direct_potentials",
                                                          owned_points );
    Kokkos::View<scalar_type*[3], Kokkos::HostSpace> direct_forces( "direct_forces",
                                                          owned_points );
    Kokkos::deep_copy(direct_potentials, 0.0);
    Kokkos::deep_copy(direct_forces, 0.0);
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
            scalar_type dx = pos_slice_host(other_pid, 0) - pos_slice_host( this_pid, 0 );
            scalar_type dy = pos_slice_host(other_pid, 1) - pos_slice_host( this_pid, 1 );
            scalar_type dz = pos_slice_host(other_pid, 2) - pos_slice_host( this_pid, 2 );
            scalar_type dist = Kokkos::sqrt( dx * dx + dy * dy + dz * dz );
            // printf("dp(%d) += other(%d): dx/y/z: %.2lf, %.2lf, %.2lf\n", this_pid, other_pid, dx, dy, dz);
            direct_potentials(this_pid) += q_h( other_pid ) / dist;

            // Force calculation
            scalar_type dist_inv  = 1.0 / dist;
            scalar_type dist_inv3 = dist_inv * dist_inv * dist_inv;
            scalar_type fp = -1 * q_h(this_pid) * q_h(other_pid) * dist_inv3;
            direct_forces(this_pid, 0) += fp * dx;
            direct_forces(this_pid, 1) += fp * dy;
            direct_forces(this_pid, 2) += fp * dz;     
        }
    }

    // Copy particles to device
    auto aosoa_device = std::make_shared<particle_aosoa_type_f>("aosoa_device", particle_aosoa_host.size());
    Cabana::deep_copy(*aosoa_device, particle_aosoa_host);
        
    // Fill the tree. This migrates particles to their correct rank.
    bool run_load_balance = !balanced;
    tree->solve(aosoa_device, run_load_balance);

    // Check mesh information for leaf layer. Must be done after solve or else the tree has
    // not yet been constructed.
    int cells_per_dimension_leaf = cells_per_tile * leaf_tiles;
    Kokkos::Array<scalar_type, 3> cell_size;
    for (int i = 0; i < 3; ++i)
    {
        cell_size[i] = (global_high_corner[i] - global_low_corner[i]) / cells_per_dimension_leaf;
    }
    ASSERT_EQ(tree->layer(0)->cellsPerDim(), cells_per_dimension_leaf) << "testMultipole2Local: Error: Unexpected cells_per_leaf_dimension";
    ASSERT_EQ(tree->layer(0)->tilesPerDim(), leaf_tiles) << "testMultipole2Local: Error: Unexpected leaf_tiles";
    ASSERT_EQ(tree->layer(0)->cellSize(), cell_size) << "testMultipole2Local: Error: Unexpected cell_size";


    // Gather all particles from the tree back to rank 0 for testing
    auto tmp = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), *(tree->data()));
    particle_aosoa_type_f_h tree_particles("tree_particles", tmp.size());
    Cabana::deep_copy(tree_particles, tmp);

    // Remove ghost particles
    tree_particles.resize(tree->numOwnedParticles());
    
    // Send particles back to rank 0 for testing
    Kokkos::View<int*, Kokkos::HostSpace> send_to("send_to", tree->numOwnedParticles());
    Kokkos::deep_copy(send_to, 0);
    Cabana::Distributor<Kokkos::HostSpace> distributor(MPI_COMM_WORLD, send_to);
    Cabana::migrate( distributor, tree_particles );

    // Sort the particles by increasing cell_id
    auto tree_id_slice = Cabana::slice<4>(tree_particles);
    auto sort_data = Cabana::sortByKey( tree_id_slice );
    Cabana::permute( sort_data, tree_particles );
    tree_id_slice = Cabana::slice<4>(tree_particles);
    auto tree_potentials = Cabana::slice<MD_f::out>(tree_particles);
    auto tree_forces = Cabana::slice<MD_f::force>(tree_particles);

    double max_error_potential = 0.0;
    double max_error_force = 0.0;
    for (int i = 0; i < owned_points; i++)
    {
        auto direct_potential = direct_potentials(i);
        auto particle_id = tree_id_slice(i);
        auto solver_potential = tree_potentials(i);
        EXPECT_NEAR(solver_potential, direct_potential, 0.003) << " at particle " << i;
        const auto error_p = Kokkos::abs(direct_potential - solver_potential);
        if (error_p > max_error_potential)
            max_error_potential = error_p;
        for (int d = 0; d < 3; d++)
        {
            auto direct_force = direct_forces(i, d);
            auto solver_force = tree_forces(i, d);
            EXPECT_NEAR(solver_force, direct_force, 0.6) << " at particle " << i;
            const auto error_f = Kokkos::abs(direct_force - solver_force);
            if (error_f > max_error_force)
                max_error_force = error_f;
        }
        // printf("i%d, pid %d: direct: %.6lf, mesh: %.6lf\n", i, particle_id, direct_potential, mesh_potential);
    }
    if (owned_points > 0)
        printf("Max errors: potential: %.5lf, force: %.5lf\n", max_error_potential, max_error_force);
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( SingleSolve, testSingleSolve_balanced )
{ 
    testSolver<6>(500, true);
}
TEST( SingleSolve, testSingleSolve_unbalanced )
{ 
    testSolver<6>(500, false);
}

//---------------------------------------------------------------------------//

} // end namespace Test