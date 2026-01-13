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

// pos/charge/potential/global particle id
using particle_tuple_type = Cabana::MemberTypes<double[3], double, double, int>;
using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;

void testCell2Bound()
{
    // Define arrays
    std::array<int, 6> include;
    std::array<int, 6> exclude;
    std::array<int, 6> correct_include;
    std::array<int, 6> correct_exclude;

    // Start layer = layer
    Kokkos::Array<int, 3> cell_ijk = {0, 0, 0};
    int layer = 0;
    int start_layer = 2;
    int start_cpd = 4;
    int cell_incr_factor = 2;

    auto bounds = Canopy::cell2Bound(cell_ijk, layer, start_layer,
        start_cpd, cell_incr_factor);

    for (int i = 0; i < 6; i++)
    {
        include[i] = bounds.first[i];
        exclude[i] = bounds.second[i];
    }

    correct_include = {0, 0, 0, 6, 6, 6};
    correct_exclude = {0, 0, 0, 3, 3, 3};

    EXPECT_EQ(include, correct_include);
    EXPECT_EQ(exclude, correct_exclude);

    // Next test
    cell_ijk = {8, 12, 12};

    bounds = Canopy::cell2Bound(cell_ijk, layer, start_layer,
        start_cpd, cell_incr_factor);

    for (int i = 0; i < 6; i++)
    {
        include[i] = bounds.first[i];
        exclude[i] = bounds.second[i];
    }

    correct_include = {4, 8, 8, 14, 16, 16};
    correct_exclude = {6, 10, 10, 11, 15, 15};

    EXPECT_EQ(include, correct_include);
    EXPECT_EQ(exclude, correct_exclude);

    // Next test
    cell_ijk = {10, 7, 3};

    bounds = Canopy::cell2Bound(cell_ijk, layer, start_layer,
        start_cpd, cell_incr_factor);

    for (int i = 0; i < 6; i++)
    {
        include[i] = bounds.first[i];
        exclude[i] = bounds.second[i];
    }

    correct_include = {6, 2, 0, 16, 12, 8};
    correct_exclude = {8, 5, 1, 13, 10, 6};
    // 6	2	0	16	12	8	8	5	1	13	10	6


    EXPECT_EQ(include, correct_include);
    EXPECT_EQ(exclude, correct_exclude);

    // Next test
    cell_ijk = {0, 0, 0};
    layer = 1;
    start_layer = 1;
    start_cpd = 16;
    cell_incr_factor = 2;

    bounds = Canopy::cell2Bound(cell_ijk, layer, start_layer,
        start_cpd, cell_incr_factor);

    for (int i = 0; i < 6; i++)
    {
        include[i] = bounds.first[i];
        exclude[i] = bounds.second[i];
    }

    correct_include = {0, 0, 0, 16, 16, 16};
    correct_exclude = {0, 0, 0, 3, 3, 3};

    EXPECT_EQ(include, correct_include);
    EXPECT_EQ(exclude, correct_exclude);
}

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
void testMultipole2Local0(int points_per_proc_in, bool balanced)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create a tree of depth 3.
    std::array<double, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<double, 3> global_high_corner = { 3.0, 3.0, 3.0 };

    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 2;
    static constexpr std::size_t p = p_val;
    std::size_t leaf_tiles, red_factor;
    red_factor = comm_size * 8, leaf_tiles = comm_size * 8;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, particle_aosoa_type, 0, 1, 2,
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

    fillRandomCoordinates(cart_coords, coord_bounds, 123);
    fillRandomScalar(q, charge_bounds, 321);

    // Activate cell with the target point
    // cart_coords(0, 0) = -2.9;
    // cart_coords(0, 1) = -2.8;
    // cart_coords(0, 2) = -2.85;
    // q(0) = 0.0;

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

    // Calculate potentials directly, considering all particles in cells more than 2 cells away
    // Iterate over particles and calculate potential
    Kokkos::View<double*, Kokkos::HostSpace> direct_potentials( "direct_potentials",
                                                          total_points );
    Kokkos::deep_copy(direct_potentials, 0.0);

    // Only rank 0 executes this loop because it owns all points
    for (int this_pid = 0; this_pid < owned_points; this_pid++)
    {
        // Get the cell this point falls into
        Kokkos::Array<std::size_t, 3> this_cell_ijk;
        for (int dim = 0; dim < 3; ++dim)
        {
            this_cell_ijk[dim] = static_cast<std::size_t>(
                Kokkos::floor((pos_slice_host(this_pid, dim) - global_low_corner[dim]) / cell_size[dim]) );
        }
        // printf("this_pid(%d): (%d, %d, %d)\n", this_pid, this_cell_ijk[0], this_cell_ijk[1], this_cell_ijk[2]);

        // Set inner bound - where cells are too close for the local
        // approximation to be accurate. Inclusive on lower end,
        // exclusive on upper end
        Kokkos::Array<int, 3> inner_lower_bound;
        Kokkos::Array<int, 3> inner_upper_bound;
        for (int dim = 0; dim < 3; ++dim)
        {
            inner_upper_bound[dim] = Kokkos::min(static_cast<int>(this_cell_ijk[dim]) + 3, cells_per_leaf_dimension);
            inner_lower_bound[dim] = Kokkos::max(static_cast<int>(this_cell_ijk[dim]) - 2, 0);
        }

        // Iterate over all particles inserted into the mesh. If it falls into a cell
        // within 2 cells of the target point's cell, skip it. If not, add its contribution
        // to the potential at the target point.
        for (int other_pid = 0; other_pid < owned_points; other_pid++)
        {
            if (this_pid == other_pid)
                continue;

            Kokkos::Array<int, 3> cell_ijk;
            for (int dim = 0; dim < 3; ++dim)
            {
                cell_ijk[dim] = static_cast<int>(
                    Kokkos::floor((pos_slice_host(other_pid, dim) - global_low_corner[dim]) / cell_size[dim]) );
            }

            // Only consider cells outside the inner local bound
            if ((cell_ijk[0] >= inner_lower_bound[0] && cell_ijk[0] < inner_upper_bound[0]) &&
            (cell_ijk[1] >= inner_lower_bound[1] && cell_ijk[1] < inner_upper_bound[1]) &&
            (cell_ijk[2] >= inner_lower_bound[2] && cell_ijk[2] < inner_upper_bound[2]))
            {
                continue;
            }
            // printf("this_pid(%d): other_pid(%d): (%d, %d, %d)\n", this_pid, other_pid, cell_ijk[0], cell_ijk[1], cell_ijk[2]);
            double dx = pos_slice_host(other_pid, 0) - pos_slice_host( this_pid, 0 );
            double dy = pos_slice_host(other_pid, 1) - pos_slice_host( this_pid, 1 );
            double dz = pos_slice_host(other_pid, 2) - pos_slice_host( this_pid, 2 );
            double dist = Kokkos::sqrt( dx * dx + dy * dy + dz * dz );
            // printf("dp(%d) += other(%d): dx/y/z: %.2lf, %.2lf, %.2lf\n", this_pid, other_pid, dx, dy, dz);
            direct_potentials(this_pid) += q_h( other_pid ) / dist;        
        }
    }

    // Send direct_potentials view to all ranks so they can check it against their points    
    MPI_Bcast(direct_potentials.data(), total_points, MPI_DOUBLE, 0, MPI_COMM_WORLD );

    // Copy to device
    auto particle_aosoa =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // Fill the tree
    bool run_load_balance = !balanced;
    tree->create_multipoles(particle_aosoa, run_load_balance);

    // Just test single-layer multipole to local conversion.
    layer->multipole_to_local(16, tree->numLayers() - 2);

    // Get locals
    auto locals = layer->locals();
    auto ijk2index = layer->cellijk2l();
    auto locals_slice = Cabana::slice<0>(locals);

    // Get particles
    auto tree_particles = tree->particles();
    auto tree_particle_positions = Cabana::slice<0>(tree_particles);

    // Sort the particles by increasing cell_id
    auto tree_id_slice = Cabana::slice<3>(tree_particles);
    auto sort_data = Cabana::sortByKey( tree_id_slice );
    Cabana::permute( sort_data, tree_particles );
    tree_id_slice = Cabana::slice<3>(tree_particles);
    auto tree_potentials = Cabana::slice<2>(tree_particles);

    // Reset tree_potentials
    Cabana::deep_copy(tree_potentials, 0.0);

    // Device-friendly version of global low corner
    Kokkos::Array<double, 3> global_low_corner_k;
    for (int i = 0; i < 3; i++)
        global_low_corner_k[i] = global_low_corner[i];

    // Iterate over points and use locals to calculate potential
    Kokkos::parallel_for(
        "populate_local_potential",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, tree_particles.size() ),
        KOKKOS_LAMBDA( const int tpi ) {

            // Get the cell this point falls into
            Kokkos::Array<std::size_t, 3> target_cell_ijk;
            for (int dim = 0; dim < 3; ++dim)
            {
                target_cell_ijk[dim] = static_cast<std::size_t>(
                    Kokkos::floor((tree_particle_positions(tpi, dim) - global_low_corner_k[dim]) / cell_size[dim]) );
            }

            // Only continue if this cell exists in the mesh.
            // It always should.
            auto cell_exists = ijk2index.exists(target_cell_ijk);
            if (!cell_exists)
                return;

             // Center of local expansion is the cell center
            Kokkos::Array<double, 3> l_center;
            for (int i = 0; i < 3; i++)
                l_center[i] = global_low_corner_k[i] + (static_cast<double>(target_cell_ijk[i]) + 0.5) * cell_size[i];

            // printf("t%d, ijk(%d, %d, %d), c(%.2lf, %.2lf, %.2lf)\n", tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2]);


            // Convert target point to spherical coordinates relative to local center
            double r, theta, phi;
            Canopy::Kernel::cart2sph( tree_particle_positions(tpi, 0) - l_center[0],
                                      tree_particle_positions(tpi, 1) - l_center[1],
                                      tree_particle_positions(tpi, 2) - l_center[2],
                                      r, theta, phi );

            auto ijk2l_index = ijk2index.find(target_cell_ijk);
            // printf("tpi: %d, target_cell_ijk: (%lu, %lu, %lu) ijk2index index: %u\n", tpi,
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2], ijk2l_index);
            auto local_index = ijk2index.value_at(ijk2l_index);
            // printf("tpi: %d, local index: %lu\n", tpi, local_index);
            // printf("tcell(%d, %d, %d): l(%d): (%.2lf, %.2lf, %.2lf, %.2lf)\n",
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //         local_index, locals(local_index, 0).real(), locals(local_index, 1).real(),
            //         locals(local_index, 2).real(), locals(local_index, 3).real());
            // Calculate potential using locals
            cdouble accumulator(0.0, 0.0);
            for ( int j = 0; j <= p_val; ++j )
            {
                for ( int k = -j; k <= j; ++k )
                {
                    int idx = Canopy::Kernel::Scalar::index( j, k );

                    /* Target point 1 calculations */
                    // Greengard eq. 3.59
                    cdouble val = cdouble(locals_slice(local_index, idx, 0), locals_slice(local_index, idx, 1));
                    accumulator +=
                        val * Kokkos::pow( r, j ) *
                        Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
                }
            }
            tree_potentials(tpi) = accumulator.real();
        } );
    Kokkos::fence();
    
    // p_int for error checking.
    int p_int = static_cast<int>(p_val);

    auto tree_particles_h = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), tree_particles);
    auto pid_h = Cabana::slice<3>(tree_particles);
    auto p_pot = Cabana::slice<2>(tree_particles);
    
    for (int i = 0; i < owned_points; i++)
    {
        auto particle_id = pid_h(i);
        auto particle_potential = p_pot(i);
        auto direct_potential = direct_potentials(particle_id);
        ASSERT_NEAR(particle_potential, direct_potential, Kokkos::pow(10, -p_int+2));
        // printf("R%d: particle %d: direct: %.5lf, tree: %.5lf\n", rank, particle_id, particle_potential, direct_potential);
    }
}

template <std::size_t p_val>
void testMultipole2Local1(int points_per_proc_in, bool balanced)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Create a tree of depth 3.
    std::array<double, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<double, 3> global_high_corner = { 3.0, 3.0, 3.0 };

    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 2;
    static constexpr std::size_t p = p_val;
    std::size_t leaf_tiles, red_factor;
    red_factor = comm_size, leaf_tiles = comm_size * 32;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, particle_aosoa_type, 0, 1, 2,
        num_dim, cells_per_tile, p>(
            global_low_corner, global_high_corner, leaf_tiles, red_factor, MPI_COMM_WORLD);
    
    // The tree depth should always be at least three, but this check is here just in case.
    // If the depth is less than 3, this test may not work correctly.
    // if (rank == 0) printf("R%d: num tree layers: %d\n", rank, tree->numLayers());
    // ASSERT_EQ(tree->numLayers(), 3) << "testMultipole2Local: Error: Tree depth must be depth 3.";

    // Check mesh information for leaf layer
    int cells_per_leaf_dimension = cells_per_tile * leaf_tiles;
    Kokkos::Array<double, 3> cell_size;
    for (int i = 0; i < 3; ++i)
    {
        cell_size[i] = (global_high_corner[i] - global_low_corner[i]) / cells_per_leaf_dimension;
    }
    ASSERT_EQ(tree->layer(0)->cellsPerDim(), cells_per_leaf_dimension) << "testMultipole2Local: Error: Unexpected cells_per_leaf_dimension";
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

    fillRandomCoordinates(cart_coords, coord_bounds, 123);
    fillRandomScalar(q, charge_bounds, 321);

    // Activate cell with the target point
    // cart_coords(0, 0) = -2.9;
    // cart_coords(0, 1) = -2.8;
    // cart_coords(0, 2) = -2.85;
    // q(0) = 0.0;

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

    // Calculate potentials directly, considering all particles in cells more than 2 cells away
    // Iterate over particles and calculate potential
    Kokkos::View<double*, Kokkos::HostSpace> direct_potentials( "direct_potentials",
                                                          total_points );
    Kokkos::deep_copy(direct_potentials, 0.0);

    // Only rank 0 executes this loop because it owns all points
    for (int this_pid = 0; this_pid < owned_points; this_pid++)
    {
        // Get the cell this point falls into
        Kokkos::Array<std::size_t, 3> this_cell_ijk;
        for (int dim = 0; dim < 3; ++dim)
        {
            this_cell_ijk[dim] = static_cast<std::size_t>(
                Kokkos::floor((pos_slice_host(this_pid, dim) - global_low_corner[dim]) / cell_size[dim]) );
        }
        // printf("this_pid(%d): (%d, %d, %d)\n", this_pid, this_cell_ijk[0], this_cell_ijk[1], this_cell_ijk[2]);

        // Set inner bound - where cells are too close for the local
        // approximation to be accurate. Inclusive on lower end,
        // exclusive on upper end
        Kokkos::Array<int, 3> inner_lower_bound;
        Kokkos::Array<int, 3> inner_upper_bound;
        for (int dim = 0; dim < 3; ++dim)
        {
            inner_upper_bound[dim] = Kokkos::min(static_cast<int>(this_cell_ijk[dim]) + 3, cells_per_leaf_dimension);
            inner_lower_bound[dim] = Kokkos::max(static_cast<int>(this_cell_ijk[dim]) - 2, 0);
        }

        // Iterate over all particles inserted into the mesh. If it falls into a cell
        // within 2 cells of the target point's cell, skip it. If not, add its contribution
        // to the potential at the target point.
        for (int other_pid = 0; other_pid < owned_points; other_pid++)
        {
            if (this_pid == other_pid)
                continue;

            Kokkos::Array<int, 3> cell_ijk;
            for (int dim = 0; dim < 3; ++dim)
            {
                cell_ijk[dim] = static_cast<int>(
                    Kokkos::floor((pos_slice_host(other_pid, dim) - global_low_corner[dim]) / cell_size[dim]) );
            }

            // Only consider cells outside the inner local bound
            if ((cell_ijk[0] >= inner_lower_bound[0] && cell_ijk[0] < inner_upper_bound[0]) &&
            (cell_ijk[1] >= inner_lower_bound[1] && cell_ijk[1] < inner_upper_bound[1]) &&
            (cell_ijk[2] >= inner_lower_bound[2] && cell_ijk[2] < inner_upper_bound[2]))
            {
                continue;
            }
            // printf("this_pid(%d): other_pid(%d): (%d, %d, %d)\n", this_pid, other_pid, cell_ijk[0], cell_ijk[1], cell_ijk[2]);
            double dx = pos_slice_host(other_pid, 0) - pos_slice_host( this_pid, 0 );
            double dy = pos_slice_host(other_pid, 1) - pos_slice_host( this_pid, 1 );
            double dz = pos_slice_host(other_pid, 2) - pos_slice_host( this_pid, 2 );
            double dist = Kokkos::sqrt( dx * dx + dy * dy + dz * dz );
            // printf("dp(%d) += other(%d): dx/y/z: %.2lf, %.2lf, %.2lf\n", this_pid, other_pid, dx, dy, dz);
            direct_potentials(this_pid) += q_h( other_pid ) / dist;        
        }
    }

    // Send direct_potentials view to all ranks so they can check it against their points    
    MPI_Bcast(direct_potentials.data(), total_points, MPI_DOUBLE, 0, MPI_COMM_WORLD );

    // Copy to device
    auto particle_aosoa =
        Cabana::create_mirror_view_and_copy( TEST_MEMSPACE(), particle_aosoa_host );
        
    // Fill the tree
    bool run_load_balance = !balanced;
    tree->create_multipoles(particle_aosoa, run_load_balance);

    tree->multipole_to_local();

    // Get locals at leaf layer
    auto layer = tree->layer(0);
    auto locals = layer->locals();
    auto ijk2index = layer->cellijk2l();
    auto locals_slice = Cabana::slice<0>(locals);

    // Get particles
    auto tree_particles = tree->particles();
    auto tree_particle_positions = Cabana::slice<0>(tree_particles);

    // Sort the particles by increasing cell_id
    auto tree_id_slice = Cabana::slice<3>(tree_particles);
    auto sort_data = Cabana::sortByKey( tree_id_slice );
    Cabana::permute( sort_data, tree_particles );
    tree_id_slice = Cabana::slice<3>(tree_particles);
    auto tree_potentials = Cabana::slice<2>(tree_particles);

    // Reset tree_potentials
    Cabana::deep_copy(tree_potentials, 0.0);

    // Device-friendly version of global low corner
    Kokkos::Array<double, 3> global_low_corner_k;
    for (int i = 0; i < 3; i++)
        global_low_corner_k[i] = global_low_corner[i];

    // Iterate over points and use locals to calculate potential
    Kokkos::parallel_for(
        "populate_local_potential",
        Kokkos::RangePolicy<TEST_EXECSPACE>( 0, tree_particles.size() ),
        KOKKOS_LAMBDA( const int tpi ) {

            // Get the cell this point falls into
            Kokkos::Array<std::size_t, 3> target_cell_ijk;
            for (int dim = 0; dim < 3; ++dim)
            {
                target_cell_ijk[dim] = static_cast<std::size_t>(
                    Kokkos::floor((tree_particle_positions(tpi, dim) - global_low_corner_k[dim]) / cell_size[dim]) );
            }

            // Only continue if this cell exists in the mesh.
            // It always should.
            auto cell_exists = ijk2index.exists(target_cell_ijk);
            if (!cell_exists)
                return;

             // Center of local expansion is the cell center
            Kokkos::Array<double, 3> l_center;
            for (int i = 0; i < 3; i++)
                l_center[i] = global_low_corner_k[i] + (static_cast<double>(target_cell_ijk[i]) + 0.5) * cell_size[i];

            // printf("t%d, ijk(%d, %d, %d), c(%.2lf, %.2lf, %.2lf)\n", tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2]);


            // Convert target point to spherical coordinates relative to local center
            double r, theta, phi;
            Canopy::Kernel::cart2sph( tree_particle_positions(tpi, 0) - l_center[0],
                                      tree_particle_positions(tpi, 1) - l_center[1],
                                      tree_particle_positions(tpi, 2) - l_center[2],
                                      r, theta, phi );

            auto ijk2l_index = ijk2index.find(target_cell_ijk);
            // printf("tpi: %d, target_cell_ijk: (%lu, %lu, %lu) ijk2index index: %u\n", tpi,
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2], ijk2l_index);
            auto local_index = ijk2index.value_at(ijk2l_index);
            // printf("tpi: %d, local index: %lu\n", tpi, local_index);
            // printf("tcell(%d, %d, %d): l(%d): (%.2lf, %.2lf, %.2lf, %.2lf)\n",
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //         local_index, locals(local_index, 0).real(), locals(local_index, 1).real(),
            //         locals(local_index, 2).real(), locals(local_index, 3).real());
            // Calculate potential using locals
            cdouble accumulator(0.0, 0.0);
            for ( int j = 0; j <= p_val; ++j )
            {
                for ( int k = -j; k <= j; ++k )
                {
                    int idx = Canopy::Kernel::Scalar::index( j, k );

                    /* Target point 1 calculations */
                    // Greengard eq. 3.59
                    cdouble val = cdouble(locals_slice(local_index, idx, 0), locals_slice(local_index, idx, 1));
                    accumulator +=
                        val * Kokkos::pow( r, j ) *
                        Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
                }
            }
            tree_potentials(tpi) = accumulator.real();
        } );
    Kokkos::fence();
    
    // p_int for error checking.
    int p_int = static_cast<int>(p_val);

    auto tree_particles_h = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), tree_particles);
    auto pid_h = Cabana::slice<3>(tree_particles);
    auto p_pot = Cabana::slice<2>(tree_particles);
    
    for (int i = 0; i < owned_points; i++)
    {
        auto particle_id = pid_h(i);
        auto particle_potential = p_pot(i);
        auto direct_potential = direct_potentials(particle_id);
        ASSERT_NEAR(particle_potential, direct_potential, Kokkos::pow(10, -p_int+3));
        // printf("R%d: particle %d: direct: %.5lf, tree: %.5lf\n", rank, particle_id, particle_potential, direct_potential);
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( Helper, testCell2Bound)
{
    testCell2Bound();
}

// Test accuracy with increasing truncation cutoffs of multipole coefficients.
// Test with a balanced particle distribution.
TEST( Tree, testMultipole2Local0_balanced )
{ 
    testMultipole2Local0<3>(200, true);     
}

TEST( Tree, testMultipole2Local1_balanced )
{ 
    testMultipole2Local1<6>(80, true); 
}

//---------------------------------------------------------------------------//

} // end namespace Test