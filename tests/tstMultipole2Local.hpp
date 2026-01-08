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
    int points_per_proc = points_per_proc_in;
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

    // Activate cell with the target point
    cart_coords(0, 0) = -2.9;
    cart_coords(0, 1) = -2.8;
    cart_coords(0, 2) = -2.85;
    q(0) = 0.0;

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

    // Just test single-layer multipole to local conversion.
    layer->multipole_to_local(16, tree->numLayers() - 2);

    // Create target points at which to calculate potential directly, omitting
    // nearest and second-nearest neighbor cells.
    int num_target_points = 100;
    Kokkos::View<double*[3], TEST_MEMSPACE> target_points( "target_points",
                                                          num_target_points );
    fillRandomCoordinates(target_points, coord_bounds, 456);

    // Put target point in cell (14, 12, 5)
    // cell: (14, 12, 5) center: (2.44, 1.69, -0.94)
    // target_points(0, 0) = 2.5;
    // target_points(0, 1) = 1.5;
    // target_points(0, 2) = -0.9;

    // Put target point in upper corner of domain
    target_points(0, 0) = -2.88;
    target_points(0, 1) = -2.92;
    target_points(0, 2) = -2.89;
    
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

            // if (target_cell_ijk[0] == 14 && target_cell_ijk[1] == 12 && target_cell_ijk[2] == 5)
            //     printf("R%d: cell %d, %d, %d, neighbor %d, %d, %d (op %d)\n", rank,
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //         cell_ijk[0], cell_ijk[1], cell_ijk[2], op);
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
    auto ijk2index = layer->cellijk2l();
    auto locals_slice = Cabana::slice<0>(locals);

    // Track which cells are activated in the mesh. If a target point is in a non-
    // activated cell, skip it when checking pootentials
    Kokkos::View<int*, TEST_MEMSPACE> is_activated("is_activated", num_target_points);
    Kokkos::deep_copy(is_activated, 0);

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
            auto cell_exists = ijk2index.exists(target_cell_ijk);
            if (!cell_exists)
                return;

            // Set cell to activated
            is_activated(tpi) = 1;

             // Center of local expansion is the cell center
            Kokkos::Array<double, 3> l_center;
            for (int i = 0; i < 3; i++)
                l_center[i] = global_low_corner_k[i] + (static_cast<double>(target_cell_ijk[i]) + 0.5) * cell_size[i];

            // printf("t%d, ijk(%d, %d, %d), c(%.2lf, %.2lf, %.2lf)\n", tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2]);


            // Convert target point to spherical coordinates relative to local center
            double r, theta, phi;
            Canopy::Kernel::cart2sph( target_points(tpi, 0) - l_center[0],
                                      target_points(tpi, 1) - l_center[1],
                                      target_points(tpi, 2) - l_center[2],
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
            for ( int j = 0; j <= p_val; ++j )
            {
                for ( int k = -j; k <= j; ++k )
                {
                    int idx = Canopy::Kernel::Scalar::index( j, k );

                    /* Target point 1 calculations */
                    // Greengard eq. 3.59
                    cdouble val = cdouble(locals_slice(local_index, idx, 0), locals_slice(local_index, idx, 1));
                    local_potential(tpi) +=
                        val * Kokkos::pow( r, j ) *
                        Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
                }
            }
            // printf("t%d, cell(%d, %d, %d), center(%.2lf, %.2lf, %.2lf), lp: %.3lf\n", tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2], local_potential(tpi).real());
            // printf("ppp%d, t%d, ijk(%d, %d, %d), c(%.2lf, %.2lf, %.2lf), dp: %.7lf, lp: %.7lf\n", points_per_proc, tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2], direct_potentials(tpi), local_potential(tpi).real());
        } );
    Kokkos::fence();

    // Copy to host and test
    auto is_activated_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), is_activated);
    int p_int = static_cast<int>(p_val);
    auto local_potential_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_potential);
    for (std::size_t i = 0; i < local_potential.extent(0); i++)
    {
        if (!is_activated_host(i))
            continue;

        auto direct_potential = direct_potentials(i);
        auto local_potential = local_potential_h(i).real();
        ASSERT_NEAR(local_potential, direct_potential, Kokkos::pow(10, -p_int+2));
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
    using particle_tuple_type = Cabana::MemberTypes<double[3], double>;
    using particle_aosoa_type = Cabana::AoSoA<particle_tuple_type, TEST_MEMSPACE, 4>;
    std::array<double, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<double, 3> global_high_corner = { 3.0, 3.0, 3.0 };

    static constexpr std::size_t num_dim = 3;
    static constexpr std::size_t cells_per_tile = 2;
    static constexpr std::size_t p = p_val;
    std::size_t leaf_tiles, red_factor;
    red_factor = comm_size, leaf_tiles = comm_size * 32;
    if (red_factor < 2) red_factor = 2;
    auto tree = Canopy::createTree<TEST_EXECSPACE, TEST_MEMSPACE, Cabana::Grid::Cell,
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
    int points_per_proc = points_per_proc_in;
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
    
    // Kokkos::Array<double, 6> coord_bounds1 = {2.3, 2.3, 2.3, bound_val, bound_val, bound_val};
    fillRandomCoordinates(cart_coords, coord_bounds, 123);
    fillRandomScalar(q, charge_bounds, 321);

    // Put first point in the same cell as the target point
    cart_coords(0, 0) = -2.9;
    cart_coords(0, 1) = -2.8;
    cart_coords(0, 2) = -2.85;
    q(0) = 0.0;

    // Insert at (3, 2, 2) on layer 1
    cart_coords(1, 0) = 2.62;
    cart_coords(1, 1) = 1.17;
    cart_coords(1, 2) = 1.30;
    q(1) = 10.0;

    // Insert at (3, 3, 3) on layer 1
    cart_coords(2, 0) = 2.9;
    cart_coords(2, 1) = 2.8;
    cart_coords(2, 2) = 2.85;
    q(2) = 10.0;

    // Insert at (3, 2, 3) on layer 1
    cart_coords(3, 0) = 2.9;
    cart_coords(3, 1) = 1.17;
    cart_coords(3, 2) = 2.85;
    q(1) = 10.0;

    // cart_coords(2, 0) = 2.9;
    // cart_coords(2, 1) = 2.8;
    // cart_coords(2, 2) = 2.85;
    // q(2) = 10.0;

    // cart_coords(3, 0) = -0.76;
    // cart_coords(3, 1) = -1.47;
    // cart_coords(3, 2) = -2.45;
    // q(3) = 0.0;

    // q(4) = 0.0;
    // q(5) = 0.0;
    // q(6) = 0.0;
    // q(7) = 0.0;
    // q(8) = 0.0;
    // q(9) = 0.0;
    // q(10) = 0.0;
    // q(11) = 0.0;

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

    // Just test single-layer multipole to local conversion.
    tree->multipole_to_local();

    // Create target points at which to calculate potential directly, omitting
    // nearest and second-nearest neighbor cells.
    int num_target_points = 20;
    Kokkos::View<double*[3], TEST_MEMSPACE> target_points( "target_points",
                                                          num_target_points );
    fillRandomCoordinates(target_points, coord_bounds, 456);

    // Put target point in lower corner of domain
    target_points(0, 0) = -2.88;
    target_points(0, 1) = -2.92;
    target_points(0, 2) = -2.89;

    // Insert at (3, 2, 2) on layer 1
    target_points(1, 0) = 2.61;
    target_points(1, 1) = 1.18;
    target_points(1, 2) = 1.31;
    
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
            inner_upper_bound[dim] = Kokkos::min(static_cast<int>(target_cell_ijk[dim]) + 3, cells_per_dimension_leaf);
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

            // if (target_cell_ijk[0] == 14 && target_cell_ijk[1] == 12 && target_cell_ijk[2] == 5)
            //     printf("R%d: cell %d, %d, %d, neighbor %d, %d, %d (op %d)\n", rank,
            //         target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //         cell_ijk[0], cell_ijk[1], cell_ijk[2], op);
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
    auto leaf_layer = tree->layer(0);
    auto locals = leaf_layer->locals();
    auto ijk2index = leaf_layer->cellijk2l();
    auto locals_slice = Cabana::slice<0>(locals);

    // Track which cells are activated in the mesh. If a target point is in a non-
    // activated cell, skip it when checking potentials
    Kokkos::View<int*, TEST_MEMSPACE> is_activated("is_activated", num_target_points);
    Kokkos::deep_copy(is_activated, 0);

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
            auto cell_exists = ijk2index.exists(target_cell_ijk);
            if (!cell_exists)
                return;
            
            // Set cell to activated
            is_activated(tpi) = 1;

             // Center of local expansion is the cell center
            Kokkos::Array<double, 3> l_center;
            for (int i = 0; i < 3; i++)
                l_center[i] = global_low_corner_k[i] + (static_cast<double>(target_cell_ijk[i]) + 0.5) * cell_size[i];

            // printf("t%d, ijk(%d, %d, %d), c(%.2lf, %.2lf, %.2lf)\n", tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2]);


            // Convert target point to spherical coordinates relative to local center
            double r, theta, phi;
            Canopy::Kernel::cart2sph( target_points(tpi, 0) - l_center[0],
                                      target_points(tpi, 1) - l_center[1],
                                      target_points(tpi, 2) - l_center[2],
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
            for ( int j = 0; j <= p_val; ++j )
            {
                for ( int k = -j; k <= j; ++k )
                {
                    int idx = Canopy::Kernel::Scalar::index( j, k );

                    /* Target point 1 calculations */
                    // Greengard eq. 3.59
                    cdouble val = cdouble(locals_slice(local_index, idx, 0), locals_slice(local_index, idx, 1));
                    local_potential(tpi) +=
                        val * Kokkos::pow( r, j ) *
                        Canopy::Kernel::Scalar::Ynm( j, k, theta, phi );
                }
            }
            // printf("t%d, cell(%d, %d, %d), center(%.2lf, %.2lf, %.2lf), lp: %.3lf\n", tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2], local_potential(tpi).real());
            // printf("ppp%d, t%d, ijk(%d, %d, %d), c(%.2lf, %.2lf, %.2lf), dp: %.5lf, lp: %.5lf\n", points_per_proc, tpi,
            //     target_cell_ijk[0], target_cell_ijk[1], target_cell_ijk[2],
            //     l_center[0], l_center[1], l_center[2], direct_potentials(tpi), local_potential(tpi).real());
            // L1: R0: cell 3, 2, 3, neighbor 0, 0, 2: L: 7442.639, -21492.622, 33200.079
        } );
    Kokkos::fence();

    // Copy to host and test
    auto is_activated_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), is_activated);
    int p_int = static_cast<int>(p_val);
    auto local_potential_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_potential);
    for (std::size_t i = 0; i < local_potential.extent(0); i++)
    {
        if (!is_activated_host(i))
            continue;
        
        // Get the cell this point falls into for error printing
        Kokkos::Array<std::size_t, 3> target_cell_ijk;
        for (int dim = 0; dim < 3; ++dim)
        {
            target_cell_ijk[dim] = static_cast<std::size_t>(
                Kokkos::floor((target_points_host(i, dim) - global_low_corner[dim]) / cell_size[dim]) );
        }

        auto direct_potential = direct_potentials(i);
        auto local_potential = local_potential_h(i).real();
        double allowed_error = Kokkos::pow(10, -p_int+4);
        EXPECT_NEAR(local_potential, direct_potential, allowed_error) << " at cell ijk ("
            << target_cell_ijk[0] << ", " << target_cell_ijk[1] << ", "
            << target_cell_ijk[2] << ")";
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

// TEST( Tree, testMultipole2Local2_balanced )
// { 
//     for (int i = 4; i < 10; i++)
//     {
//         printf("******* 2: i = %d *******\n", i);
//         testMultipole2Local2<6>(i, true); 
//     }
// }

//---------------------------------------------------------------------------//

} // end namespace Test