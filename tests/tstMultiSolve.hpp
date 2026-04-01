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

#include <random>

namespace Test
{
//---------------------------------------------------------------------------//

// Define input aosoa data including velocity and force.
// pos/force/mass/potential/velocity/global particle id
using particle_tuple_type_mv =
    Cabana::MemberTypes<scalar_type[3], scalar_type[3], scalar_type,
                        scalar_type, scalar_type[3], int>;
using particle_aosoa_type_mv =
    Cabana::AoSoA<particle_tuple_type_mv, TEST_MEMSPACE, 4>;
using particle_aosoa_type_mv_h =
    Cabana::AoSoA<particle_tuple_type_mv, Kokkos::HostSpace, 4>;
using MD_mv =
    Canopy::ParticleMetadata<particle_aosoa_type_mv, scalar_type, 0, 2, 3, 1>;


/**
 * Regression test that advances an n-body system for multiple timesteps and
 * compares final exact and Canopy particle positions.
 */
template <int p>
void testSolver(int points_per_proc_in, bool balanced, int num_timesteps)
{
    static_assert(p > 0);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    (void)comm_size;

    // Create a tree of depth 3.
    std::array<scalar_type, 3> global_low_corner = { -3.0, -3.0, -3.0 };
    std::array<scalar_type, 3> global_high_corner = { 3.0, 3.0, 3.0 };

    static constexpr std::size_t cells_per_tile = 2;
    std::size_t leaf_tiles = 16;
    std::size_t red_factor = 2;
    auto tree =
        Canopy::createSolver<TEST_MEMSPACE, TEST_EXECSPACE, MD_mv,
                             cells_per_tile, p>( global_low_corner,
                                                 global_high_corner, leaf_tiles,
                                                 red_factor, MPI_COMM_WORLD );

    int total_points = points_per_proc_in;
    int owned_points = ( rank == 0 ) ? total_points : 0;

    particle_aosoa_type_mv_h particle_aosoa_host( "particle_aosoa",
                                                  owned_points );
    auto pos_slice_host = Cabana::slice<MD_mv::pos>( particle_aosoa_host );
    auto force_slice_host = Cabana::slice<MD_mv::force>( particle_aosoa_host );
    auto mass_slice_host = Cabana::slice<MD_mv::in>( particle_aosoa_host );
    auto potential_slice_host = Cabana::slice<MD_mv::out>( particle_aosoa_host );
    auto velocity_slice_host = Cabana::slice<4>( particle_aosoa_host );
    auto id_slice_host = Cabana::slice<5>( particle_aosoa_host );
    Cabana::deep_copy( force_slice_host, 0.0 );
    Cabana::deep_copy( potential_slice_host, 0.0 );
    Cabana::deep_copy( velocity_slice_host, 0.0 );

    Kokkos::View<scalar_type* [3], Kokkos::HostSpace> exact_positions(
        "exact_positions", owned_points );
    Kokkos::View<scalar_type* [3], Kokkos::HostSpace> exact_velocities(
        "exact_velocities", owned_points );
    Kokkos::View<scalar_type* [3], Kokkos::HostSpace> exact_forces(
        "exact_forces", owned_points );
    Kokkos::View<scalar_type*, Kokkos::HostSpace> exact_masses( "exact_masses",
                                                                owned_points );
    Kokkos::deep_copy( exact_velocities, 0.0 );
    Kokkos::deep_copy( exact_forces, 0.0 );

    const scalar_type domain_padding = 1.0e-3;
    const scalar_type time_step = 1.0e-3;
    const scalar_type position_tolerance = balanced ? 2.5e-2 : 4.0e-2;
    const scalar_type placement_margin = balanced ? 0.25 : 0.2;
    const scalar_type unbalanced_half_width = 0.1;
    const scalar_type minimum_separation = balanced ? 0.08 : 0.05;
    const scalar_type minimum_separation_sq =
        minimum_separation * minimum_separation;

    if ( rank == 0 )
    {
        std::mt19937 rng( balanced ? 24680 : 13579 );
        std::uniform_real_distribution<scalar_type> x_dist(
            global_low_corner[0] + placement_margin,
            global_high_corner[0] - placement_margin );
        std::uniform_real_distribution<scalar_type> y_dist(
            global_low_corner[1] + placement_margin,
            global_high_corner[1] - placement_margin );
        std::uniform_real_distribution<scalar_type> z_dist(
            balanced ? ( global_low_corner[2] + placement_margin )
                     : -unbalanced_half_width,
            balanced ? ( global_high_corner[2] - placement_margin )
                     : unbalanced_half_width );
        std::uniform_real_distribution<scalar_type> mass_dist( 0.25, 1.25 );

        for ( int i = 0; i < owned_points; ++i )
        {
            scalar_type x = 0.0;
            scalar_type y = 0.0;
            scalar_type z = 0.0;
            bool accepted = false;
            for ( int attempt = 0; attempt < 512 && !accepted; ++attempt )
            {
                x = x_dist( rng );
                y = y_dist( rng );
                z = z_dist( rng );
                accepted = true;
                for ( int j = 0; j < i; ++j )
                {
                    const scalar_type dx = exact_positions( j, 0 ) - x;
                    const scalar_type dy = exact_positions( j, 1 ) - y;
                    const scalar_type dz = exact_positions( j, 2 ) - z;
                    const scalar_type dist_sq = dx * dx + dy * dy + dz * dz;
                    if ( dist_sq < minimum_separation_sq )
                    {
                        accepted = false;
                        break;
                    }
                }
            }

            const scalar_type mass = mass_dist( rng );

            exact_positions( i, 0 ) = x;
            exact_positions( i, 1 ) = y;
            exact_positions( i, 2 ) = z;
            exact_masses( i ) = mass;

            pos_slice_host( i, 0 ) = x;
            pos_slice_host( i, 1 ) = y;
            pos_slice_host( i, 2 ) = z;
            mass_slice_host( i ) = mass;
            id_slice_host( i ) = i;
        }
    }

    auto canopy_particles = std::make_shared<particle_aosoa_type_mv>(
        "canopy_particles", particle_aosoa_host.size() );
    Cabana::deep_copy( *canopy_particles, particle_aosoa_host );

    const bool run_load_balance = !balanced;

    auto clamp_position = [&]( scalar_type& pos, scalar_type& vel,
                               scalar_type low, scalar_type high ) {
        if ( pos < low )
        {
            pos = low;
            vel *= -1.0;
        }
        else if ( pos > high )
        {
            pos = high;
            vel *= -1.0;
        }
    };

    for ( int step = 0; step < num_timesteps; ++step )
    {
        if ( rank == 0 )
        {
            Kokkos::deep_copy( exact_forces, 0.0 );
            for ( int this_pid = 0; this_pid < owned_points; ++this_pid )
            {
                for ( int other_pid = 0; other_pid < owned_points; ++other_pid )
                {
                    if ( this_pid == other_pid )
                        continue;

                    const scalar_type dx =
                        exact_positions( other_pid, 0 ) -
                        exact_positions( this_pid, 0 );
                    const scalar_type dy =
                        exact_positions( other_pid, 1 ) -
                        exact_positions( this_pid, 1 );
                    const scalar_type dz =
                        exact_positions( other_pid, 2 ) -
                        exact_positions( this_pid, 2 );
                    const scalar_type dist_sq = dx * dx + dy * dy + dz * dz;

                    if ( dist_sq == 0.0 )
                        continue;

                    const scalar_type dist = Kokkos::sqrt( dist_sq );
                    const scalar_type dist_inv = 1.0 / dist;
                    const scalar_type dist_inv3 =
                        dist_inv * dist_inv * dist_inv;
                    const scalar_type fp =
                        -1.0 * exact_masses( this_pid ) *
                        exact_masses( other_pid ) * dist_inv3;

                    exact_forces( this_pid, 0 ) += fp * dx;
                    exact_forces( this_pid, 1 ) += fp * dy;
                    exact_forces( this_pid, 2 ) += fp * dz;
                }
            }

            for ( int pid = 0; pid < owned_points; ++pid )
            {
                const scalar_type inv_mass = 1.0 / exact_masses( pid );
                for ( int dim = 0; dim < 3; ++dim )
                {
                    exact_velocities( pid, dim ) +=
                        time_step * exact_forces( pid, dim ) * inv_mass;
                    exact_positions( pid, dim ) +=
                        time_step * exact_velocities( pid, dim );
                }

                clamp_position( exact_positions( pid, 0 ),
                                exact_velocities( pid, 0 ),
                                global_low_corner[0] + domain_padding,
                                global_high_corner[0] - domain_padding );
                clamp_position( exact_positions( pid, 1 ),
                                exact_velocities( pid, 1 ),
                                global_low_corner[1] + domain_padding,
                                global_high_corner[1] - domain_padding );
                clamp_position( exact_positions( pid, 2 ),
                                exact_velocities( pid, 2 ),
                                global_low_corner[2] + domain_padding,
                                global_high_corner[2] - domain_padding );
            }
        }

        tree->solve( canopy_particles, run_load_balance );

        auto positions = Cabana::slice<MD_mv::pos>( *canopy_particles );
        auto masses = Cabana::slice<MD_mv::in>( *canopy_particles );
        auto forces = Cabana::slice<MD_mv::force>( *canopy_particles );
        auto velocities = Cabana::slice<4>( *canopy_particles );
        const scalar_type low_x = global_low_corner[0] + domain_padding;
        const scalar_type low_y = global_low_corner[1] + domain_padding;
        const scalar_type low_z = global_low_corner[2] + domain_padding;
        const scalar_type high_x = global_high_corner[0] - domain_padding;
        const scalar_type high_y = global_high_corner[1] - domain_padding;
        const scalar_type high_z = global_high_corner[2] - domain_padding;

        Kokkos::parallel_for(
            "Test::MultiSolve::advance_canopy_particles",
            Kokkos::RangePolicy<TEST_EXECSPACE>( 0, canopy_particles->size() ),
            KOKKOS_LAMBDA( const int i ) {
                const scalar_type inv_mass = 1.0 / masses( i );
                for ( int dim = 0; dim < 3; ++dim )
                {
                    velocities( i, dim ) +=
                        time_step * forces( i, dim ) * inv_mass;
                    positions( i, dim ) += time_step * velocities( i, dim );
                }

                if ( positions( i, 0 ) < low_x )
                {
                    positions( i, 0 ) = low_x;
                    velocities( i, 0 ) *= -1.0;
                }
                else if ( positions( i, 0 ) > high_x )
                {
                    positions( i, 0 ) = high_x;
                    velocities( i, 0 ) *= -1.0;
                }

                if ( positions( i, 1 ) < low_y )
                {
                    positions( i, 1 ) = low_y;
                    velocities( i, 1 ) *= -1.0;
                }
                else if ( positions( i, 1 ) > high_y )
                {
                    positions( i, 1 ) = high_y;
                    velocities( i, 1 ) *= -1.0;
                }

                if ( positions( i, 2 ) < low_z )
                {
                    positions( i, 2 ) = low_z;
                    velocities( i, 2 ) *= -1.0;
                }
                else if ( positions( i, 2 ) > high_z )
                {
                    positions( i, 2 ) = high_z;
                    velocities( i, 2 ) *= -1.0;
                }
            } );
        Kokkos::fence();
    }

    auto tmp =
        Cabana::create_mirror_view_and_copy( Kokkos::HostSpace(), *( tree->data() ) );
    particle_aosoa_type_mv_h tree_particles( "tree_particles", tmp.size() );
    Cabana::deep_copy( tree_particles, tmp );

    tree_particles.resize( tree->numOwnedParticles() );

    Kokkos::View<int*, Kokkos::HostSpace> send_to( "send_to",
                                                   tree->numOwnedParticles() );
    Kokkos::deep_copy( send_to, 0 );
    Cabana::Distributor<Kokkos::HostSpace> distributor( MPI_COMM_WORLD, send_to );
    Cabana::migrate( distributor, tree_particles );

    auto tree_id_slice = Cabana::slice<5>( tree_particles );
    auto sort_data = Cabana::sortByKey( tree_id_slice );
    Cabana::permute( sort_data, tree_particles );
    tree_id_slice = Cabana::slice<5>( tree_particles );

    if ( rank == 0 )
    {
        ASSERT_EQ( static_cast<int>( tree_particles.size() ), owned_points );

        auto tree_positions = Cabana::slice<MD_mv::pos>( tree_particles );
        double max_position_error = 0.0;
        for ( int i = 0; i < owned_points; ++i )
        {
            const int particle_id = tree_id_slice( i );
            for ( int dim = 0; dim < 3; ++dim )
            {
                const scalar_type exact_position =
                    exact_positions( particle_id, dim );
                const scalar_type canopy_position =
                    tree_positions( i, dim );
                EXPECT_NEAR( canopy_position, exact_position,
                             position_tolerance )
                    << " at particle " << particle_id << " dim " << dim;

                const double error =
                    Kokkos::abs( canopy_position - exact_position );
                if ( error > max_position_error )
                    max_position_error = error;
            }
        }

        if ( owned_points > 0 )
            printf( "Max multi-step position error: %.6lf\n",
                    max_position_error );
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( MultiSolve, testMultiSolve_balanced )
{ 
    testSolver<4>(300, true, 8);
}
TEST( MultiSolve, testMultiSolve_unbalanced )
{ 
    testSolver<4>(300, false, 8);
}

//---------------------------------------------------------------------------//

} // end namespace Test
