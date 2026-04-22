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

#include <Canopy_Experimental_TreeBuilder.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdio>
#include <random>

using namespace Canopy::Experimental;

// ============================================================================
// Example: Adaptive octree with incremental updates over timesteps
//
// Demonstrates:
//   1. Initial full tree build
//   2. Small particle displacements absorbed by tolerance (no tree change)
//   3. Moderate displacements causing local refinement/coarsening
//   4. Large displacements triggering a full rebuild
// ============================================================================

// Define particle data layout
enum FieldIdx
{
    Position = 0
};

using DataTypes = Cabana::MemberTypes<double[3]>;

using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

using AoSoA_t = Cabana::AoSoA<DataTypes, MemorySpace>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

// ============================================================================
// Generate test particles with a non-uniform distribution
// ============================================================================
void generate_test_particles( AoSoA_t& particles, int num_particles,
                              int rank, int /* nprocs */ )
{
    AoSoA_ht particles_h("particles_h", num_particles);
    auto h_positions = Cabana::slice<Position>(particles_h);

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> uniform( 0.0, 1.0 );
    std::normal_distribution<double> clustered( 0.1, 0.02 );

    for ( int i = 0; i < num_particles; ++i )
    {
        if ( uniform( gen ) < 0.3 )
        {
            h_positions( i, 0 ) =
                std::clamp( clustered( gen ), 0.0, 1.0 );
            h_positions( i, 1 ) =
                std::clamp( clustered( gen ), 0.0, 1.0 );
            h_positions( i, 2 ) =
                std::clamp( clustered( gen ), 0.0, 1.0 );
        }
        else
        {
            h_positions( i, 0 ) = uniform( gen );
            h_positions( i, 1 ) = uniform( gen );
            h_positions( i, 2 ) = uniform( gen );
        }
    }

    particles.resize( num_particles );

    Cabana::deep_copy( particles, particles_h );
}

// ============================================================================
// Displace particles by a given magnitude (simulating a timestep)
// ============================================================================
void displace_particles( AoSoA_t& particles, int num_particles,
                         double displacement_magnitude, int seed )
{
    AoSoA_ht particles_h("particles_h", num_particles);
    Cabana::deep_copy(particles_h, particles);
    auto h_positions = Cabana::slice<Position>(particles_h);

    std::mt19937 gen( seed );
    std::normal_distribution<double> disp( 0.0, displacement_magnitude );

    for ( int i = 0; i < num_particles; ++i )
    {
        for ( int d = 0; d < 3; ++d )
        {
            h_positions( i, d ) += disp( gen );
            // Don't clamp — let particles escape the box if they move
            // far enough, which tests the full-rebuild path
        }
    }

    Cabana::deep_copy( particles, particles_h );
}

// ============================================================================
// Print tree statistics
// ============================================================================
void print_tree_stats( const std::vector<CellInfo>& cells, int rank,
                       const char* label )
{
    if ( rank != 0 )
        return;

    int total_cells = static_cast<int>( cells.size() );
    int num_leaves = 0;
    int num_internal = 0;
    int max_depth = 0;
    int total_particles_in_leaves = 0;

    std::map<int, int> cells_per_depth;

    for ( const auto& c : cells )
    {
        cells_per_depth[c.depth]++;
        if ( c.depth > max_depth )
            max_depth = c.depth;

        if ( c.is_leaf )
        {
            num_leaves++;
            total_particles_in_leaves += c.global_count;
        }
        else
        {
            num_internal++;
        }
    }

    std::printf( "\n--- %s ---\n", label );
    std::printf( "  Total cells: %d (internal: %d, leaves: %d)\n",
                 total_cells, num_internal, num_leaves );
    std::printf( "  Max depth: %d, Particles in leaves: %d\n",
                 max_depth, total_particles_in_leaves );
}

// ============================================================================
// Main
// ============================================================================
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    Kokkos::initialize( argc, argv );
    {
        // Parameters
        int num_particles_per_rank = 10000;
        int ncrit = 128;
        int max_depth = 15;
        double tolerance = 0.1; // 10% buffer on bounding box and leaf
                                // cell particle count.
        int num_timesteps = 20;

        if ( argc > 1 )
            num_particles_per_rank = std::atoi( argv[1] );
        if ( argc > 2 )
            ncrit = std::atoi( argv[2] );
        if ( argc > 3 )
            max_depth = std::atoi( argv[3] );
        if ( argc > 4 )
            tolerance = std::atof( argv[4] );

        if ( rank == 0 )
        {
            std::printf(
                "Running with %d ranks, %d particles/rank, "
                "ncrit=%d, max_depth=%d, bb tol=%.3f, leaf tol=%.3f\n",
                nprocs, num_particles_per_rank, ncrit, max_depth,
                tolerance, tolerance );
        }

        // Create and fill particles
        AoSoA_t particles( "particles", num_particles_per_rank );
        generate_test_particles( particles, num_particles_per_rank, rank,
                                 nprocs );

        auto positions = Cabana::slice<Position>( particles );

        // Build the tree builder with tolerance
        // using PositionSlice = decltype( positions );
        TreeBuilder<MemorySpace, ExecutionSpace> builder(
            MPI_COMM_WORLD, ncrit, max_depth, tolerance, tolerance );

        // -----------------------------------------------------------
        // Initial full build
        // -----------------------------------------------------------
        builder.build( positions, num_particles_per_rank );
        print_tree_stats( builder.cells(), rank, "Initial build" );

        // -----------------------------------------------------------
        // Simulate timesteps with increasing displacement
        // -----------------------------------------------------------
        for ( int step = 1; step <= num_timesteps; ++step )
        {
            // Displacement grows over time to exercise all code paths:
            //   Steps 1-5:   tiny (well within tolerance)
            //   Steps 6-12:  moderate (some cells refine/coarsen)
            //   Steps 13-20: large (may trigger full rebuild)
            double disp;
            if ( step <= 5 )
                disp = 0.0001; // tiny
            else if ( step <= 12 )
                disp = 0.005; // moderate
            else
                disp = 0.05; // large

            displace_particles( particles, num_particles_per_rank, disp,
                                step * 137 + rank );

            // Re-slice after potential resize (not strictly needed here
            // since we don't change particle count, but good practice)
            positions = Cabana::slice<Position>( particles );

            // Incremental update
            auto result =
                builder.update( positions, num_particles_per_rank );

            if ( rank == 0 )
            {
                char label[256];
                if ( result.full_rebuild_done )
                {
                    std::snprintf( label, sizeof( label ),
                                   "Step %2d (disp=%.4f): FULL REBUILD",
                                   step, disp );
                }
                else
                {
                    std::snprintf(
                        label, sizeof( label ),
                        "Step %2d (disp=%.4f): migrated=%d "
                        "refined=%d coarsened=%d",
                        step, disp, result.particles_migrated,
                        result.cells_refined, result.cells_coarsened );
                }
                print_tree_stats( builder.cells(), rank, label );
            }
        }

        // -----------------------------------------------------------
        // Final verification
        // -----------------------------------------------------------
        auto particle_keys = builder.particle_keys();
        auto h_keys = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), particle_keys );

        std::set<MortonKey> leaf_keys;
        for ( const auto& c : builder.cells() )
        {
            if ( c.is_leaf )
                leaf_keys.insert( c.key );
        }

        int bad_count = 0;
        for ( int i = 0; i < num_particles_per_rank; ++i )
        {
            if ( leaf_keys.find( h_keys( i ) ) == leaf_keys.end() )
            {
                bad_count++;
                if ( bad_count <= 5 )
                {
                    std::printf(
                        "Rank %d: particle %d has key %lu which is "
                        "not a leaf cell!\n",
                        rank, i,
                        static_cast<unsigned long>( h_keys( i ) ) );
                }
            }
        }

        if ( bad_count > 0 )
        {
            std::printf( "Rank %d: %d particles not in leaf cells!\n",
                         rank, bad_count );
        }
        else if ( rank == 0 )
        {
            std::printf(
                "\nVerification passed: all particles are in "
                "leaf cells after %d timesteps.\n",
                num_timesteps );
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
