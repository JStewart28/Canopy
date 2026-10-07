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

#include <Canopy_TreeBuilder.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy;

namespace TreeBuilderTest
{

enum FieldIdx
{
    Position = 0
};

using DataTypes = Cabana::MemberTypes<double[3]>;
using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

// Generate test particles with a non-uniform (partially clustered) distribution
void generate_test_particles( AoSoA_t& particles, int num_particles, int rank )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    auto h_positions = Cabana::slice<Position>( particles_h );

    std::mt19937 gen( 42 + rank );
    std::uniform_real_distribution<double> uniform( 0.0, 1.0 );
    std::normal_distribution<double> clustered( 0.1, 0.02 );

    for ( int i = 0; i < num_particles; ++i )
    {
        if ( uniform( gen ) < 0.3 )
        {
            h_positions( i, 0 ) = std::clamp( clustered( gen ), 0.0, 1.0 );
            h_positions( i, 1 ) = std::clamp( clustered( gen ), 0.0, 1.0 );
            h_positions( i, 2 ) = std::clamp( clustered( gen ), 0.0, 1.0 );
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

// Displace particles by a given magnitude (simulating a timestep). Particles
// are not clamped, so large displacements can push them outside the initial
// bounding box and trigger a full rebuild.
void displace_particles( AoSoA_t& particles, int num_particles,
                         double displacement_magnitude, int seed )
{
    AoSoA_ht particles_h( "particles_h", num_particles );
    Cabana::deep_copy( particles_h, particles );
    auto h_positions = Cabana::slice<Position>( particles_h );

    std::mt19937 gen( seed );
    std::normal_distribution<double> disp( 0.0, displacement_magnitude );

    for ( int i = 0; i < num_particles; ++i )
    {
        for ( int d = 0; d < 3; ++d )
            h_positions( i, d ) += disp( gen );
    }

    Cabana::deep_copy( particles, particles_h );
}

// Count how many particles map to a key that is actually a leaf cell.
int count_particles_in_leaves(
    const std::vector<CellInfo>& cells,
    const Kokkos::View<const MortonKey*, Kokkos::HostSpace>& h_keys,
    int num_particles )
{
    std::set<MortonKey> leaf_keys;
    for ( const auto& c : cells )
    {
        if ( c.is_leaf )
            leaf_keys.insert( c.key );
    }

    int bad_count = 0;
    for ( int i = 0; i < num_particles; ++i )
    {
        if ( leaf_keys.find( h_keys( i ) ) == leaf_keys.end() )
            bad_count++;
    }
    return bad_count;
}

} // namespace TreeBuilderTest

//---------------------------------------------------------------------------//
/**
 * Test incremental adaptive octree construction across several timesteps.
 * Particles are displaced by increasing magnitudes to exercise the
 * no-op/refine/coarsen/full-rebuild code paths in TreeBuilder::update().
 * After every step, every particle must map to a leaf cell of the tree.
 */
void testTreeBuilder( int num_particles_per_rank, int ncrit, int max_depth,
                      std::array<double, 6> bb_tol, double ncrit_tol, int num_timesteps )
{
    using namespace TreeBuilderTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        MPI_COMM_WORLD, ncrit, max_depth, bb_tol, ncrit_tol );

    // Initial full build
    builder.build( positions, num_particles_per_rank );

    // Every particle must be in a leaf after the initial build.
    {
        auto h_keys = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), builder.particle_keys() );
        int bad = count_particles_in_leaves( builder.cells(), h_keys,
                                             num_particles_per_rank );
        ASSERT_EQ( bad, 0 )
            << "After initial build, " << bad
            << " particles are not in leaf cells (rank " << rank << ")";
    }

    // Verify at least one leaf exists and all leaves collectively cover
    // every particle on this rank.
    {
        int num_leaves = 0;
        for ( const auto& c : builder.cells() )
            if ( c.is_leaf )
                ++num_leaves;
        ASSERT_GT( num_leaves, 0 ) << "Initial tree has no leaves";
    }

    // Simulate timesteps with increasing displacement magnitudes:
    //   steps 1-5  : tiny    (well within tolerance, typically no change)
    //   steps 6-12 : moderate (some local refine/coarsen)
    //   steps 13-N : large   (may trigger a full rebuild)
    for ( int step = 1; step <= num_timesteps; ++step )
    {
        double disp;
        if ( step <= 5 )
            disp = 0.0001;
        else if ( step <= 12 )
            disp = 0.005;
        else
            disp = 0.05;

        displace_particles( particles, num_particles_per_rank, disp,
                            step * 137 + rank );

        positions = Cabana::slice<Position>( particles );

        auto result = builder.update( positions, num_particles_per_rank );

        // Update counters should be non-negative.
        EXPECT_GE( result.particles_migrated, 0 );
        EXPECT_GE( result.cells_refined, 0 );
        EXPECT_GE( result.cells_coarsened, 0 );

        auto h_keys = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), builder.particle_keys() );
        int bad = count_particles_in_leaves( builder.cells(), h_keys,
                                             num_particles_per_rank );
        ASSERT_EQ( bad, 0 )
            << "After step " << step << " (disp=" << disp << ", rank " << rank
            << "), " << bad << " particles are not in leaf cells";
    }
}

//---------------------------------------------------------------------------//
// B2 of tasks/tree-opt.md: TreeBuilder::quantized_half_width is exact. Over a
// sweep of inputs the result is a power of two, >= the input, < 2x the input,
// and the input itself when that is already a power of two. Then one build per
// knob state: root_half_width() is what build() stamped on the root cell, it
// is today's max half extent of root_box() knob off and that value quantized
// knob on, the centre is unchanged, and every particle is still in a leaf.
//---------------------------------------------------------------------------//
void testQuantizedRootHalfWidth()
{
    using namespace TreeBuilderTest;
    using builder_type = TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE>;

    const auto is_pow2 = []( double x )
    {
        int e = 0;
        return std::frexp( x, &e ) == 0.5;
    };

    // Inputs: every power of two in [2^-40, 2^40], its two neighbours, and
    // random mantissas at each exponent.
    std::vector<double> inputs;
    std::mt19937 gen( 4242 );
    std::uniform_real_distribution<double> mant( 0.5, 1.0 );
    for ( int e = -40; e <= 40; ++e )
    {
        const double p = std::ldexp( 1.0, e );
        inputs.push_back( p );
        inputs.push_back( std::nextafter( p, 0.0 ) );
        inputs.push_back( std::nextafter( p, 2.0 * p ) );
        for ( int k = 0; k < 16; ++k )
            inputs.push_back( std::ldexp( mant( gen ), e ) );
    }
    for ( const double hw : inputs )
    {
        const double q = builder_type::quantized_half_width( hw );
        EXPECT_TRUE( is_pow2( q ) ) << "hw " << hw << " -> " << q;
        EXPECT_GE( q, hw ) << "rounded down: hw " << hw << " -> " << q;
        EXPECT_LT( q, 2.0 * hw ) << "hw " << hw << " -> " << q;
        if ( is_pow2( hw ) )
            EXPECT_EQ( q, hw ) << "a power of two moved: " << hw;
    }
    EXPECT_EQ( builder_type::quantized_half_width( 0.0 ), 0.0 );

    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    const int n = 500;
    AoSoA_t particles( "particles", n );
    generate_test_particles( particles, n, rank );
    auto positions = Cabana::slice<Position>( particles );
    const std::array<double, 6> tol{ 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };

    for ( const bool quantize : { false, true } )
    {
        builder_type builder( MPI_COMM_WORLD, 32, 10, tol, 0.1, quantize );
        builder.build( positions, n );

        const auto& box = builder.root_box();
        double box_hw = 0.0;
        for ( int d = 0; d < 3; ++d )
            box_hw = std::max( box_hw, 0.5 * ( box.max[d] - box.min[d] ) );
        const double expect =
            quantize ? builder_type::quantized_half_width( box_hw ) : box_hw;
        EXPECT_EQ( builder.root_half_width(), expect )
            << "quantize " << quantize;
        if ( quantize )
            EXPECT_TRUE( is_pow2( builder.root_half_width() ) );

        bool found = false;
        for ( const auto& c : builder.cells() )
        {
            if ( c.key != ROOT_KEY )
                continue;
            found = true;
            EXPECT_EQ( c.half_width, builder.root_half_width() )
                << "quantize " << quantize;
            for ( int d = 0; d < 3; ++d )
                EXPECT_EQ( c.center[d], 0.5 * ( box.min[d] + box.max[d] ) )
                    << "quantize " << quantize << " axis " << d;
        }
        EXPECT_TRUE( found );

        auto h_keys = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), builder.particle_keys() );
        EXPECT_EQ( count_particles_in_leaves( builder.cells(), h_keys, n ), 0 )
            << "quantize " << quantize;
    }
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( TreeBuilder, testIncrementalUpdates )
{
    testTreeBuilder( 10000, 128, 15, std::array<double, 6>{0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 0.1, 20 );
}

TEST( TreeBuilder, testSmallTree ) { testTreeBuilder( 500, 32, 10, std::array<double, 6>{0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 0.1, 10 ); }

TEST( TreeBuilder, quantizedRootHalfWidth ) { testQuantizedRootHalfWidth(); }

//---------------------------------------------------------------------------//

} // end namespace Test
