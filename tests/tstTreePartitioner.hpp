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
#include <Canopy_Experimental_TreePartitioner.hpp>

#include <test_helpers.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include <mpi.h>

#include <random>
#include <set>

namespace Test
{
//---------------------------------------------------------------------------//

using namespace Canopy::Experimental;

namespace TreePartitionerTest
{

enum FieldIdx
{
    Position = 0
};

using DataTypes = Cabana::MemberTypes<double[3]>;
using AoSoA_t = Cabana::AoSoA<DataTypes, TEST_MEMSPACE>;
using AoSoA_ht = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;

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

} // namespace TreePartitionerTest

//---------------------------------------------------------------------------//
/**
 * Build an adaptive octree, partition its leaves across MPI ranks with
 * TreePartitioner, then verify:
 *
 *   1. The ownership vector has one entry per cell in the tree.
 *   2. Cells at depth <= replication_depth are OWNER_SHARED; all deeper
 *      cells have an owner in [0, nprocs).
 *   3. The total particle count is conserved across ranks after migration.
 *   4. After rebuilding the tree on the new particle distribution, every
 *      particle's leaf is owned by the local rank (or is OWNER_SHARED).
 */
void testPartitioner( int num_particles_per_rank, int ncrit, int max_depth,
                      double tolerance, int replication_depth )
{
    using namespace TreePartitionerTest;

    int rank, nprocs;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

    AoSoA_t particles( "particles", num_particles_per_rank );
    generate_test_particles( particles, num_particles_per_rank, rank );

    auto positions = Cabana::slice<Position>( particles );

    TreeBuilder<TEST_MEMSPACE, TEST_EXECSPACE> builder(
        ncrit, max_depth, MPI_COMM_WORLD, tolerance, tolerance );

    builder.build( positions, num_particles_per_rank );

    ASSERT_GT( builder.cells().size(), 0u )
        << "Tree has no cells after initial build";

    // -----------------------------------------------------------------------
    // Partition
    // -----------------------------------------------------------------------
    TreePartitioner<TEST_MEMSPACE, TEST_EXECSPACE> partitioner(
        MPI_COMM_WORLD, replication_depth );

    partitioner.partition( builder, particles, num_particles_per_rank );

    // -----------------------------------------------------------------------
    // Check 1: ownership vector aligns with cells vector
    // -----------------------------------------------------------------------
    ASSERT_EQ( partitioner.ownership().size(), builder.cells().size() )
        << "Ownership vector size does not match cells vector";

    // -----------------------------------------------------------------------
    // Check 2: every cell has a valid ownership assignment
    // -----------------------------------------------------------------------
    for ( std::size_t i = 0; i < builder.cells().size(); ++i )
    {
        const auto& c = builder.cells()[i];
        int owner = partitioner.ownership()[i].owner_rank;

        if ( owner == OWNER_SHARED )
        {
            // Only coarse layers may be replicated
            EXPECT_LE( c.depth, replication_depth )
                << "Cell at depth " << c.depth
                << " is OWNER_SHARED but exceeds replication_depth "
                << replication_depth;
        }
        else
        {
            EXPECT_GE( owner, 0 ) << "Cell " << i << " has negative owner rank";
            EXPECT_LT( owner, nprocs ) << "Cell " << i << " owner " << owner
                                       << " >= nprocs " << nprocs;
        }
    }

    // -----------------------------------------------------------------------
    // Check 3: total particle count is conserved after migration
    // -----------------------------------------------------------------------
    int new_local_count = partitioner.num_local_particles();
    EXPECT_GE( new_local_count, 0 );

    int total_after = 0;
    MPI_Allreduce( &new_local_count, &total_after, 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD );
    EXPECT_EQ( total_after, num_particles_per_rank * nprocs )
        << "Total particle count changed during migration";

    // -----------------------------------------------------------------------
    // Check 4: rebuild tree on migrated particles and confirm every local
    //          particle's leaf is owned by this rank (or is shared)
    // -----------------------------------------------------------------------
    positions = Cabana::slice<Position>( particles );
    builder.build( positions, new_local_count );

    auto h_keys = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), builder.particle_keys() );

    int bad_count = 0;
    for ( int i = 0; i < new_local_count; ++i )
    {
        MortonKey key = h_keys( i );
        int owner = partitioner.cell_owner( key );
        if ( owner != rank && owner != OWNER_SHARED )
            bad_count++;
    }

    EXPECT_EQ( bad_count, 0 )
        << "Rank " << rank << ": " << bad_count
        << " particles reside in a leaf not owned by this rank";
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//

TEST( TreePartitioner, testBasicPartition )
{
    testPartitioner( 10000, 128, 15, 0.1, 3 );
}

TEST( TreePartitioner, testSmallTree )
{
    testPartitioner( 500, 32, 10, 0.1, 2 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
